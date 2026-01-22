if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
import threading
from omegaconf import OmegaConf
import pathlib
import copy
import random
import wandb
import tqdm
import dill
import h5py
import numpy as np
import gym
import gymnasium
from collections import deque
from stable_baselines3.common.buffers import ReplayBuffer

from diffusion_policy.workspace.base_workspace import BaseWorkspace, _copy_to_cpu
from diffusion_policy.policy.flow_match_vib_unet_image_policy import FlowMatchVibUnetImagePolicy
from diffusion_policy.policy.residue_policy import ResiduePolicy
from diffusion_policy.policy.sum_policy import SumPolicy
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
from diffusion_policy.model.vision.crop_randomizer import CropRandomizerV2
from diffusion_policy.env_runner.robomimic_image_runner import create_env
from diffusion_policy.gym_util.sync_vector_env import SyncVectorEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.env.robomimic.robomimic_image_wrapper import RobomimicImageWrapper, RobomimicEarlyStopWrapper
import robomimic.utils.file_utils as FileUtils

OmegaConf.register_new_resolver("eval", eval, replace=True)

class TrainOnlineResRealExampleWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'global_update', 'base_ckpt']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

        # configure training state
        self.global_step = 0
        self.global_update = 0
        self.n_episode = 0
        self.checkpoint_thread = None

    def _save_checkpoint(self, path, payload):
        step = payload["global_step"]
        if self.checkpoint_thread is not None and self.checkpoint_thread.is_alive():
            print(f"Skipping checkpoint at step {step} because previous save is still running.")
            return

        self.checkpoint_thread = threading.Thread(
            target=self._save_worker,
            args=(path, payload, step)
        )
        self.checkpoint_thread.start()

    def _save_worker(self, path, payload, step):
        # ensure directory exists
        path.parent.mkdir(parents=False, exist_ok=True)
        
        # save checkpoint
        torch.save(payload, path.open('wb'), pickle_module=dill)
        
        # save global_step.txt
        step_path = path.parent / 'global_step.txt'
        with step_path.open('w') as f:
            f.write(str(step))

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        # configure policies
        ## load base policy
        base_payload = torch.load(open(cfg.online_task.base_ckpt, 'rb'), pickle_module=dill)
        base_cfg = base_payload['cfg']
        assert base_cfg.task_name == cfg.task_name, \
            f"Base policy task {base_cfg.task_name} does not match current task {cfg.task_name}"
        base_cfg.policy.n_action_steps = cfg.n_action_steps # may be different
        self.base_policy: FlowMatchUnetImagePolicy
        self.base_policy = hydra.utils.instantiate(base_cfg.policy)
        self.base_policy.load_state_dict(base_payload['state_dicts']['ema_model'])
        print(f"Loaded base policy from {cfg.online_task.base_ckpt}")
        self.base_policy.eval()
        self.base_policy.requires_grad_(False)

        crop_randomizers = list()
        for m in self.base_policy.modules():
            if isinstance(m, CropRandomizerV2):
                crop_randomizers.append(m)
        def set_rand_crop(mode):
            for m in crop_randomizers:
                m.force_random_crop = mode

        ## configure res policy
        To = cfg.n_obs_steps
        Ta = cfg.n_action_steps
        do = self.base_policy.obs_feature_dim
        Do = To * do  # obs chunk dim
        assert Do == do, f"Only support To == 1 for now, got To={To}"
        dz = self.base_policy.vib_latent_dim
        da = cfg.shape_meta.action.shape[0]
        Da = Ta * da  # action chunk dim

        self.res_policy: ResiduePolicy = hydra.utils.instantiate(
            cfg.res_policy, obs_dim=do, action_dim=Da)
        print(f"Residue policy with do={do}, Da={Da}, gamma={self.res_policy.gamma}")

        ## sum policy
        sum_policy = SumPolicy(
            res_scale=cfg.training.res_scale,
            obs_emb_dim=do,
            action_dim=da,
            n_action_steps=Ta,
            base_policy=self.base_policy,
            res_policy=self.res_policy
        )

        # configure env
        ## eval, only average score needed
        eval_env_runner: BaseImageRunner = hydra.utils.instantiate(
            cfg.online_task.env_runner,
            output_dir=self.output_dir)
        ## train
        ### fetch env_meta
        dataset_path = os.path.expanduser(cfg.online_task.dataset_path)
        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
        env_meta['env_kwargs']['use_object_obs'] = False
        if cfg.online_task.abs_action:
            env_meta['env_kwargs']['controller_configs']['control_delta'] = False
        
        shape_meta = cfg.shape_meta

        def env_fn():
            robomimic_env = create_env(
                env_meta=env_meta, 
                shape_meta=shape_meta
            )
            robomimic_env.env.hard_reset = False
            return MultiStepWrapper(
                RobomimicImageWrapper(
                    env=RobomimicEarlyStopWrapper(robomimic_env),
                    shape_meta=shape_meta,
                    init_state=None,
                    render_obs_key=cfg.online_task.env_runner.render_obs_key
                ),
                n_obs_steps=cfg.n_obs_steps,
                n_action_steps=cfg.n_action_steps,
                max_episode_steps=cfg.online_task.env_runner.max_steps,
                reward_agg_method='discounted_sum'
            )
        assert cfg.training.n_envs == 1, "Only support n_envs=1 for real training."
        env_fns = [env_fn] * cfg.training.n_envs
        envs = SyncVectorEnv(env_fns)

        # configure logging
        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging
        )
        wandb.config.update(
            {
                "output_dir": self.output_dir,
            }
        )

        # device transfer, optimizers
        device = torch.device(cfg.training.device)
        self.base_policy.to(device)
        self.res_policy.to(device)

        optimizers = self.res_policy.get_optimizer(
            policy_lr=cfg.training.policy_lr,
            q_lr=cfg.training.q_lr
        )
        q_opt = optimizers['q_optimizer']
        actor_opt = optimizers['actor_optimizer']
        alpha_opt = optimizers['alpha_optimizer']

        # replay buffer
        dummy_obs_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(do,), dtype=np.float32
        )
        dummy_buf_action_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(Da * 3,), dtype=np.float32
        )
        rb = ReplayBuffer(
            cfg.training.buffer_size,
            dummy_obs_space,
            dummy_buf_action_space,
            device=device,
            n_envs=cfg.training.n_envs,
            handle_timeout_termination=False,
        )

        if cfg.training.debug:
            cfg.training.num_steps = 5000
            cfg.training.prog_explore = 1000
            cfg.training.learning_start = 500
            cfg.training.checkpoint_every = 1000
            cfg.training.eval_every = 5000
            cfg.training.log_every = 1000

        # simplify necessary cfg
        training_freq = cfg.training.training_freq
        log_every = cfg.training.log_every
        eval_every = cfg.training.eval_every
        checkpoint_every = cfg.training.checkpoint_every
        utd = cfg.training.utd
        n_steps = cfg.training.num_steps
        n_envs = cfg.training.n_envs
        n_updates_per_training = int(training_freq * utd)
        learning_start = cfg.training.learning_start
        res_scale = cfg.training.res_scale

        ## check parameters for code clarity
        assert (
            log_every % training_freq == 0 and
            eval_every % training_freq == 0 and
            checkpoint_every % training_freq == 0 and
            learning_start % training_freq == 0 and
            eval_every % log_every == 0
        ), f"log_every({log_every}), eval_every({eval_every}), checkpoint_every({checkpoint_every}), learning_start({learning_start}) must be divisible by training_freq({training_freq}) for code clarity."

        # action preprocess: from action to env action
        rot_tf = None
        if cfg.online_task.abs_action:
            rot_tf = RotationTransformer('axis_angle', 'rotation_6d')

        def undo_transform_action(action):
            if rot_tf is None:
                return action

            # undo rotation transformation
            raw_shape = action.shape
            if raw_shape[-1] == 20:  # dual arm
                action = action.reshape((-1, 2, 10))
            
            d_rot = action.shape[-1] - 4
            pos = action[..., :3]
            rot = action[..., 3:3+d_rot]
            gripper = action[..., [-1]]
            rot = rot_tf.inverse(rot)
            uaction = np.concatenate([pos, rot, gripper], axis=-1)
            if raw_shape[-1] == 20:  # dual arm
                uaction = uaction.reshape((*raw_shape[:-1], 14))
            return uaction

        # training loop
        MAXLEN = 100
        recent_done_successes = deque(maxlen=MAXLEN)
        recent_done_epi_len = deque(maxlen=MAXLEN)
        def get_recent_success_stats():
            count = len(recent_done_successes)
            rate = float(np.mean(recent_done_successes)) if count > 0 else 0.0
            lens = float(np.mean(recent_done_epi_len)) if count > 0 else 0.0
            stats = {
                'count': count,
                'rate': rate,
                'len': lens,
            }
            return stats

        # resume training
        if cfg.training.resume_from is not None:
            resume_path = pathlib.Path(cfg.training.resume_from)
            latest_ckpt = resume_path / 'checkpoints' / 'latest.ckpt'
            assert latest_ckpt.exists(), f"{latest_ckpt} does not exist."

            print(f"Resuming from {latest_ckpt}")
            payload = torch.load(open(latest_ckpt, 'rb'), pickle_module=dill)

            # load state
            self.res_policy.load_state_dict(payload['res_policy'])
            q_opt.load_state_dict(payload['q_optimizer'])
            actor_opt.load_state_dict(payload['actor_optimizer'])
            alpha_opt.load_state_dict(payload['alpha_optimizer'])
            self.global_step = payload['global_step']
            self.global_update = payload['global_update']
            self.n_episode = payload['n_episode']
            recent_done_successes = deque(payload['recent_done_successes'], maxlen=MAXLEN)
            recent_done_epi_len = deque(payload['recent_done_epi_len'], maxlen=MAXLEN)
            rb = payload.get('replay_buffer', rb)  # in case of no buffer saved

            print(f"Resumed at global_step={self.global_step}")

        obs_seq = envs.reset()
        obs_seq_tensor = dict_apply(
            obs_seq, lambda x: torch.from_numpy(x).to(device=device))
        base_dict = self.base_policy.predict_action(obs_seq_tensor)
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as logger:
            set_rand_crop(False)
            sum_policy.eval()
            eval_log = eval_env_runner.run(sum_policy)
            sum_policy.train()
            set_rand_crop(True)
            logger.log(eval_log)
            wandb_run.log(eval_log, step=self.global_step)

            try:
                while self.global_step < n_steps:
                    step_log = dict()
                    # collect samples
                    for _ in tqdm.tqdm(
                            range(training_freq // n_envs), 
                            desc=f"{self.global_step} / {n_steps} samples collected",
                            leave=False):
                        self.global_step += n_envs

                        ## retrieve from base cache
                        obs_emb_tensor = base_dict['obs_emb'][:, -do:].detach()
                        base_naction_tensor = base_dict['naction'].detach()
                        base_naction_flat = base_naction_tensor.flatten(start_dim=1).cpu().numpy()  # (B, Ta*da)

                        ## pi-dec progressive exploration
                        res_ratio = min(
                            max(self.global_step, 0) / cfg.training.prog_explore, 1)
                        ## uncomment to disable progressive exploration
                        res_ratio = 1.0

                        ## prepare masks for progressive exploration
                        if self.global_step < learning_start:
                            # Mask all residues during warmup phase: base only
                            res_masks = torch.ones(n_envs, device=device, dtype=torch.bool)
                        else:
                            # Progressive exploration: pi-dec style
                            res_masks = torch.rand(n_envs, device=device) >= res_ratio

                        ## forward sum policy with cached base info
                        sum_dict = sum_policy.predict_train_action(
                            base_naction_tensor,
                            obs_emb_tensor,
                            res_mask=res_masks
                        )
                        sum_dict = dict_apply(
                            sum_dict, lambda x: x.detach().cpu().numpy())
                        res_naction_flat = sum_dict['res_naction_flat']
                        action = sum_dict['action']

                        ## env_action and step
                        env_action = undo_transform_action(action)
                        next_obs_seq, rewards, dones, infos = envs.step(env_action)
                        for reward, done, info in zip(rewards, dones, infos):
                            if done:
                                recent_done_successes.append(float(reward) > 0.9)
                                recent_done_epi_len.append(info['episode_length'])
                                self.n_episode += 1

                        ## prepare transitions for rb
                        assert cfg.training.bootstrap_at_done == 'never'
                        next_obs_seq_tensor = dict_apply(
                            next_obs_seq, lambda x: torch.from_numpy(x).to(device=device))
                        next_base_dict = self.base_policy.predict_action(next_obs_seq_tensor)
                        next_obs_emb_tensor = next_base_dict['obs_emb'][:, -do:].detach()
                        next_base_naction_tensor = next_base_dict['naction'].detach()
                        actions_to_save = np.concatenate(
                            [
                                res_naction_flat,
                                base_naction_flat,
                                next_base_naction_tensor.flatten(start_dim=1).cpu().numpy()
                            ],
                            axis=-1
                        )

                        rb.add(
                            obs=obs_emb_tensor.cpu().numpy(),
                            next_obs=next_obs_emb_tensor.cpu().numpy(),
                            action=actions_to_save,
                            reward=rewards,
                            done=dones,
                            infos=infos
                        )

                        ## switch to next step
                        base_dict = next_base_dict

                        # ensure the global_step aligns with training_freq
                        if self.global_step % training_freq == 0:
                            break

                    if self.global_step < learning_start:
                        # Warmup phase: skip training, only collect data
                        continue

                    # training
                    for _ in tqdm.tqdm(
                            range(n_updates_per_training),
                            desc=f"Update {n_updates_per_training} times",
                            leave=False):
                        self.global_update += 1

                        ## fetch data
                        batch = rb.sample(cfg.training.batch_size)

                        ## update critics
                        critic_loss, critic_info = self.res_policy.compute_critic_loss(batch, None)
                        q_opt.zero_grad()
                        critic_loss.backward()
                        q1_grad_norm = torch.nn.utils.clip_grad_norm_(
                            self.res_policy.qs.parameters(), cfg.training.max_grad_norm)
                        q_opt.step()

                        ## update target
                        if self.global_update % cfg.training.target_freq == 0:
                            self.res_policy.target_update()
                        
                        ## update policy
                        if self.global_update % cfg.training.policy_freq == 0:
                            actor_loss, actor_info = self.res_policy.compute_actor_loss(batch)
                            actor_opt.zero_grad()
                            actor_loss.backward()
                            actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                                self.res_policy.actor.parameters(),
                                cfg.training.max_grad_norm
                            )
                            actor_opt.step()

                            alpha = self.res_policy.init_alpha
                            if cfg.res_policy.auto_alpha:
                                alpha_loss = self.res_policy.compute_alpha_loss(batch)
                                alpha_opt.zero_grad()
                                alpha_loss.backward()
                                alpha_opt.step()
                                alpha = self.res_policy.log_alpha.exp().item()

                    ## training metrics
                    stats = get_recent_success_stats()
                    recent_done_count = stats['count']
                    recent_done_sr = stats['rate']
                    recent_done_avg_len = stats['len']

                    step_log = {
                        'info/global_step': self.global_step,
                        'info/global_update': self.global_update,
                        'info/n_episode': self.n_episode,

                        'info/res_ratio': res_ratio,
                        'info/q_target': critic_info['q_target'],
                        'info/q_predicted': critic_info['q_predicted'],
                        'info/q_predicted_min': critic_info['q_predicted_min'],
                        'info/q_predicted_max': critic_info['q_predicted_max'],
                        'info/actor_entropy': actor_info['actor_entropy'],
                        'info/rewards': critic_info['rewards'],
                        'info/dones': critic_info['dones'],
                        'info/res_naction_norm': actor_info['res_naction_norm'],
                        'info/base_naction_norm': actor_info['base_naction_norm'],
                        'info/recent_done_sr': recent_done_sr,
                        'info/recent_done_count': recent_done_count,
                        'info/recent_done_avg_len': recent_done_avg_len,

                        'loss/critic_loss': critic_loss.item() / 2.0,
                        'loss/actor_loss': actor_loss.item(),
                        'loss/q1_grad_norm': q1_grad_norm.item(),
                        'loss/actor_grad_norm': actor_grad_norm.item(),
                        'loss/alpha': alpha,
                    }
                    if cfg.res_policy.auto_alpha:
                        step_log['loss/alpha_loss'] = alpha_loss.item()

                    # evaluation
                    sum_policy.eval()
                    if self.global_step > 0 and self.global_step % eval_every == 0:
                        set_rand_crop(False)
                        eval_log = eval_env_runner.run(sum_policy)
                        step_log.update(eval_log)
                        set_rand_crop(True)
                    sum_policy.train()

                    # logging
                    logger.log(step_log)
                    if self.global_step % log_every == 0:
                        wandb_run.log(step_log, step=self.global_step)

                    # checkpointing
                    if self.global_step % checkpoint_every == 0:
                        path = pathlib.Path(self.output_dir) / 'checkpoints' / 'latest.ckpt'
                        
                        # prepare payload in main thread to avoid race conditions on model/optimizers
                        # ReplayBuffer is NOT saved in periodic checkpoints to save time
                        payload = {
                            'cfg': self.cfg,
                            'res_policy': _copy_to_cpu(self.res_policy.state_dict()),
                            'q_optimizer': _copy_to_cpu(q_opt.state_dict()),
                            'actor_optimizer': _copy_to_cpu(actor_opt.state_dict()),
                            'alpha_optimizer': _copy_to_cpu(alpha_opt.state_dict()),
                            'global_step': self.global_step,
                            'global_update': self.global_update,
                            'n_episode': self.n_episode,
                            'recent_done_successes': list(recent_done_successes),
                            'recent_done_epi_len': list(recent_done_epi_len),
                            'replay_buffer': None # SKIP buffer
                        }
                        self._save_checkpoint(path, payload)

            except KeyboardInterrupt:
                print("\nKeyboard Interrupt! Saving full checkpoint with ReplayBuffer...")
                # wait for any background save to finish to avoid corruption
                if self.checkpoint_thread is not None and self.checkpoint_thread.is_alive():
                    print("Waiting for background save to finish...")
                    self.checkpoint_thread.join()
                
                path = pathlib.Path(self.output_dir) / 'checkpoints' / 'latest.ckpt'
                payload = {
                    'cfg': self.cfg,
                    'res_policy': _copy_to_cpu(self.res_policy.state_dict()),
                    'q_optimizer': _copy_to_cpu(q_opt.state_dict()),
                    'actor_optimizer': _copy_to_cpu(actor_opt.state_dict()),
                    'alpha_optimizer': _copy_to_cpu(alpha_opt.state_dict()),
                    'global_step': self.global_step,
                    'global_update': self.global_update,
                    'n_episode': self.n_episode,
                    'recent_done_successes': list(recent_done_successes),
                    'recent_done_epi_len': list(recent_done_epi_len),
                    'replay_buffer': rb # INCLUDE buffer
                }
                self._save_worker(path, payload, self.global_step)
                # re-raise to exit
                raise

        envs.close()


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")),
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainOnlineResRealExampleWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
