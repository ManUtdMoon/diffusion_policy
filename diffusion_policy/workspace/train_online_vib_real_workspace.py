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
from diffusion_policy.policy.latent_policy import ResiduePolicy, SumPolicy
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.vision.crop_randomizer import CropRandomizerV2
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.env.juicing.juicing_env import JuicingEnv
from diffusion_policy.env.flip.flip_env import FlipEnv
from diffusion_policy.env.box.box_env import BoxEnv


OmegaConf.register_new_resolver("eval", eval, replace=True)

class TrainOnlineVibRealWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'global_update', 'base_ckpt']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

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
        self.base_policy: FlowMatchVibUnetImagePolicy
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
        set_rand_crop(True)

        ## configure res policy
        To = cfg.n_obs_steps
        Ta = cfg.n_action_steps
        do = self.base_policy.obs_feature_dim
        Do = To * do  # obs chunk dim
        dz = self.base_policy.vib_latent_dim
        da = cfg.shape_meta.action.shape[0]
        Da = Ta * da  # action chunk dim

        self.res_policy: ResiduePolicy = hydra.utils.instantiate(
            cfg.res_policy, obs_dim=Do, z_dim=dz, action_dim=Da)
        print(f"Residue policy with Do={Do}, Dz={dz}, Da={Da}, gamma={self.res_policy.gamma}")

        ## sum policy
        sum_policy = SumPolicy(
            res_scale=cfg.training.res_scale,
            base_policy=self.base_policy,
            res_policy=self.res_policy
        )
        sum_policy.train()

        # configure env
        def env_fn():
            env = BoxEnv(smooth=True)
            return MultiStepWrapper(
                env,
                n_obs_steps=To,
                n_action_steps=Ta,
                max_episode_steps=cfg.online_task.max_steps,
                reward_agg_method='discounted_sum'
            )

        assert cfg.training.n_envs == 1, "Only support n_envs=1 for real training."
        envs = env_fn()

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
            shape=(Do + 3 * dz,), dtype=np.float32
        )  # obs_emb + z_mean + z_logvar + z
        dummy_buf_action_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(dz,), dtype=np.float32
        )  # res z
        rb = ReplayBuffer(
            cfg.training.buffer_size,
            dummy_obs_space,
            dummy_buf_action_space,
            device=device,
            n_envs=cfg.training.n_envs,
            handle_timeout_termination=False,
        )

        if cfg.training.debug:
            cfg.training.num_steps = 2000
            cfg.training.learning_start = 500
            cfg.training.checkpoint_every = 1000
            cfg.training.log_every = 100

        # simplify necessary cfg
        training_freq = cfg.training.training_freq
        log_every = cfg.training.log_every
        checkpoint_every = cfg.training.checkpoint_every
        utd = cfg.training.utd
        n_steps = cfg.training.num_steps
        n_envs = cfg.training.n_envs
        n_updates_per_training = int(training_freq * utd)
        learning_start = cfg.training.learning_start

        ## check parameters for code clarity
        assert (
            log_every % training_freq == 0 and
            checkpoint_every % training_freq == 0 and
            learning_start % training_freq == 0
        ), f"log_every({log_every}), checkpoint_every({checkpoint_every}), learning_start({learning_start}) must be divisible by training_freq({training_freq}) for code clarity."

        # training loop
        MAXLEN = 40
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
            latest_ckpt = resume_path
            assert latest_ckpt.exists(), f"{latest_ckpt} does not exist."

            print(f"Resuming from {latest_ckpt}")
            input("Do you specify the correct resume_from path? Press Enter to continue...")
            payload = torch.load(open(latest_ckpt, 'rb'), pickle_module=dill)

            # load state
            self.res_policy.load_state_dict(payload['res_policy'])
            q_opt.load_state_dict(payload['q_optimizer'])
            actor_opt.load_state_dict(payload['actor_optimizer'])
            alpha_opt.load_state_dict(payload['alpha_optimizer'])
            self.global_step = payload['global_step']
            self.global_update = payload['global_update']
            self.n_episode = payload.get('n_episode', 0)
            recent_done_successes = deque(payload.get('recent_done_successes', []), maxlen=MAXLEN)
            recent_done_epi_len = deque(payload.get('recent_done_epi_len', []), maxlen=MAXLEN)
            rb = payload.get('replay_buffer', rb)  # in case of no buffer saved

            print(f"Resumed at global_step={self.global_step}")
        else:
            input("No resume_from specified. Start training from scratch? Press Enter to continue...")

        obs_seq = envs.reset()
        obs_seq_tensor = dict_apply(
            obs_seq,
            lambda x: torch.from_numpy(x).to(device=device).unsqueeze(0)
        )
        with torch.no_grad():
            obs_emb_tensor = self.base_policy.encode_obs(obs_seq_tensor)
            _, z_mean, z_logvar, z = self.base_policy.vib_forward(obs_emb_tensor)
            obs_z = torch.cat([obs_emb_tensor, z_mean, z_logvar, z], dim=-1)
        log_path = os.path.join(self.output_dir, 'logs.json.txt')

        with JsonLogger(log_path) as logger:
            try:
                while self.global_step < n_steps:
                    step_log = dict()
                    # collect samples
                    for _ in tqdm.tqdm(
                            range(training_freq // n_envs), 
                            desc=f"{self.global_step} / {n_steps} samples collected",
                            leave=False):
                        self.global_step += n_envs

                        ## prepare masks for progressive exploration
                        if self.global_step < learning_start:
                            perturb = True  # T, F
                        else:
                            perturb = True

                        ## forward sum policy
                        sum_dict = sum_policy.predict_train_action(obs_z, perturb)
                        sum_dict = dict_apply(
                            sum_dict, lambda x: x.detach())
                        res_z = sum_dict['res_z']
                        action = sum_dict['action'].cpu().numpy()

                        ## env_action and step
                        assert action.shape == (1, Ta, da), \
                            f"Action shape {action.shape} does not match expected {(1, Ta, da)}"
                        next_obs_seq, reward, done, infos = envs.step(action.squeeze(0))
                        if done:
                            if reward > 0.5:
                                assert np.any(infos['is_success']), "Done with reward but is_success not marked."
                            recent_done_successes.append(float(reward) > 0.5)
                            recent_done_epi_len.append(infos['episode_length'])
                            self.n_episode += 1
                            # Run post-processing before resetting the episode.
                            envs.reset_end()
                            next_obs_seq = envs.reset()

                        ## prepare transitions for rb
                        assert cfg.training.bootstrap_at_done == 'never'
                        next_obs_seq_tensor = dict_apply(
                            next_obs_seq,
                            lambda x: torch.from_numpy(x).to(device=device).unsqueeze(0)
                        )
                        with torch.no_grad():
                            next_obs_emb_tensor = self.base_policy.encode_obs(next_obs_seq_tensor).detach()
                            _, next_z_mean, next_z_logvar, next_z = self.base_policy.vib_forward(next_obs_emb_tensor)
                            next_obs_z = torch.cat([next_obs_emb_tensor, next_z_mean, next_z_logvar, next_z], dim=-1)

                        rb.add(
                            obs=obs_z.detach().cpu().numpy(),
                            next_obs=next_obs_z.detach().cpu().numpy(),
                            action=res_z.cpu().numpy(),
                            reward=np.array([reward]),
                            done=np.array([done]),
                            infos=[{}]
                        )

                        ## switch to next step
                        obs_z = next_obs_z

                        # ensure the global_step aligns with training_freq
                        if self.global_step % training_freq == 0:
                            break

                    if self.global_step < learning_start:
                        # Warmup phase: skip training, only collect data
                        continue

                    # training
                    for _ in range(n_updates_per_training):
                        self.global_update += 1

                        ## fetch data
                        batch = rb.sample(cfg.training.batch_size)

                        ## update critics
                        critic_loss, critic_info = self.res_policy.compute_critic_loss(batch)
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

                        'info/q_target': critic_info['q_target'],
                        'info/q_predicted': critic_info['q_predicted'],
                        'info/q_predicted_min': critic_info['q_predicted_min'],
                        'info/q_predicted_max': critic_info['q_predicted_max'],
                        'info/actor_entropy': actor_info['actor_entropy'],
                        'info/rewards': critic_info['rewards'],
                        'info/dones': critic_info['dones'],
                        'info/res_naction_norm': actor_info['res_z_norm'],
                        'info/z_mean_norm': actor_info['z_mean_norm'],
                        'info/z_norm': actor_info['z_norm'],
                        'info/z_mean_rms': actor_info['z_mean_rms'],
                        'info/z_rms': actor_info['z_rms'],
                        'info/res_z_rms': actor_info['res_z_rms'],
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

                    # logging
                    logger.log(step_log)
                    if self.global_step % log_every == 0:
                        wandb_run.log(step_log, step=self.global_step)

                    # checkpointing
                    if self.global_step % checkpoint_every == 0:
                        path = pathlib.Path(self.output_dir) / 'checkpoints' / f'step-{self.global_step}.ckpt'

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
                            'replay_buffer': rb
                        }
                        self._save_checkpoint(path, payload)
            # catch all kinds of interrupts to ensure we save a final checkpoint with replay buffer
            except (KeyboardInterrupt, Exception) as e:
                print(f"\nException {type(e).__name__} occurred: {e}. Saving full checkpoint with ReplayBuffer...")
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
    workspace = TrainOnlineVibRealWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
