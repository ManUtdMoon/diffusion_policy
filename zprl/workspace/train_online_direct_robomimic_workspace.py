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
from omegaconf import OmegaConf
import pathlib
import copy
import random
import wandb
import tqdm
import dill
import numpy as np
import gymnasium
import collections
from stable_baselines3.common.buffers import ReplayBuffer

from zprl.workspace.base_workspace import BaseWorkspace
from zprl.policy.flow_match_vib_unet_image_policy import FlowMatchVibUnetImagePolicy
from zprl.policy.direct_policy import DirectPolicy, DirectActionPolicy
from zprl.common.json_logger import JsonLogger
from zprl.common.pytorch_util import dict_apply
from zprl.model.common.rotation_transformer import RotationTransformer
from zprl.model.vision.crop_randomizer import CropRandomizerV2
from zprl.env_runner.robomimic_image_runner import create_env
from zprl.gym_util.async_vector_env import AsyncVectorEnv
from zprl.gym_util.multistep_wrapper import MultiStepWrapper
from zprl.env.robomimic.robomimic_image_wrapper import RobomimicImageWrapper, RobomimicEarlyStopWrapper
import robomimic.utils.file_utils as FileUtils

OmegaConf.register_new_resolver("eval", eval, replace=True)


class TrainOnlineDirectRobomimicWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'global_update', 'base_ckpt']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # configure training state
        self.global_step = 0
        self.global_update = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        # configure policies
        ## load base policy
        base_payload = torch.load(open(cfg.online_task.base_ckpt, 'rb'), pickle_module=dill)
        base_cfg = base_payload['cfg']
        # patch legacy checkpoint configs that still use the old package name
        base_cfg_yaml = OmegaConf.to_yaml(base_cfg)
        if 'diffusion_policy' in base_cfg_yaml:
            base_cfg_yaml = base_cfg_yaml.replace('diffusion_policy', 'zprl')
            base_cfg = OmegaConf.create(base_cfg_yaml)
        assert base_cfg.task_name == cfg.task_name, \
            f"Base policy task {base_cfg.task_name} does not match current task {cfg.task_name}"
        base_cfg.policy.n_action_steps = cfg.n_action_steps # may be different
        base_cfg.policy.num_inference_steps = cfg.num_inference_steps
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

        ## configure direct policy
        To = cfg.n_obs_steps
        Ta = cfg.n_action_steps
        do = self.base_policy.obs_feature_dim
        Do = To * do  # obs chunk dim
        da = cfg.shape_meta.action.shape[0]
        Da = Ta * da  # action chunk dim

        self.direct_policy: DirectPolicy = hydra.utils.instantiate(
            cfg.direct_policy, obs_dim=Do, action_dim=Da)
        print(f"Direct policy with Do={Do}, Da={Da}, gamma={self.direct_policy.gamma}")

        ## direct action policy
        direct_action_policy = DirectActionPolicy(
            obs_emb_dim=Do,
            action_dim=da,
            n_action_steps=cfg.n_action_steps,
            base_policy=self.base_policy,
            direct_policy=self.direct_policy
        )

        # configure env
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
        def dummy_env_fn():
            robomimic_env = create_env(
                env_meta=env_meta, 
                shape_meta=shape_meta,
                enable_render=False
            )
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
        env_fns = [env_fn] * cfg.training.n_envs
        envs = AsyncVectorEnv(env_fns, dummy_env_fn=dummy_env_fn)

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
        self.direct_policy.to(device)

        optimizers = self.direct_policy.get_optimizer(
            policy_lr=cfg.training.policy_lr,
            q_lr=cfg.training.q_lr
        )
        q_opt = optimizers['q_optimizer']
        actor_opt = optimizers['actor_optimizer']

        # replay buffer
        dummy_obs_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(Do,), dtype=np.float32
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

        # offline demo preload
        if cfg.training.preload_offline_data:
            raise NotImplementedError(
                "Direct policy does not support preload_offline_data yet. "
                "The existing preload helper stores residual actions."
            )

        if cfg.training.debug:
            cfg.training.num_steps = 5000
            cfg.training.prog_explore = 1000
            cfg.training.learning_start = 1000
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
        n_envs = cfg.online_task.n_envs
        n_updates_per_training = int(training_freq * utd)
        learning_start = cfg.training.learning_start
        fixed_std = cfg.direct_policy.fixed_std

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
        recent_done_successes = collections.deque(maxlen=100)
        def get_recent_success_stats():
            count = len(recent_done_successes)
            rate = float(np.mean(recent_done_successes)) if count > 0 else 0.0
            return count, rate

        obs_seq = envs.reset()
        obs_seq_tensor = dict_apply(
            obs_seq, lambda x: torch.from_numpy(x).to(device=device))
        base_dict = self.base_policy.predict_action(obs_seq_tensor)
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as logger:
            # set_rand_crop(False)
            # sum_policy.eval()
            # eval_log = eval_env_runner.run(sum_policy)
            # sum_policy.train()
            set_rand_crop(True)
            # logger.log(eval_log)
            # wandb_run.log(eval_log, step=self.global_step)

            while self.global_step < n_steps:
                step_log = dict()
                # collect samples
                for _ in tqdm.tqdm(
                        range(training_freq // n_envs), 
                        desc=f"{self.global_step} / {n_steps} samples collected",
                        leave=False):
                    self.global_step += n_envs

                    ## retrieve from base cache
                    obs_emb_tensor = base_dict['obs_emb'].detach()
                    base_naction_tensor = base_dict['naction'].detach()
                    base_naction_flat = base_naction_tensor.flatten(start_dim=1).cpu().numpy()  # (B, Ta*da)

                    ## direct policy progressive exploration
                    action_ratio = min(
                        max(self.global_step, 0) / cfg.training.prog_explore, 1)

                    ## prepare masks for progressive exploration
                    if self.global_step < learning_start:
                        # Mask all direct actions during warmup phase: base only
                        action_masks = torch.ones(n_envs, device=device, dtype=torch.bool)
                    else:
                        action_masks = torch.rand(n_envs, device=device) >= action_ratio

                    ## forward direct policy with cached base info
                    direct_dict = direct_action_policy.predict_train_action(
                        base_naction_tensor,
                        obs_emb_tensor,
                        stddev=fixed_std,
                        action_mask=action_masks
                    )
                    direct_dict = dict_apply(
                        direct_dict, lambda x: x.detach().cpu().numpy())
                    naction_flat = direct_dict['naction_flat']
                    action = direct_dict['action']

                    ## env_action and step
                    env_action = undo_transform_action(action)
                    next_obs_seq, rewards, dones, infos = envs.step(env_action)
                    for reward, done in zip(rewards, dones):
                        if done:
                            recent_done_successes.append(float(reward) > 0.9)

                    ## prepare transitions for rb
                    ## because we do not bootstrap at done, we can use next_obs_seq directly
                    assert cfg.training.bootstrap_at_done == 'never'
                    next_obs_seq_tensor = dict_apply(
                        next_obs_seq, lambda x: torch.from_numpy(x).to(device=device))
                    next_base_dict = self.base_policy.predict_action(next_obs_seq_tensor)
                    next_obs_emb_tensor = next_base_dict['obs_emb'].detach()
                    next_base_naction_tensor = next_base_dict['naction'].detach()
                    actions_to_save = np.concatenate(
                        [
                            naction_flat,
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
                    obs_seq = next_obs_seq
                    base_dict = next_base_dict

                if self.global_step < learning_start:
                    # Warmup phase: skip training, only collect data
                    continue

                # Q pre-training
                if (
                    self.global_step == learning_start and
                    cfg.training.q_pretrain_steps > 0
                ):
                    print("Q pre-training starts...")
                    pretrain_q_losses = []
                    for _ in tqdm.tqdm(
                        range(cfg.training.q_pretrain_steps),
                        desc=f"Q pre-training for {cfg.training.q_pretrain_steps} steps."
                    ):
                        self.global_update += 1
                        batch = rb.sample(cfg.training.batch_size)
                        critic_loss, critic_info = self.direct_policy.compute_critic_loss(batch, fixed_std)
                        q_opt.zero_grad()
                        critic_loss.backward()
                        q_opt.step()
                        pretrain_q_losses.append(critic_loss.item())

                        if self.global_update % cfg.training.target_freq == 0:
                            self.direct_policy.target_update()
                    
                    print("Q pre-training finished.")
                    
                    # Log pre-training metrics
                    pretrain_log = {
                        'info/global_step': self.global_step,
                        'info/global_update': self.global_update,

                        'info/q_target': critic_info['q_target'],
                        'info/q_predicted': critic_info['q_predicted'],
                        'info/q_predicted_min': critic_info['q_predicted_min'],
                        'info/q_predicted_max': critic_info['q_predicted_max'],
                        'info/rewards': critic_info['rewards'],
                        'info/dones': critic_info['dones'],

                        'loss/critic_loss': critic_loss.item() / cfg.direct_policy.num_qs,
                    }
                    logger.log(pretrain_log)
                    wandb_run.log(pretrain_log, step=self.global_step)

                    continue  # pretrain Q only

                # training
                for _ in tqdm.tqdm(
                        range(n_updates_per_training),
                        desc=f"Update {n_updates_per_training} times",
                        leave=False):
                    self.global_update += 1

                    ## fetch data
                    batch = rb.sample(cfg.training.batch_size)

                    ## update critics
                    critic_loss, critic_info = self.direct_policy.compute_critic_loss(batch, fixed_std)
                    q_opt.zero_grad()
                    critic_loss.backward()
                    q1_grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.direct_policy.qs.parameters(), cfg.training.max_grad_norm)
                    q_opt.step()

                    ## update target
                    if self.global_update % cfg.training.target_freq == 0:
                        self.direct_policy.target_update()
                    
                    ## update policy
                    if self.global_update % cfg.training.policy_freq == 0:
                        actor_loss, actor_info = self.direct_policy.compute_actor_loss(batch, fixed_std)
                        actor_opt.zero_grad()
                        actor_loss.backward()
                        actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                            self.direct_policy.actor.parameters(),
                            cfg.training.max_grad_norm
                        )
                        actor_opt.step()

                ## training metrics
                recent_done_count, recent_done_sr = get_recent_success_stats()

                step_log = {
                    'info/global_step': self.global_step,
                    'info/global_update': self.global_update,

                    'info/res_ratio': action_ratio,
                    'info/q_target': critic_info['q_target'],
                    'info/q_predicted': critic_info['q_predicted'],
                    'info/q_predicted_min': critic_info['q_predicted_min'],
                    'info/q_predicted_max': critic_info['q_predicted_max'],
                    'info/rewards': critic_info['rewards'],
                    'info/dones': critic_info['dones'],

                    'info/n_rms': actor_info['n_rms'],
                    'info/res_n_rms': actor_info['delta_n_rms'],
                    'info/delta_mean_rms': actor_info['delta_mean_rms'],
                    'info/base_n_rms': actor_info['base_n_rms'],
                    'info/recent_done_sr': recent_done_sr,
                    'info/recent_done_count': recent_done_count,

                    'loss/critic_loss': critic_loss.item() / cfg.direct_policy.num_qs,
                    'loss/actor_loss': actor_loss.item(),
                    'loss/actor_rl_loss': actor_info['actor_rl_loss'],
                    'loss/actor_bc_loss': actor_info['bc_loss'],
                    'loss/q1_grad_norm': q1_grad_norm.item(),
                    'loss/actor_grad_norm': actor_grad_norm.item(),
                }

                # evaluation
                # sum_policy.eval()
                # if self.global_step > 0 and self.global_step % eval_every == 0:
                #     set_rand_crop(False)
                #     eval_log = eval_env_runner.run(sum_policy)
                #     step_log.update(eval_log)
                #     set_rand_crop(True)
                direct_action_policy.train()

                # logging
                logger.log(step_log)
                if self.global_step % log_every == 0:
                    wandb_run.log(step_log, step=self.global_step)

                # checkpointing
                # if self.global_step % checkpoint_every == 0:
                #     path = pathlib.Path(self.output_dir) / 'checkpoints' / f'step_{self.global_step}.ckpt'
                #     path.parent.mkdir(parents=False, exist_ok=True)
                #     payload = {
                #         'cfg': self.cfg,
                #         'direct_policy': self.direct_policy.state_dict(),
                #     }
                #     torch.save(payload, path.open('wb'), pickle_module=dill)


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")),
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainOnlineDirectRobomimicWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
