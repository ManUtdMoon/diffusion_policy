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

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.flow_match_unet_image_policy import FlowMatchUnetImagePolicy
from diffusion_policy.policy.cond_res_policy import CondResPolicy, CondSumPolicy
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
from diffusion_policy.model.vision.crop_randomizer import CropRandomizerV2
from diffusion_policy.env_runner.robomimic_image_runner import create_env
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.env.robomimic.robomimic_image_wrapper import RobomimicImageWrapper, RobomimicEarlyStopWrapper
import robomimic.utils.file_utils as FileUtils

OmegaConf.register_new_resolver("eval", eval, replace=True)

class TrainOnlineCondResWorkspace(BaseWorkspace):
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
        assert base_cfg.task_name == cfg.task_name, \
            f"Base policy task {base_cfg.task_name} does not match current task {cfg.task_name}"
        base_cfg.policy.n_action_steps = cfg.n_action_steps
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

        ## obs_emb dimensions
        # do: per-timestep feature dim; Do = To * do: full global cond dim fed to flow model
        do = self.base_policy.obs_feature_dim
        obs_emb_dim = do * cfg.n_obs_steps  # Do
        act_dim = cfg.shape_meta.action.shape[0]  # da

        ## configure cond_res policy
        self.cond_res_policy: CondResPolicy = hydra.utils.instantiate(
            cfg.cond_res_policy, obs_dim=obs_emb_dim)
        print(f"CondResPolicy: do={do}, Do(obs_emb_dim)={obs_emb_dim}, "
              f"gamma={self.cond_res_policy.gamma}")

        ## sum policy (emb_scale is owned by cond_res_policy)
        sum_policy = CondSumPolicy(
            obs_emb_dim=obs_emb_dim,
            action_dim=act_dim,
            n_action_steps=cfg.n_action_steps,
            base_policy=self.base_policy,
            cond_res_policy=self.cond_res_policy,
        )

        # configure env
        ## eval
        eval_env_runner: BaseImageRunner = hydra.utils.instantiate(
            cfg.online_task.env_runner,
            output_dir=self.output_dir)
        ## train
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
        wandb.config.update({"output_dir": self.output_dir})

        # device transfer, optimizers
        device = torch.device(cfg.training.device)
        self.base_policy.to(device)
        self.cond_res_policy.to(device)

        optimizers = self.cond_res_policy.get_optimizer(
            policy_lr=cfg.training.policy_lr,
            q_lr=cfg.training.q_lr
        )
        q_opt     = optimizers['q_optimizer']
        actor_opt = optimizers['actor_optimizer']
        alpha_opt = optimizers['alpha_optimizer']

        # replay buffer
        # obs / next_obs: obs_emb (Do,)
        # action:         res_emb  (Do,)  — same dim as obs
        dummy_obs_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_emb_dim,), dtype=np.float32
        )
        dummy_action_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_emb_dim,), dtype=np.float32
        )
        rb = ReplayBuffer(
            cfg.training.buffer_size,
            dummy_obs_space,
            dummy_action_space,
            device=device,
            n_envs=cfg.training.n_envs,
            handle_timeout_termination=False,
        )

        if cfg.training.debug:
            cfg.training.num_steps = 5000
            cfg.training.prog_explore = 1000
            cfg.training.learning_start = 1000
            cfg.training.checkpoint_every = 1000
            cfg.training.eval_every = 5000
            cfg.training.log_every = 1000

        # simplify necessary cfg
        training_freq      = cfg.training.training_freq
        log_every          = cfg.training.log_every
        eval_every         = cfg.training.eval_every
        checkpoint_every   = cfg.training.checkpoint_every
        utd                = cfg.training.utd
        n_steps            = cfg.training.num_steps
        n_envs             = cfg.online_task.n_envs
        n_updates_per_training = int(training_freq * utd)
        learning_start     = cfg.training.learning_start

        assert (
            log_every % training_freq == 0 and
            eval_every % training_freq == 0 and
            checkpoint_every % training_freq == 0 and
            learning_start % training_freq == 0 and
            eval_every % log_every == 0
        ), (f"log_every({log_every}), eval_every({eval_every}), "
            f"checkpoint_every({checkpoint_every}), learning_start({learning_start}) "
            f"must be divisible by training_freq({training_freq}).")

        # action preprocess: from action to env action
        rot_tf = None
        if cfg.online_task.abs_action:
            rot_tf = RotationTransformer('axis_angle', 'rotation_6d')

        def undo_transform_action(action):
            if rot_tf is None:
                return action
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
            set_rand_crop(False)
            sum_policy.eval()
            eval_log = eval_env_runner.run(sum_policy)
            sum_policy.train()
            set_rand_crop(True)
            logger.log(eval_log)
            wandb_run.log(eval_log, step=self.global_step)

            while self.global_step < n_steps:
                step_log = dict()
                # collect samples
                for _ in tqdm.tqdm(
                        range(training_freq // n_envs),
                        desc=f"{self.global_step} / {n_steps} samples collected",
                        leave=False):
                    self.global_step += n_envs

                    ## retrieve full obs_emb (Do,) from base policy cache
                    obs_emb_tensor = base_dict['obs_emb'].detach()  # (B, Do)

                    ## pi-dec progressive exploration
                    res_ratio = min(
                        max(self.global_step, 0) / cfg.training.prog_explore, 1)

                    ## prepare masks for progressive exploration
                    if self.global_step < learning_start:
                        # warmup: all residuals zeroed, base embedding only
                        res_masks = torch.ones(n_envs, device=device, dtype=torch.bool)
                    else:
                        # progressive: fraction (1 - res_ratio) of envs still use base only
                        res_masks = torch.rand(n_envs, device=device) >= res_ratio

                    ## forward sum policy
                    sum_dict = sum_policy.predict_train_action(
                        obs_emb_tensor,
                        res_mask=res_masks
                    )
                    sum_dict = dict_apply(sum_dict, lambda x: x.detach().cpu().numpy())
                    res_emb_np  = sum_dict['res_emb']   # (B, Do)
                    action      = sum_dict['action']    # (B, Ta, da)

                    ## env step
                    env_action = undo_transform_action(action)
                    next_obs_seq, rewards, dones, infos = envs.step(env_action)
                    for reward, done in zip(rewards, dones):
                        if done:
                            recent_done_successes.append(float(reward) > 0.9)

                    ## next obs_emb
                    assert cfg.training.bootstrap_at_done == 'never'
                    next_obs_seq_tensor = dict_apply(
                        next_obs_seq, lambda x: torch.from_numpy(x).to(device=device))
                    next_base_dict = self.base_policy.predict_action(next_obs_seq_tensor)
                    next_obs_emb_tensor = next_base_dict['obs_emb'].detach()  # (B, Do)

                    ## store transition: (obs_emb, res_emb, next_obs_emb)
                    rb.add(
                        obs=obs_emb_tensor.cpu().numpy(),
                        next_obs=next_obs_emb_tensor.cpu().numpy(),
                        action=res_emb_np,
                        reward=rewards,
                        done=dones,
                        infos=infos
                    )

                    ## advance
                    obs_seq  = next_obs_seq
                    base_dict = next_base_dict

                if self.global_step < learning_start:
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
                        critic_loss, critic_info = self.cond_res_policy.compute_critic_loss(batch, None)
                        q_opt.zero_grad()
                        critic_loss.backward()
                        q_opt.step()
                        pretrain_q_losses.append(critic_loss.item())

                        if self.global_update % cfg.training.target_freq == 0:
                            self.cond_res_policy.target_update()

                    print("Q pre-training finished.")

                    pretrain_log = {
                        'info/global_step':       self.global_step,
                        'info/global_update':     self.global_update,
                        'info/q_target':          critic_info['q_target'],
                        'info/q_predicted':       critic_info['q_predicted'],
                        'info/q_predicted_min':   critic_info['q_predicted_min'],
                        'info/q_predicted_max':   critic_info['q_predicted_max'],
                        'info/rewards':           critic_info['rewards'],
                        'info/dones':             critic_info['dones'],
                        'loss/critic_loss':       critic_loss.item() / 2.0,
                    }
                    logger.log(pretrain_log)
                    wandb_run.log(pretrain_log, step=self.global_step)
                    continue

                # training
                for _ in tqdm.tqdm(
                        range(n_updates_per_training),
                        desc=f"Update {n_updates_per_training} times",
                        leave=False):
                    self.global_update += 1

                    batch = rb.sample(cfg.training.batch_size)

                    ## update critics
                    critic_loss, critic_info = self.cond_res_policy.compute_critic_loss(batch, None)
                    q_opt.zero_grad()
                    critic_loss.backward()
                    q1_grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.cond_res_policy.qs.parameters(), cfg.training.max_grad_norm)
                    q_opt.step()

                    ## update target
                    if self.global_update % cfg.training.target_freq == 0:
                        self.cond_res_policy.target_update()

                    ## update actor
                    if self.global_update % cfg.training.policy_freq == 0:
                        actor_loss, actor_info = self.cond_res_policy.compute_actor_loss(batch)
                        actor_opt.zero_grad()
                        actor_loss.backward()
                        actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                            self.cond_res_policy.actor.parameters(),
                            cfg.training.max_grad_norm
                        )
                        actor_opt.step()

                        alpha = self.cond_res_policy.init_alpha
                        if cfg.cond_res_policy.auto_alpha:
                            alpha_loss = self.cond_res_policy.compute_alpha_loss(batch)
                            alpha_opt.zero_grad()
                            alpha_loss.backward()
                            alpha_opt.step()
                            alpha = self.cond_res_policy.log_alpha.exp().item()

                ## training metrics
                recent_done_count, recent_done_sr = get_recent_success_stats()

                step_log = {
                    'info/global_step':       self.global_step,
                    'info/global_update':     self.global_update,
                    'info/res_ratio':         res_ratio,
                    'info/q_target':          critic_info['q_target'],
                    'info/q_predicted':       critic_info['q_predicted'],
                    'info/q_predicted_min':   critic_info['q_predicted_min'],
                    'info/q_predicted_max':   critic_info['q_predicted_max'],
                    'info/actor_entropy':     actor_info['actor_entropy'],
                    'info/rewards':           critic_info['rewards'],
                    'info/dones':             critic_info['dones'],
                    'info/res_naction_rms':   actor_info['res_emb_rms'],
                    'info/base_emb_rms':      actor_info['base_emb_rms'],
                    'info/recent_done_sr':    recent_done_sr,
                    'info/recent_done_count': recent_done_count,
                    'loss/critic_loss':       critic_loss.item() / 2.0,
                    'loss/actor_loss':        actor_loss.item(),
                    'loss/q1_grad_norm':      q1_grad_norm.item(),
                    'loss/actor_grad_norm':   actor_grad_norm.item(),
                    'loss/alpha':             alpha,
                }
                if cfg.cond_res_policy.auto_alpha:
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
                    path = pathlib.Path(self.output_dir) / 'checkpoints' / f'step={self.global_step}.ckpt'
                    path.parent.mkdir(parents=False, exist_ok=True)
                    payload = {
                        'cfg': self.cfg,
                        'cond_res_policy': self.cond_res_policy.state_dict(),
                    }
                    torch.save(payload, path.open('wb'), pickle_module=dill)

        # envs.close()


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")),
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainOnlineCondResWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
