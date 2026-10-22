import collections
import copy
import os
import random

import gymnasium
import hydra
import numpy as np
import torch
import tqdm
import wandb
from omegaconf import OmegaConf

from zprl.common.json_logger import JsonLogger
from zprl.common.oneleg_online import load_oneleg_base, oneleg_online_snapshot
from zprl.common.oneleg_episode_replay_buffer import OneLegEpisodeReplayBuffer, OneLegEpisodeCollector
from zprl.env.furniturebench.oneleg_online_env import OneLegOnlineEnv
from zprl.policy.oneleg_history_policy import OneLegHistoryNoiseSumPolicy
from zprl.workspace.base_workspace import BaseWorkspace


class TrainOnlineReactiveHistoryNoiseOneLegWorkspace(BaseWorkspace):
    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir)
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.allow_tf32 = True
        self.global_step = 0
        self.global_update = 0
        self.num_episodes = 0

    def _update(self, cfg, optimizers):
        policy = self.noise_policy
        metrics = {}
        for _ in tqdm.tqdm(
                range(int(cfg.training.training_freq * cfg.training.utd)),
                desc='Updating', unit='update', mininterval=cfg.training.tqdm_interval_sec,
                leave=False, position=0):
            self.global_update += 1
            batch = self.replay.sample(cfg.training.batch_size)
            critic_loss, critic_info = policy.compute_critic_loss(batch)
            optimizers['q_optimizer'].zero_grad()
            critic_loss.backward()
            q_norm = torch.nn.utils.clip_grad_norm_(policy.qs.parameters(), cfg.training.max_grad_norm)
            optimizers['q_optimizer'].step()
            if self.global_update % cfg.training.target_freq == 0:
                policy.target_update()
            metrics.update({f'info/{key}': value.item() for key, value in critic_info.items()})
            metrics.update({'loss/critic_loss': critic_loss.item() / policy.num_qs,
                            'loss/q1_grad_norm': q_norm.item()})
            if self.global_update % cfg.training.policy_freq == 0:
                actor_loss, actor_info = policy.compute_actor_loss(batch)
                optimizers['actor_optimizer'].zero_grad()
                actor_loss.backward()
                actor_norm = torch.nn.utils.clip_grad_norm_(policy.actor.parameters(), cfg.training.max_grad_norm)
                optimizers['actor_optimizer'].step()
                metrics.update({f'info/{key}': value.item() for key, value in actor_info.items()})
                metrics.update({'loss/actor_loss': actor_loss.item(),
                                'loss/actor_grad_norm': actor_norm.item()})
                if policy.auto_alpha:
                    alpha_loss = policy.compute_alpha_loss(batch)
                    optimizers['alpha_optimizer'].zero_grad()
                    alpha_loss.backward()
                    optimizers['alpha_optimizer'].step()
                    metrics['loss/alpha_loss'] = alpha_loss.item()
            metrics['loss/alpha'] = (policy.log_alpha.detach().exp().item()
                                     if policy.auto_alpha else policy.init_alpha)
        return metrics

    def get_snapshot(self):
        return oneleg_online_snapshot(self.cfg, 'dsrl', self.noise_policy,
            self.global_step, self.global_update, self.num_episodes)

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        device = torch.device(cfg.training.device)
        self.base_policy, _ = load_oneleg_base(
            cfg.online_task.base_ckpt, device=device, num_inference_steps=cfg.num_inference_steps)
        obs_dim = self.base_policy.obs_feature_dim * cfg.n_obs_steps
        action_dim = cfg.shape_meta.action.shape[0]
        exec_dim = cfg.n_action_steps * action_dim
        self.noise_policy = hydra.utils.instantiate(
            cfg.noise_policy, obs_dim=obs_dim, action_dim=cfg.n_noise_steps * action_dim, exec_dim=exec_dim).to(device)
        self.base_ckpt = cfg.online_task.base_ckpt
        sum_policy = OneLegHistoryNoiseSumPolicy(
            noise_scale=cfg.training.noise_scale,
            base_obs_emb_dim=obs_dim, subtask_dim=0,
            action_dim=action_dim,
            n_action_steps=cfg.n_action_steps,
            horizon=cfg.horizon,
            n_obs_steps=cfg.n_obs_steps,
            n_noise_steps=cfg.n_noise_steps,
            base_policy=self.base_policy,
            noise_policy=self.noise_policy
        )
        optimizers = self.noise_policy.get_optimizer(
            cfg.training.policy_lr, cfg.training.q_lr, cfg.training.enc_lr)
        if cfg.training.compile:
            for module in (self.noise_policy.actor, self.noise_policy.qs, self.noise_policy.q_targets):
                module.forward = torch.compile(module.forward, fullgraph=True, dynamic=False)
        self.replay = OneLegEpisodeReplayBuffer(
            cfg.training.buffer_size,
            gymnasium.spaces.Box(-np.inf, np.inf, (obs_dim,), dtype=np.float32),
            gymnasium.spaces.Box(-np.inf, np.inf, (cfg.n_noise_steps * action_dim,), dtype=np.float32),
            device=device, method='dsrl',
            handle_timeout_termination=cfg.training.bootstrap_at_done == 'truncated',
            n_steps=cfg.training.n_step, gamma=self.noise_policy.gamma,
            exec_dim=exec_dim, action_step_dim=action_dim,
            history_length=cfg.noise_policy.history_length)
        assert cfg.training.bootstrap_at_done in ('never', 'truncated')

        # prepare helpers
        collector = self.collector = OneLegEpisodeCollector(
            self.replay, cfg.training.n_envs)
        next_train_step = None
        recent_success = collections.deque(maxlen=100)
        env = wandb_run = None
        train_metrics = {}

        def metrics():
            return {
                **train_metrics,
                'info/global_step': self.global_step * cfg.n_action_steps,
                'info/chunk_step': self.global_step,
                'info/global_update': self.global_update,
                'info/num_episodes': self.num_episodes,
                'info/recent_done_sr': float(np.mean(recent_success)) if recent_success else 0.,
                'info/recent_done_count': len(recent_success),

                'info/solid_steps': env.solid_steps,
                'info/idle_steps': env.idle_steps,
                'info/idle_ratio': env.idle_steps / max(env.solid_steps + env.idle_steps, 1),
            }

        try:
            env = OneLegOnlineEnv(
                n_action_steps=cfg.n_action_steps,
                max_steps=cfg.online_task.max_steps,
                single_gamma=cfg.single_gamma,
                reward_offset=cfg.training.reward_offset,
                num_envs=cfg.training.n_envs,
                compute_device_id=device.index or 0,
                graphics_device_id=device.index or 0,
                render_size=tuple(cfg.online_task.render_size),
                randomness=cfg.online_task.randomness
            )
            wandb_run = wandb.init(
                dir=str(self.output_dir),
                config=OmegaConf.to_container(cfg, resolve=True),
                **cfg.logging
            )
            next_log_step = cfg.training.learning_start
            with JsonLogger(os.path.join(self.output_dir, 'logs.json.txt')) as logger, tqdm.tqdm(
                    total=cfg.training.num_steps * cfg.n_action_steps,
                    initial=self.global_step * cfg.n_action_steps,
                    desc='Sampling', unit='step', mininterval=cfg.training.tqdm_interval_sec,
                    leave=False, position=0) as sampling:
                try:
                    while self.global_step < cfg.training.num_steps:
                        if env.batch_done:
                            obs = env.reset()
                            sum_policy.reset(obs)
                            collector.begin_batch()
                        # Match DSRL's pre-action count for noise warmup.
                        decision_step = self.global_step + int(env.active.sum())
                        action = sum_policy.predict_action(
                            obs, global_step=decision_step,
                            learning_start=cfg.training.learning_start)['action']
                        obs, rewards, dones, infos = env.step(action)
                        transition = sum_policy.on_env_step(obs, dones, infos)
                        self.global_step += sum(info['transition_valid'] for info in infos)
                        sampling.update(self.global_step * cfg.n_action_steps - sampling.n)
                        collector.collect(transition, rewards, dones, infos)
                        for done, info in zip(dones.tolist(), infos):
                            if done:
                                self.num_episodes += 1
                                recent_success.append(float(info['success']))

                        # First batch has ended: learning_start / n_envs * Tr > max_steps.
                        if self.global_step >= cfg.training.learning_start:
                            if next_train_step is None:
                                next_train_step = self.global_step
                            while self.global_step >= next_train_step:
                                sampling.clear()
                                train_metrics.update(self._update(cfg, optimizers))
                                next_train_step += cfg.training.training_freq

                        if self.global_step >= next_log_step:
                            record = metrics()
                            logger.log(record)
                            wandb_run.log(record, step=self.global_step * cfg.n_action_steps)
                            next_log_step = (self.global_step // cfg.training.log_every + 1) * cfg.training.log_every
                finally:
                    collector.discard_pending()
                    self.last_log = metrics()
                    if self.global_step >= cfg.training.learning_start:
                        logger.log(self.last_log)
                        wandb_run.log(self.last_log, step=self.global_step * cfg.n_action_steps)

            # Saving remains disabled; base weights are only an external reference.
            # import dill
            # torch.save(self.get_snapshot(), checkpoint_path, pickle_module=dill)
        finally:
            try:
                if env is not None:
                    env.close()
            finally:
                if wandb_run is not None:
                    wandb_run.finish()
