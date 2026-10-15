import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from zprl.common.reactive_history import NoiseHistoryReplayBufferSamples, HistoryWindow
from zprl.model.common.module_attr_mixin import ModuleAttrMixin
from zprl.model.common.shape_util import assert_shape
from zprl.model.rl_history import HistoryActor, HistoryQEnsemble
from zprl.policy.reactive_history_policy import _module_mode, _frozen_parameters


class HistoryNoisePolicy(ModuleAttrMixin):
    """Learn normalized initial noise; history actions are executed model actions."""

    def __init__(self,
            obs_dim: int,
            action_dim: int,
            exec_dim: int,
            history_length: int,
            hidden_dim: int = 256,
            log_std_min: float = -10.0,
            log_std_max: float = 2.0,
            gamma: float = 0.97,
            tau: float = 0.01,
            init_alpha: float = 0.01,
            auto_alpha: bool = True,
            target_entropy: float = None,
            q_ent: bool = True,
            num_qs: int = 2,
            num_subset: int = 2,
            debug: bool = False,
            sub_ratio: int = 2,
            target_clip: list = None,
            gru_input_dim: int = 128,
            gru_hidden_dim: int = 128,
            gru_num_layers: int = 2):
        super().__init__()
        if target_clip is not None:
            if len(target_clip) != 2:
                raise ValueError('target_clip must contain [lb, ub]')
            target_clip = tuple(float(bound) for bound in target_clip)
            if not target_clip[0] <= target_clip[1]:
                raise ValueError('target_clip lb must not exceed the ub')
        assert action_dim > 0 and exec_dim > 0
        assert 0 < num_subset <= num_qs
        common = dict(
            obs_dim=obs_dim, exec_dim=exec_dim, history_length=history_length,
            hidden_dim=hidden_dim, debug=debug, sub_ratio=sub_ratio,
            gru_input_dim=gru_input_dim, gru_hidden_dim=gru_hidden_dim,
            gru_num_layers=gru_num_layers)
        self.actor = HistoryActor(action_dim=action_dim, condition_dim=0,
                                  log_std_min=log_std_min, log_std_max=log_std_max,
                                  **common)
        self.qs = HistoryQEnsemble(num_qs=num_qs, candidate_action_dim=action_dim,
                                  critic_action_input='noise', **common)
        self.q_targets = copy.deepcopy(self.qs).requires_grad_(False).eval()
        self.log_alpha = nn.Parameter(torch.log(torch.tensor(init_alpha, dtype=torch.float32)))

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.exec_dim = exec_dim
        self.history_length = history_length
        self.sub_ratio = sub_ratio
        self.num_qs = num_qs
        self.num_subset = num_subset
        self.gamma = gamma
        self.tau = tau
        self.init_alpha = init_alpha
        self.auto_alpha = auto_alpha
        self.target_entropy = -action_dim / 2 if target_entropy is None else target_entropy
        self.q_ent = q_ent
        self.target_clip = target_clip

    def train(self, mode=True):
        super().train(mode)
        self.q_targets.eval()
        return self

    def get_optimizer(self, policy_lr, q_lr, enc_lr=None):
        def param_groups(module, head_lr):
            if enc_lr is None:
                return module.parameters()
            encoder_parameters = list(module.encoder.parameters())
            encoder_ids = {id(parameter) for parameter in encoder_parameters}
            return [
                {'params': [parameter for parameter in module.parameters()
                            if id(parameter) not in encoder_ids], 'lr': head_lr},
                {'params': encoder_parameters, 'lr': enc_lr},
            ]

        return {
            'actor_optimizer': torch.optim.Adam(param_groups(self.actor, policy_lr), lr=policy_lr),
            'q_optimizer': torch.optim.Adam(param_groups(self.qs, q_lr), lr=q_lr),
            'alpha_optimizer': torch.optim.Adam([self.log_alpha], lr=q_lr),
        }

    def _sample_noise_log_prob(self, obs, a_exec, obs_valid):
        result = self.actor.get_action(obs, a_exec, obs_valid)
        assert_shape(result['sample'], (obs.shape[0], self.action_dim))
        assert_shape(result['log_prob'], (obs.shape[0], 1))
        return result['sample'], result['log_prob'], result['log_std']

    def compute_critic_loss(self, batch: NoiseHistoryReplayBufferSamples):
        bs = batch.rewards.shape[0]
        assert_shape(batch.action, (bs, self.action_dim))
        alpha = self.log_alpha.detach().exp() if self.auto_alpha else self.init_alpha

        with _module_mode(self.actor, False), torch.no_grad():
            next_noise, next_log_prob, _ = self._sample_noise_log_prob(
                batch.next_obs, batch.next_a_exec, batch.next_obs_valid)
            target_q_all = self.q_targets(
                batch.next_obs, batch.next_a_exec, batch.next_obs_valid, next_noise)
            subset_indices = torch.randperm(self.num_qs, device=target_q_all.device)[:self.num_subset]
            target_q_next = target_q_all[subset_indices].min(dim=0).values
            if self.q_ent:
                target_q_next -= alpha * next_log_prob
            assert_shape(target_q_next, (bs, 1))
            discounts = self.gamma if batch.discounts is None else batch.discounts.flatten()
            if isinstance(discounts, torch.Tensor):
                assert_shape(discounts, (bs,))
            target_q = batch.rewards.flatten() + (
                (1 - batch.dones.flatten()) * discounts * target_q_next.flatten())
            if self.target_clip is not None:
                target_q = target_q.clamp(*self.target_clip)
            target_q = target_q.unsqueeze(0).expand(self.num_qs, -1)

        with _module_mode(self.qs, True):
            all_q_preds = self.qs(
                batch.obs, batch.a_exec, batch.obs_valid, batch.action).squeeze(-1)
        critic_loss = F.mse_loss(all_q_preds, target_q, reduction='none').mean(dim=-1).sum()
        info = {
            'q_target': target_q.mean().detach(),
            'q_predicted': all_q_preds.mean().detach(),
            'q_predicted_min': all_q_preds.mean(dim=0).min().detach(),
            'q_predicted_max': all_q_preds.mean(dim=0).max().detach(),
            'rewards': batch.rewards.mean().detach(),
            'dones': batch.dones.float().mean().detach(),
        }
        return critic_loss, info

    def compute_actor_loss(self, batch: NoiseHistoryReplayBufferSamples):
        bs = batch.rewards.shape[0]
        alpha = self.log_alpha.detach().exp() if self.auto_alpha else self.init_alpha
        with _module_mode(self.actor, True):
            noise, log_prob, log_std = self._sample_noise_log_prob(
                batch.obs, batch.a_exec, batch.obs_valid)
            with _module_mode(self.qs, False), _frozen_parameters(self.qs):
                all_q_preds = self.qs(batch.obs, batch.a_exec, batch.obs_valid, noise)
            predicted_q = all_q_preds.mean(dim=0)
            assert_shape(predicted_q, (bs, 1))
            actor_loss = (alpha * log_prob - predicted_q).mean()

        info = {
            'actor_entropy': -log_prob.mean().detach(),
            'res_n_rms': noise.square().mean(dim=-1).sqrt().mean().detach(),
            'log_std': log_std.mean().detach(),
        }
        return actor_loss, info

    def compute_alpha_loss(self, batch: NoiseHistoryReplayBufferSamples):
        with _module_mode(self.actor, True), torch.no_grad():
            _, log_prob, _ = self._sample_noise_log_prob(
                batch.obs, batch.a_exec, batch.obs_valid)
        return (-self.log_alpha * (log_prob + self.target_entropy).detach()).mean()

    @torch.no_grad()
    def target_update(self):
        for parameter, target_parameter in zip(self.qs.parameters(), self.q_targets.parameters()):
            target_parameter.lerp_(parameter, self.tau)
        self.q_targets.eval()

    @torch.no_grad()
    def predict_noise(self, obs, a_exec, obs_valid, argmax=False):
        with _module_mode(self.actor, False):
            if argmax:
                return self.actor.get_eval_action(obs, a_exec, obs_valid)
            return self.actor.get_action(obs, a_exec, obs_valid)['sample']


class HistoryNoiseSumPolicy:
    """Decode one complete frozen base trajectory per noise decision."""

    def __init__(self,
            noise_scale: float,
            base_obs_emb_dim: int,
            subtask_dim: int,
            action_dim: int,
            n_action_steps: int,
            horizon: int,
            n_obs_steps: int,
            base_policy,
            noise_policy: HistoryNoisePolicy,
            n_noise_steps: int = 1):
        assert hasattr(base_policy, 'encode_obs')
        assert hasattr(base_policy, 'conditional_predict_from_noise')
        assert horizon == base_policy.horizon
        assert n_obs_steps == base_policy.n_obs_steps
        assert action_dim == base_policy.action_dim
        assert noise_scale > 0 and n_noise_steps > 0
        assert horizon % n_noise_steps == 0
        assert 1 <= n_action_steps <= horizon - n_obs_steps + 1
        assert base_obs_emb_dim == n_obs_steps * base_policy.obs_feature_dim
        assert noise_policy.obs_dim == base_obs_emb_dim + subtask_dim
        assert noise_policy.action_dim == n_noise_steps * action_dim
        assert noise_policy.exec_dim == n_action_steps * action_dim
        self.noise_scale = noise_scale
        self.base_obs_emb_dim = base_obs_emb_dim
        self.subtask_dim = subtask_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.n_noise_steps = n_noise_steps
        self.base_policy = base_policy.requires_grad_(False).eval()
        self.noise_policy = noise_policy
        self.normalizer = base_policy.normalizer
        # Only the execution slice changes; the checkpoint trajectory shape is fixed.
        self.base_policy.n_action_steps = n_action_steps
        self.reset()

    @property
    def device(self):
        return self.noise_policy.device

    @property
    def dtype(self):
        return self.noise_policy.dtype

    def train(self, mode=True):
        self.noise_policy.train(mode)
        self.base_policy.eval()
        return self

    def eval(self):
        return self.train(False)

    def _tensor_obs(self, obs_dict):
        return {key: (value.to(self.device) if isinstance(value, torch.Tensor)
                      else torch.as_tensor(value, device=self.device))
                for key, value in obs_dict.items()}

    @torch.no_grad()
    def encode_obs(self, obs_dict):
        base_obs = dict(obs_dict)
        stage_mask = None
        if self.subtask_dim > 0:
            stage_mask = base_obs.pop('completed_stage_mask')[:, -1]
            assert_shape(stage_mask, (None, self.subtask_dim))
        base_cond = self.base_policy.encode_obs(base_obs)
        assert_shape(base_cond, (None, self.base_obs_emb_dim))
        obs_emb = base_cond
        if stage_mask is not None:
            obs_emb = torch.cat([obs_emb, stage_mask], dim=-1)
        assert_shape(obs_emb, (None, self.noise_policy.obs_dim))
        return base_cond, obs_emb

    @torch.no_grad()
    def reset(self, init_obs=None):
        self.history = HistoryWindow(self.noise_policy.history_length, self.noise_policy.exec_dim)
        self.current_base_cond = None
        self.current_obs_emb = None
        self.pending = None
        if init_obs is not None:
            base_cond, obs_emb = self.encode_obs(self._tensor_obs(init_obs))
            self.current_base_cond = base_cond.detach().clone()
            self.current_obs_emb = obs_emb.detach().clone()
            self.history.reset(self.current_obs_emb)

    def _repeat_noise(self, noise):
        assert_shape(noise, (None, self.noise_policy.action_dim))
        return noise.reshape(-1, self.n_noise_steps, self.action_dim).repeat(
            1, self.horizon // self.n_noise_steps, 1)

    @torch.no_grad()
    def predict_action(self, obs_dict, deterministic=None, global_step=None, learning_start=None):
        """Use reset/on_env_step observations; training counts select warmup or actor sampling."""
        if self.current_obs_emb is None:
            raise RuntimeError('Call reset(init_obs) before predict_action')
        if self.pending is not None:
            raise RuntimeError('Call on_env_step before the next predict_action')
        assert (global_step is None) == (learning_start is None)
        if deterministic is None:
            deterministic = False
        obs, a_exec, obs_valid, _ = self.history.get()
        bs = obs.shape[0]
        if global_step is not None and global_step < learning_start:
            epsilon = obs.new_empty(bs, self.noise_policy.action_dim).normal_()
            noise = epsilon.clamp(-self.noise_scale, self.noise_scale) / self.noise_scale
        else:
            noise = self.noise_policy.predict_noise(
                obs, a_exec, obs_valid, argmax=deterministic if global_step is None else False)
        assert_shape(noise, (bs, self.noise_policy.action_dim))
        result = self.base_policy.conditional_predict_from_noise(
            self.current_base_cond, self._repeat_noise(noise) * self.noise_scale)
        assert_shape(result['naction_pred'], (bs, self.horizon, self.action_dim))
        assert_shape(result['action_pred'], (bs, self.horizon, self.action_dim))
        start = self.n_obs_steps - 1
        naction = result['naction_pred'][:, start:start + self.n_action_steps]
        action = self.normalizer['action'].unnormalize(naction)
        assert_shape(naction, (bs, self.n_action_steps, self.action_dim))
        assert_shape(action, (bs, self.n_action_steps, self.action_dim))
        self.pending = {
            'obs': self.current_obs_emb.detach().clone(),
            'action': noise.detach().clone(),
            'a_exec': naction.flatten(start_dim=1).detach().clone(),
        }
        return {**result, 'action': action, 'naction': naction, 'nnoise': noise}

    @torch.no_grad()
    def on_env_step(self, next_obs_dict, dones, infos):
        if self.pending is None:
            raise RuntimeError('Call predict_action before on_env_step')
        batch_size = self.current_obs_emb.shape[0]
        dones = torch.as_tensor(dones, device=self.device, dtype=torch.bool)
        assert_shape(dones, (batch_size,))
        assert len(infos) == batch_size
        exec_steps = torch.tensor([info['executed_action_steps'] for info in infos],
                                  device=self.device, dtype=torch.long)
        assert torch.all((exec_steps > 0) & (exec_steps <= self.n_action_steps))
        timeouts = torch.tensor(
            [bool(info.get('TimeLimit.truncated', False)) for info in infos],
            device=self.device, dtype=torch.bool)
        assert not torch.any(timeouts & ~dones)
        pending = self.pending
        executed = pending['a_exec'].reshape(batch_size, self.n_action_steps, self.action_dim)
        executed_mask = torch.arange(self.n_action_steps, device=self.device)[None] < exec_steps[:, None]
        a_exec = executed.masked_fill(~executed_mask[..., None], 0).flatten(start_dim=1)
        self.history.complete(a_exec, exec_steps)

        obs_dict = self._tensor_obs(next_obs_dict)
        next_base_cond, next_obs_emb = self.encode_obs(obs_dict)
        rb_next_obs = next_obs_emb.clone()
        if torch.any(timeouts):
            terminal_indices = torch.nonzero(timeouts).flatten().tolist()
            terminal_obs_list = [
                self._tensor_obs(infos[index]['terminal_observation'])
                for index in terminal_indices
            ]
            terminal_obs = {
                key: torch.stack([obs[key] for obs in terminal_obs_list])
                for key in obs_dict
            }
            _, terminal_emb = self.encode_obs(terminal_obs)
            rb_next_obs[terminal_indices] = terminal_emb

        self.history.append(next_obs_emb)
        if torch.any(dones):
            self.history.reset(next_obs_emb, dones)
        self.current_base_cond = next_base_cond.detach().clone()
        self.current_obs_emb = next_obs_emb.detach().clone()
        self.pending = None
        return {
            'obs': pending['obs'],
            'action': pending['action'],
            'a_exec': a_exec.detach().clone(),
            'exec_steps': exec_steps.clone(),
            'next_obs': rb_next_obs.detach().clone(),
        }
