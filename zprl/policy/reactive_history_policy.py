import copy
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F

from zprl.common.reactive_history import HistoryReplayBufferSamples, HistoryWindow
from zprl.model.common.module_attr_mixin import ModuleAttrMixin
from zprl.model.common.shape_util import assert_shape
from zprl.model.rl_history import HistoryActor, HistoryQEnsemble
from zprl.policy.reactive_policy import SumPolicy as ReactiveSumPolicy


@contextmanager
def _module_mode(module, training):
    previous = module.training
    module.train(training)
    try:
        yield
    finally:
        module.train(previous)


@contextmanager
def _frozen_parameters(module):
    parameters = list(module.parameters())
    previous = [p.requires_grad for p in parameters]
    module.requires_grad_(False)
    try:
        yield
    finally:
        for parameter, requires_grad in zip(parameters, previous):
            parameter.requires_grad_(requires_grad)


class HistoryResiduePolicy(ModuleAttrMixin):
    def __init__(self,
            obs_dim: int,
            action_dim: int,
            history_length: int,
            exec_dim: int = None,
            actor_input: str = 'obs_action',
            hidden_dim: int = 256,
            log_std_min: float = -10.0,
            log_std_max: float = 2.0,
            gamma: float = 0.97,
            tau: float = 0.01,
            init_alpha: float = 0.01,
            auto_alpha: bool = True,
            res_scale: float = 0.05,
            q_ent: bool = True,
            num_qs: int = 2,
            num_subset: int = 2,
            debug: bool = False,
            critic_action_input: str = 'exec',
            sub_ratio: int = 1,
            target_clip: list = None,
            d_model: int = 256,
            nhead: int = 4,
            num_layers: int = 2,
            dim_feedforward: int = 1024,
            dropout: float = 0.0,
            causal_mask: bool = True):
        super().__init__()
        if target_clip is not None:
            if len(target_clip) != 2:
                raise ValueError('target_clip must contain [lb, ub]')
            target_clip = tuple(float(bound) for bound in target_clip)
            if not target_clip[0] <= target_clip[1]:
                raise ValueError('target_clip lb must not exceed the ub')
        exec_dim = action_dim if exec_dim is None else exec_dim
        assert exec_dim == action_dim  # ResRL combines base and residual in exec space.
        assert actor_input == 'obs_action'
        assert 0 < num_subset <= num_qs
        common = dict(
            obs_dim=obs_dim, exec_dim=exec_dim, history_length=history_length,
            hidden_dim=hidden_dim, debug=debug, sub_ratio=sub_ratio,
            d_model=d_model, nhead=nhead, num_layers=num_layers,
            dim_feedforward=dim_feedforward, dropout=dropout, causal_mask=causal_mask)
        self.actor = HistoryActor(action_dim=action_dim, log_std_min=log_std_min,
                                  log_std_max=log_std_max, **common)
        self.qs = HistoryQEnsemble(num_qs=num_qs,
                                     critic_action_input=critic_action_input, **common)
        self.q_targets = copy.deepcopy(self.qs).requires_grad_(False).eval()
        self.log_alpha = nn.Parameter(torch.log(torch.tensor(init_alpha, dtype=torch.float32)))

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.exec_dim = exec_dim
        self.history_length = history_length
        self.sub_ratio = sub_ratio
        self.actor_input = actor_input
        self.critic_action_input = critic_action_input
        self.num_qs = num_qs
        self.num_subset = num_subset
        self.gamma = gamma
        self.tau = tau
        self.init_alpha = init_alpha
        self.auto_alpha = auto_alpha
        self.target_entropy = -action_dim / 2
        self.res_scale = res_scale
        self.q_ent = q_ent
        self.target_clip = target_clip

    def train(self, mode=True):
        super().train(mode)
        self.q_targets.eval()
        return self

    def get_optimizer(self, policy_lr, q_lr):
        return {
            'actor_optimizer': torch.optim.Adam(self.actor.parameters(), lr=policy_lr),
            'q_optimizer': torch.optim.Adam(self.qs.parameters(), lr=q_lr),
            'alpha_optimizer': torch.optim.Adam([self.log_alpha], lr=q_lr),
        }

    def _sample_naction_log_prob(self, obs, a_exec, obs_valid, a_base):
        result = self.actor.get_action(obs, a_exec, obs_valid, a_base)
        assert_shape(result['sample'], (obs.shape[0], self.action_dim))
        assert_shape(result['log_prob'], (obs.shape[0], 1))
        return result['sample'], result['log_prob'], result['log_std']

    def _critic_values(self, critic, obs, a_exec, obs_valid, a_base, residual):
        if self.critic_action_input == 'base_res':
            return critic(obs, a_exec, obs_valid, a_base, residual)
        return critic(obs, a_exec, obs_valid, a_base + self.res_scale * residual)

    def compute_critic_loss(self, batch: HistoryReplayBufferSamples, dist=None):
        bs = batch.rewards.shape[0]
        assert_shape(batch.action, (bs, self.action_dim))
        assert_shape(batch.a_base, (bs, self.exec_dim))
        alpha = self.log_alpha.detach().exp() if self.auto_alpha else self.init_alpha

        with _module_mode(self.actor, False), torch.no_grad():
            next_res, next_log_prob, _ = self._sample_naction_log_prob(
                batch.next_obs, batch.next_a_exec, batch.next_obs_valid,
                batch.next_a_base)
            target_q_all = self._critic_values(
                self.q_targets, batch.next_obs, batch.next_a_exec,
                batch.next_obs_valid, batch.next_a_base, next_res)
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
            all_q_preds = self._critic_values(
                self.qs, batch.obs, batch.a_exec, batch.obs_valid,
                batch.a_base, batch.action).squeeze(-1)
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

    def compute_actor_loss(self, batch: HistoryReplayBufferSamples):
        bs = batch.rewards.shape[0]
        alpha = self.log_alpha.detach().exp() if self.auto_alpha else self.init_alpha
        with _module_mode(self.actor, True):
            res_action, log_prob, log_std = self._sample_naction_log_prob(
                batch.obs, batch.a_exec, batch.obs_valid, batch.a_base)
            with _module_mode(self.qs, False), _frozen_parameters(self.qs):
                all_q_preds = self._critic_values(
                    self.qs, batch.obs, batch.a_exec, batch.obs_valid,
                    batch.a_base, res_action)
            predicted_q = all_q_preds.mean(dim=0)
            assert_shape(predicted_q, (bs, 1))
            actor_loss = (alpha * log_prob - predicted_q).mean()

        info = {
            'actor_entropy': -log_prob.mean().detach(),
            'res_naction_norm': torch.norm(res_action, dim=-1).mean().detach(),
            'res_n_rms': torch.sqrt((res_action ** 2).mean()).detach(),
            'base_n_rms': torch.sqrt((batch.a_base ** 2).mean()).detach(),
            'log_std': log_std.mean().detach(),
        }
        return actor_loss, info

    def compute_alpha_loss(self, batch: HistoryReplayBufferSamples):
        with _module_mode(self.actor, True), torch.no_grad():
            _, log_prob, _ = self._sample_naction_log_prob(
                batch.obs, batch.a_exec, batch.obs_valid, batch.a_base)
        return (-self.log_alpha * (log_prob + self.target_entropy).detach()).mean()

    @torch.no_grad()
    def target_update(self):
        for parameter, target_parameter in zip(self.qs.parameters(), self.q_targets.parameters()):
            target_parameter.lerp_(parameter, self.tau)
        self.q_targets.eval()

    @torch.no_grad()
    def predict_res_naction(self, obs, a_exec, obs_valid, a_base, argmax=False):
        with _module_mode(self.actor, False):
            if argmax:
                return self.actor.get_eval_action(obs, a_exec, obs_valid, a_base)
            return self.actor.get_action(obs, a_exec, obs_valid, a_base)['sample']


class HistorySumPolicy(ReactiveSumPolicy):
    def reset(self, init_obs=None):
        self.history = HistoryWindow(self.res_policy.history_length, self.res_policy.exec_dim)
        self.current_obs_emb = None
        self.base_cache = None
        self.seg_idx = None
        self.pending = None
        if init_obs is not None:
            obs_dict = self._tensor_obs(init_obs)
            base_pred, obs_emb = self.encode_obs(obs_dict, predict_action=True)
            self.current_obs_emb = obs_emb.detach().clone()
            self.base_cache = base_pred['naction'].detach().clone()
            self.seg_idx = torch.zeros(obs_emb.shape[0], device=obs_emb.device,
                                       dtype=torch.long)
            self.history.reset(self.current_obs_emb)

    def _tensor_obs(self, obs_dict):
        return {key: (value.to(self.device) if isinstance(value, torch.Tensor)
                      else torch.as_tensor(value, device=self.device))
                for key, value in obs_dict.items()}

    def _base_segment(self, indices):
        batch = torch.arange(self.base_cache.shape[0], device=self.base_cache.device)
        steps = indices[:, None] * self.n_rl_steps + torch.arange(
            self.n_rl_steps, device=self.base_cache.device)[None]
        return self.base_cache[batch[:, None], steps]

    @torch.no_grad()
    def predict_action(self, obs_dict, deterministic=True, res_mask=None):
        """obs_dict is unused; only returned action is unnormalized for env.step."""
        if self.current_obs_emb is None:
            raise RuntimeError('Call reset(init_obs) before predict_action')
        if self.pending is not None:
            raise RuntimeError('Call on_env_step before the next predict_action')
        obs, a_exec, obs_valid, _ = self.history.get()
        base_naction = self._base_segment(self.seg_idx)
        res_naction = self.res_policy.predict_res_naction(
            obs, a_exec, obs_valid, base_naction.flatten(start_dim=1), deterministic)
        if res_mask is not None:
            res_mask = torch.as_tensor(res_mask, device=self.device, dtype=torch.bool)
            assert_shape(res_mask, (obs.shape[0],))
            res_naction = res_naction.masked_fill(res_mask[:, None], 0)
        sum_naction = self.res_scale * res_naction.reshape_as(base_naction) + base_naction
        assert_shape(sum_naction, (obs.shape[0], self.n_rl_steps, self.action_dim))
        self.pending = {
            'obs': self.current_obs_emb.detach().clone(),
            'a_base': base_naction.flatten(start_dim=1).detach().clone(),
            'action': res_naction.detach().clone(),
            'a_exec': sum_naction.flatten(start_dim=1).detach().clone(),
        }
        return {
            'action': self.normalizer['action'].unnormalize(sum_naction),
            'naction': sum_naction,
            'res_naction_flat': res_naction
        }

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
        assert torch.all((exec_steps > 0) & (exec_steps <= self.n_rl_steps))
        timeouts = torch.tensor(
            [bool(info.get('TimeLimit.truncated', False)) for info in infos],
            device=self.device, dtype=torch.bool)
        assert not torch.any(timeouts & ~dones)
        pending = self.pending
        self.history.complete(pending['a_exec'], exec_steps)

        next_idx = self.seg_idx + 1
        boundary = next_idx == self.n_segments
        mid_timeout_base = {}
        for index in torch.nonzero(timeouts & ~boundary).flatten().tolist():
            mid_timeout_base[index] = self.base_cache[
                index, next_idx[index] * self.n_rl_steps:
                (next_idx[index] + 1) * self.n_rl_steps].clone()

        # need to predict new base when cache is used or epi ends
        # but next_obs is always needed
        obs_dict = self._tensor_obs(next_obs_dict)
        refresh = boundary | dones
        next_obs_emb = torch.empty_like(self.current_obs_emb)
        for mask, predict_plan in ((~refresh, False), (refresh, True)):
            if torch.any(mask):
                subset = {key: value[mask] for key, value in obs_dict.items()}
                base_pred, obs_emb = self.encode_obs(subset, predict_action=predict_plan)
                next_obs_emb[mask] = obs_emb
                if predict_plan:
                    self.base_cache[mask] = base_pred['naction'].detach()
                    next_idx[mask] = 0
        next_base = self._base_segment(next_idx)
        rb_next_obs = next_obs_emb.clone()
        rb_next_base = next_base.flatten(start_dim=1).clone()

        # if timeout, the next_obs for bootstrap is the terminal
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
            terminal_boundary = boundary[terminal_indices]
            for mask, predict_plan in ((~terminal_boundary, False),
                                       (terminal_boundary, True)):
                if torch.any(mask):
                    subset = {key: value[mask] for key, value in terminal_obs.items()}
                    base_pred, obs_emb = self.encode_obs(subset, predict_action=predict_plan)
                    selected = torch.as_tensor(terminal_indices, device=self.device)[mask]
                    rb_next_obs[selected] = obs_emb
                    if predict_plan:
                        rb_next_base[selected] = base_pred['naction'][:, :self.n_rl_steps].flatten(start_dim=1)
            for index, base in mid_timeout_base.items():
                rb_next_base[index] = base.flatten()

        self.history.append(next_obs_emb)
        if torch.any(dones):
            self.history.reset(next_obs_emb, dones)
        self.current_obs_emb = next_obs_emb.detach().clone()
        self.seg_idx = next_idx
        self.pending = None
        return {
            'obs': pending['obs'],
            'a_base': pending['a_base'],
            'action': pending['action'],
            'a_exec': pending['a_exec'],
            'exec_steps': exec_steps.clone(),
            'next_obs': rb_next_obs.detach().clone(),
            'next_a_base': rb_next_base.detach().clone(),
        }
