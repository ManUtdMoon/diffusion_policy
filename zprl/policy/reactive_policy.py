from typing import Dict, Optional

import torch

from zprl.model.common.shape_util import assert_shape
from zprl.policy.base_image_policy import BaseImagePolicy
from zprl.policy.residue_policy import ResiduePolicy, SumPolicy as ActionSumPolicy


class SumPolicy(ActionSumPolicy):
    """Refresh residual actions every Tr steps within a cached Ta-step base plan."""

    def __init__(self,
            res_scale: float,
            base_obs_emb_dim: int,
            subtask_dim: int,
            action_dim: int,
            n_action_steps: int,
            base_policy: BaseImagePolicy,
            res_policy: ResiduePolicy,
            n_rl_steps: Optional[int] = None):
        super().__init__(
            res_scale=res_scale,
            base_obs_emb_dim=base_obs_emb_dim,
            subtask_dim=subtask_dim,
            action_dim=action_dim,
            n_action_steps=n_action_steps,
            base_policy=base_policy,
            res_policy=res_policy,
        )
        self.base_policy.requires_grad_(False)
        self.n_rl_steps = n_action_steps if n_rl_steps is None else n_rl_steps
        assert n_action_steps > 0 and self.n_rl_steps > 0 \
            and n_action_steps % self.n_rl_steps == 0, (
                f"n_action_steps({n_action_steps}) must be divisible by n_rl_steps({self.n_rl_steps})")
        assert res_policy.action_dim == self.n_rl_steps * action_dim
        self.n_segments = n_action_steps // self.n_rl_steps
        self.reset()

    def reset(self):
        self.base_cache = None
        self.seg_idx = 0

    @torch.no_grad()
    def encode_obs(self, obs_dict: Dict[str, torch.Tensor], predict_action=True):
        stage_mask = None
        base_obs = obs_dict
        if self.subtask_dim > 0:
            stage_mask = obs_dict['completed_stage_mask'][:, -1]
            assert_shape(stage_mask, (None, self.subtask_dim))
            base_obs = dict(obs_dict)
            del base_obs['completed_stage_mask']

        if predict_action:
            base_res = self.base_policy.predict_action(base_obs)
            obs_seq_emb = base_res['obs_emb']
            assert_shape(base_res['naction'], (None, self.n_action_steps, self.action_dim))
        else:
            base_res = None
            obs_seq_emb = self.base_policy.encode_obs(base_obs)
        obs_emb = obs_seq_emb[:, -self.base_obs_emb_dim:]
        if stage_mask is not None:
            obs_emb = torch.cat([obs_emb, stage_mask], dim=-1)
        assert_shape(obs_emb, (None, self.obs_emb_dim))
        return base_res, obs_emb

    @torch.no_grad()
    def predict_action(self,
            obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        refresh = self.base_cache is None or self.seg_idx == self.n_segments
        base_res, obs_emb = self.encode_obs(obs_dict, predict_action=refresh)
        if refresh:
            self.base_cache = base_res['naction'].detach()
            self.seg_idx = 0
        assert self.base_cache.shape[0] == obs_emb.shape[0]

        start = self.seg_idx * self.n_rl_steps
        base_naction = self.base_cache[:, start:start + self.n_rl_steps]
        res_input = obs_emb
        if self.res_policy.actor_input == 'obs_action':
            res_input = torch.cat(
                [obs_emb, base_naction.flatten(start_dim=1)], dim=-1)
        res_naction = self.res_policy.predict_res_naction(
            res_input, True).reshape_as(base_naction)
        sum_naction = self.res_scale * res_naction + base_naction
        sum_action = self.normalizer['action'].unnormalize(sum_naction)
        assert_shape(sum_action, (None, self.n_rl_steps, self.action_dim))
        self.seg_idx += 1
        return {'action': sum_action}

    @torch.no_grad()
    def predict_train_action(self, base_naction, obs_emb, res_mask=None):
        assert_shape(base_naction, (obs_emb.shape[0], self.n_rl_steps, self.action_dim))
        return super().predict_train_action(base_naction, obs_emb, res_mask)
