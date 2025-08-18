from typing import Dict, Union
import torch
import torch.nn as nn

from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.policy.residue_policy import ResiduePolicy
from diffusion_policy.model.common.shape_util import assert_shape


class SumPolicy:
    def __init__(self,
            # dimensions
            res_scale: float,
            obs_emb_dim: int,
            action_dim: int,
            n_action_steps: int,
            # policies
            base_policy: BaseImagePolicy,
            res_policy: ResiduePolicy,):
        self.base_policy = base_policy
        self.res_policy = res_policy
        self.res_scale = res_scale
        self.normalizer = base_policy.normalizer

        # make sure the base policy is in eval mode
        self.base_policy.eval()

        # store dimensions
        self.obs_emb_dim = obs_emb_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps

    def reset(self):
        pass

    def eval(self):
        self.res_policy.eval()

    def train(self):
        self.res_policy.train()

    @property
    def device(self) -> torch.device:
        return self.res_policy.device

    @property
    def dtype(self) -> torch.dtype:
        return self.res_policy.dtype

    @torch.no_grad()
    def predict_action(self,
            obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        normalizer = self.base_policy.normalizer
        base_res = self.base_policy.predict_action(obs_dict)
        obs_seq_emb = base_res['obs_emb']
        base_naction = base_res['naction']
        obs_emb = obs_seq_emb[:, -self.obs_emb_dim:]

        res_input = obs_emb
        if self.res_policy.actor_input == 'obs_action':
            res_input = torch.cat(
                [obs_emb, base_naction.flatten(start_dim=1)], dim=-1)
        res_naction = self.res_policy.predict_res_naction(
            res_input, True).reshape(-1, self.n_action_steps, self.action_dim)

        sum_naction = self.res_scale * res_naction + base_naction
        sum_action = normalizer['action'].unnormalize(sum_naction)

        # check shapes
        assert_shape(sum_action, (None, self.n_action_steps, self.action_dim))

        return {
            'action': sum_action,
        }

    @torch.no_grad()
    def predict_train_action(self,
            base_naction: torch.Tensor,
            obs_emb: torch.Tensor,
            res_mask: Union[torch.Tensor, None] = None
            ) -> Dict[str, torch.Tensor]:
        """
        Compute the final action and intermediate results given base info and masks.

        Args:
            base_naction (torch.Tensor): normalized act seq by base, (B, Ta, da)
            obs_emb(torch.Tensor): o_t embedding by base, (B,do)
            res_masks (torch.Tensor): True where res is masked, (B,)

        Returns:
            Dict[str, torch.Tensor]
        """
        # 1. forward the residue policy
        res_input = obs_emb
        if self.res_policy.actor_input == 'obs_action':
            res_input = torch.cat(
                [obs_emb, base_naction.flatten(start_dim=1)], dim=-1)
        res_naction_flat = self.res_policy.predict_res_naction(res_input)
        if res_mask is not None:
            res_naction_flat[res_mask] = 0.0  # apply the mask

        # 2. compute the sum action
        res_naction = res_naction_flat.reshape_as(base_naction)
        sum_naction = self.res_scale * res_naction + base_naction
        sum_action = self.normalizer['action'].unnormalize(sum_naction)

        return {
            'action': sum_action, # (B,Ta,da), for env step
            'naction': sum_naction,
            'res_naction_flat': res_naction_flat, # (B,Ta*da)
        }