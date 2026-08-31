from typing import Dict, Optional, Union
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from stable_baselines3.common.type_aliases import ReplayBufferSamples

from zprl.policy.base_image_policy import BaseImagePolicy
from zprl.common.pytorch_util import dict_apply
from zprl.model.common.module_attr_mixin import ModuleAttrMixin
from zprl.model.common.shape_util import assert_shape
from zprl.model.online import Actor, BatchedSoftQNet

logger = logging.getLogger(__name__)

class ResiduePolicy(ModuleAttrMixin):
    def __init__(self,
            # network params
            obs_dim: int,
            action_dim: int,
            actor_input: str = 'obs_action',
            hidden_dim: int = 256,
            log_std_min: float = -10.0,
            log_std_max: float = 2.0,
            # training params
            gamma: float = 0.97,
            tau: float = 0.01,
            init_alpha: float = 0.01,
            auto_alpha: bool = True,
            res_scale: float = 0.05,
            n_action_steps: Optional[int] = None,
            lambda_s: Optional[float] = None,
            lambda_t: Optional[float] = None,
            sigma: float = 0.02,
            # batched-q params
            num_qs: int = 2,
            num_subset: int = 2,):
        super().__init__()

        if (lambda_s is None) != (lambda_t is None):
            raise ValueError("lambda_s and lambda_t must both be None or both be set")
        smoothness_enabled = lambda_s is not None
        if smoothness_enabled:
            if lambda_s < 0 or lambda_t < 0:
                raise ValueError("lambda_s and lambda_t must be non-negative")
            if sigma < 0:
                raise ValueError("sigma must be non-negative")
            if n_action_steps is None or n_action_steps <= 1:
                raise ValueError("n_action_steps must be greater than 1 when smoothness is enabled")
            if action_dim % n_action_steps != 0:
                raise ValueError(
                    f"action_dim({action_dim}) must be divisible by n_action_steps({n_action_steps})")

        # create models
        if actor_input == 'obs':
            obs_agg_dim = obs_dim
        elif actor_input == 'obs_action':
            obs_agg_dim = obs_dim + action_dim
        else:
            raise ValueError(f"Invalid actor_input: {actor_input}")

        actor = Actor(
            obs_dim=obs_agg_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            log_std_min=log_std_min,
            log_std_max=log_std_max
        )

        qs = BatchedSoftQNet(obs_dim=obs_dim, action_dim=action_dim, num_qs=num_qs, hidden_dim=hidden_dim)
        q_targets = BatchedSoftQNet(obs_dim=obs_dim, action_dim=action_dim, num_qs=num_qs, hidden_dim=hidden_dim)
        q_targets.load_state_dict(qs.state_dict())
        q_targets.requires_grad_(False)

        log_alpha = nn.Parameter(
            torch.log(torch.tensor(init_alpha, dtype=torch.float32))
        )
        target_entropy = -action_dim / 2 # heuristic target entropy

        self.actor = actor
        self.qs = qs
        self.q_targets = q_targets
        self.log_alpha = log_alpha

        # training params
        self.gamma = gamma
        self.tau = tau
        self.init_alpha = init_alpha
        self.auto_alpha = auto_alpha
        self.target_entropy = target_entropy
        self.res_scale = res_scale
        self.actor_input = actor_input
        self.n_action_steps = n_action_steps
        self.lambda_s = lambda_s
        self.lambda_t = lambda_t
        self.sigma = sigma
        self.smoothness_enabled = smoothness_enabled

        # dimensions
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_qs = num_qs
        self.num_subset = num_subset

        logger.info(
            "number of parameters: %.2f M", sum(p.numel() for p in self.parameters()) / 1e6
        )

    # training
    def get_optimizer(self, policy_lr, q_lr):
        actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=policy_lr)
        q_optimizer = torch.optim.Adam(self.qs.parameters(), lr=q_lr)
        alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=q_lr)
        return {
            'actor_optimizer': actor_optimizer,
            'q_optimizer': q_optimizer,
            'alpha_optimizer': alpha_optimizer
        }
    
    def _sample_naction_log_prob(self, actor_input):
        """
        shared computation between loss functions
        
        Args:
            actor_input (torch.Tensor): The input to the actor network, obs_emb or [obs_emb, base_naction].
        
        Returns:
            res_naction (torch.Tensor): Predicted residual action.
            log_prob (torch.Tensor): The log probability of the residual action.
        """
        bs = actor_input.shape[0]
        res = self.actor.get_action(actor_input)

        assert_shape(res['sample'], (bs, self.action_dim))
        assert_shape(res['log_prob'], (bs, 1))

        return res['sample'], res['log_prob']

    def compute_critic_loss(self, batch: ReplayBufferSamples, dist=None):
        bs = batch.rewards.shape[0]
        res_naction, base_naction, base_next_naction = \
            torch.split(batch.actions, self.action_dim, dim=-1)

        alpha = self.init_alpha
        if self.auto_alpha:
            alpha = self.log_alpha.exp().item()

        # compute targets
        with torch.no_grad():
            actor_input = batch.next_observations
            if self.actor_input == 'obs_action':
                actor_input = torch.cat([batch.next_observations, base_next_naction], dim=-1)
            res_next_naction, next_log_prob = self._sample_naction_log_prob(actor_input)
            next_naction = res_next_naction * self.res_scale + base_next_naction

            target_q_all = self.q_targets(batch.next_observations, next_naction)
            subset_indices = torch.randperm(self.num_qs, device=target_q_all.device)[:self.num_subset]
            target_q_subset = target_q_all[subset_indices]

            target_q_next = torch.min(target_q_subset, dim=0).values  # (B,1)
            assert_shape(target_q_next, (bs, 1))
            target_q = batch.rewards.flatten() + (1 - batch.dones.flatten()) * self.gamma * target_q_next.view(-1) # (B,)
            target_q = target_q.unsqueeze(0).expand(self.num_qs, -1) # broadcast to (num_qs, B)
            assert_shape(target_q, (self.num_qs, bs))

        # compute current Q values
        current_naction = self.res_scale * res_naction + base_naction
        all_q_preds = self.qs(batch.observations, current_naction).squeeze(-1)  # (num_qs, B)

        # compute critic loss
        critic_loss = F.mse_loss(
            all_q_preds, target_q, reduction='none'
        ).mean(dim=-1).sum()

        info = {
            'q_target': target_q.mean().item(),
            'q_predicted': all_q_preds.mean().item(),
            'q_predicted_min': all_q_preds.mean(dim=0).min().item(),
            'q_predicted_max': all_q_preds.mean(dim=0).max().item(),
            "rewards": batch.rewards.mean().item(),
            "dones": batch.dones.float().mean().item(),
        }

        return critic_loss, info

    def compute_actor_loss(self, batch: ReplayBufferSamples):
        bs = batch.rewards.shape[0]
        _, base_naction, _ = \
            torch.split(batch.actions, self.action_dim, dim=-1)

        if self.auto_alpha:
            alpha = self.log_alpha.exp().item()
        else:
            alpha = self.init_alpha

        actor_input = batch.observations
        if self.actor_input == 'obs_action':
            actor_input = torch.cat([batch.observations, base_naction], dim=-1)
        actor_result = self.actor.get_action(actor_input)
        res_naction = actor_result['sample']
        res_naction_mean = actor_result['mean']
        log_prob = actor_result['log_prob']
        assert_shape(res_naction, (bs, self.action_dim))
        assert_shape(res_naction_mean, (bs, self.action_dim))
        assert_shape(log_prob, (bs, 1))

        naction = self.res_scale * res_naction + base_naction  # (B,Da)
        all_q_preds = self.qs(batch.observations, naction)  # (num_qs, B, 1)
        predicted_q = torch.mean(all_q_preds, dim=0)  # (B, 1)
        assert_shape(predicted_q, (bs, 1))

        actor_rl_loss = (alpha * log_prob - predicted_q).mean()
        spatial_smoothness_loss = actor_rl_loss.new_zeros(())
        temporal_smoothness_loss = actor_rl_loss.new_zeros(())
        if self.smoothness_enabled:
            actor_input_bar = (
                actor_input + self.sigma * torch.randn_like(actor_input)
            ).detach()
            res_naction_mean_bar = self.actor.get_eval_action(actor_input_bar)
            assert_shape(res_naction_mean_bar, (bs, self.action_dim))

            naction_mean = self.res_scale * res_naction_mean + base_naction
            naction_mean_bar = self.res_scale * res_naction_mean_bar + base_naction
            spatial_smoothness_loss = 0.5 * (
                naction_mean_bar - naction_mean
            ).pow(2).sum(dim=-1).mean()

            action_step_dim = self.action_dim // self.n_action_steps
            naction_mean = naction_mean.reshape(
                bs, self.n_action_steps, action_step_dim)
            temporal_smoothness_loss = 0.5 * (
                naction_mean[:, 1:] - naction_mean[:, :-1]
            ).pow(2).sum(dim=-1).sum(dim=-1).div(
                self.n_action_steps - 1
            ).mean()

        weighted_spatial_smoothness_loss = spatial_smoothness_loss
        weighted_temporal_smoothness_loss = temporal_smoothness_loss
        if self.smoothness_enabled:
            weighted_spatial_smoothness_loss = \
                self.lambda_s * spatial_smoothness_loss
            weighted_temporal_smoothness_loss = \
                self.lambda_t * temporal_smoothness_loss
        actor_loss = actor_rl_loss \
            + weighted_spatial_smoothness_loss \
            + weighted_temporal_smoothness_loss

        info = {
            'actor_entropy': -log_prob.mean().item(),
            'res_naction_norm': torch.norm(res_naction, dim=-1).mean().item(),
            'actor_rl_loss': actor_rl_loss.item(),
            'spatial_smoothness_loss': spatial_smoothness_loss.item(),
            'temporal_smoothness_loss': temporal_smoothness_loss.item(),
            'weighted_spatial_smoothness_loss': weighted_spatial_smoothness_loss.item(),
            'weighted_temporal_smoothness_loss': weighted_temporal_smoothness_loss.item(),
        }

        return actor_loss, info

    def compute_alpha_loss(self, batch):
        bs = batch.rewards.shape[0]
        _, base_naction, _ = \
            torch.split(batch.actions, self.action_dim, dim=-1)

        actor_input = batch.observations
        if self.actor_input == 'obs_action':
            actor_input = torch.cat([batch.observations, base_naction], dim=-1)

        with torch.no_grad():
            _, log_prob = self._sample_naction_log_prob(actor_input)

        alpha_loss = (-self.log_alpha.exp() * (log_prob + self.target_entropy)).mean()

        return alpha_loss

    def target_update(self,):
        for param, target_param in zip(self.qs.parameters(), self.q_targets.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def predict_res_naction(
            self, actor_input: torch.Tensor, argmax: bool = False) -> torch.Tensor:
        """
        Predict the next residual action based on the current actor input.
        
        Args:
            actor_input (torch.Tensor): The input to the actor network, obs_emb or [obs_emb, base_naction].
            argmax (bool): If True, return mean; if False, return a sample.
            
        Returns
            res_naction (torch.Tensor): The predicted residual action.
        """
        if argmax:
            return self.actor.get_eval_action(actor_input)
        else:
            return self.actor.get_action(actor_input)['sample']


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
