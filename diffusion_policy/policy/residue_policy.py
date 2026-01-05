from typing import Dict, Optional, Union
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from stable_baselines3.common.type_aliases import ReplayBufferSamples

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from diffusion_policy.model.common.shape_util import assert_shape
from diffusion_policy.model.online import Actor, BatchedSoftQNet

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
            # batched-q params
            num_qs: int = 2,
            num_subset: int = 2,):
        super().__init__()

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
            "reward_max": batch.rewards.max().item(),
            "reward_min": batch.rewards.min().item(),
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
        res_naction, log_prob = self._sample_naction_log_prob(actor_input)

        naction = self.res_scale * res_naction + base_naction  # (B,Da)
        all_q_preds = self.qs(batch.observations, naction)  # (num_qs, B, 1)
        predicted_q = torch.mean(all_q_preds, dim=0)  # (B, 1)
        assert_shape(predicted_q, (bs, 1))

        actor_loss = (alpha * log_prob - predicted_q).mean()

        info = {
            'actor_entropy': -log_prob.mean().item(),
            'res_naction_norm': torch.norm(res_naction, dim=-1).mean().item(),
            'base_norm': torch.norm(base_naction, dim=-1).mean().item(),
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
