from typing import Dict, Optional, Union
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.type_aliases import ReplayBufferSamples

from zprl.policy.base_image_policy import BaseImagePolicy
from zprl.common.pytorch_util import dict_apply
from zprl.model.common.module_attr_mixin import ModuleAttrMixin
from zprl.model.common.shape_util import assert_shape
from zprl.model.online_td3 import Actor
from zprl.model.online import BatchedSoftQNet

logger = logging.getLogger(__name__)


class TD3Policy(ModuleAttrMixin):
    def __init__(self,
            # network params
            obs_dim: int,
            action_dim: int,
            actor_input: str = 'obs_action',
            hidden_dim: int = 256,
            # training params
            gamma: float = 0.97,
            tau: float = 0.005,
            res_scale: float = 0.05,
            # noise clip
            stddev_clip: float = 0.3,
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
        )

        qs = BatchedSoftQNet(obs_dim=obs_dim, action_dim=action_dim, num_qs=num_qs, hidden_dim=hidden_dim)
        q_targets = BatchedSoftQNet(obs_dim=obs_dim, action_dim=action_dim, num_qs=num_qs, hidden_dim=hidden_dim)
        q_targets.load_state_dict(qs.state_dict())
        q_targets.requires_grad_(False)

        self.actor = actor
        self.qs = qs
        self.q_targets = q_targets

        # training params
        self.gamma = gamma
        self.tau = tau
        self.res_scale = res_scale
        self.actor_input = actor_input

        # noise clip
        self.stddev_clip = stddev_clip

        # dimensions
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_qs = num_qs
        self.num_subset = num_subset

        logger.info(
            "number of parameters: %.2f M", sum(p.numel() for p in self.parameters()) / 1e6
        )

    def get_optimizer(self, policy_lr, q_lr):
        actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=policy_lr)
        q_optimizer = torch.optim.Adam(self.qs.parameters(), lr=q_lr)
        return {
            'actor_optimizer': actor_optimizer,
            'q_optimizer': q_optimizer,
        }

    def compute_critic_loss(self, batch: ReplayBufferSamples, stddev: float):
        bs = batch.rewards.shape[0]
        res_naction, base_naction, base_next_naction = \
            torch.split(batch.actions, self.action_dim, dim=-1)

        with torch.no_grad():
            # build actor input for next state
            actor_input = batch.next_observations
            if self.actor_input == 'obs_action':
                actor_input = torch.cat([batch.next_observations, base_next_naction], dim=-1)

            # target policy smoothing
            next_res_naction = self.actor.sample(
                actor_input, stddev=stddev, clip=self.stddev_clip)

            next_naction = next_res_naction * self.res_scale + base_next_naction

            # batched ensemble target
            target_q_all = self.q_targets(batch.next_observations, next_naction)
            subset_indices = torch.randperm(self.num_qs, device=target_q_all.device)[:self.num_subset]
            target_q_subset = target_q_all[subset_indices]
            target_q_next = torch.min(target_q_subset, dim=0).values  # (B,1)
            assert_shape(target_q_next, (bs, 1))

            target_q = batch.rewards.flatten() + (1 - batch.dones.flatten()) * self.gamma * target_q_next.view(-1)
            target_q = target_q.unsqueeze(0).expand(self.num_qs, -1)
            assert_shape(target_q, (self.num_qs, bs))

        # current Q values
        current_naction = self.res_scale * res_naction + base_naction
        all_q_preds = self.qs(batch.observations, current_naction).squeeze(-1)  # (num_qs, B)

        critic_loss = F.mse_loss(
            all_q_preds, target_q, reduction='none'
        ).mean(dim=-1).sum()

        info = {
            'q_target': target_q.mean().item(),
            'q_predicted': all_q_preds.mean().item(),
            'q_predicted_min': all_q_preds.mean(dim=0).min().item(),
            'q_predicted_max': all_q_preds.mean(dim=0).max().item(),
            'rewards': batch.rewards.mean().item(),
            'dones': batch.dones.float().mean().item(),
        }

        return critic_loss, info

    def compute_actor_loss(self, batch: ReplayBufferSamples, stddev: float = 0.0):
        bs = batch.rewards.shape[0]
        _, base_naction, _ = \
            torch.split(batch.actions, self.action_dim, dim=-1)

        actor_input = batch.observations
        if self.actor_input == 'obs_action':
            actor_input = torch.cat([batch.observations, base_naction], dim=-1)
        if stddev > 0:
            res_naction = self.actor.sample(actor_input, stddev=stddev, clip=self.stddev_clip)
        else:
            res_naction = self.actor(actor_input)

        naction = self.res_scale * res_naction + base_naction
        all_q_preds = self.qs(batch.observations, naction)  # (num_qs, B, 1)
        predicted_q = torch.mean(all_q_preds, dim=0)  # (B, 1)
        assert_shape(predicted_q, (bs, 1))

        actor_loss = -predicted_q.mean()

        info = {
            'res_naction_norm': torch.norm(res_naction, dim=-1).mean().item(),
        }

        return actor_loss, info

    def target_update(self):
        for param, target_param in zip(self.qs.parameters(), self.q_targets.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def predict_res_naction(
            self, actor_input: torch.Tensor,
            stddev: Optional[float] = None) -> torch.Tensor:
        if stddev is None or stddev == 0:
            return self.actor.get_eval_action(actor_input)
        else:
            return self.actor.sample(actor_input, stddev=stddev)


class TD3SumPolicy:
    def __init__(self,
            res_scale: float,
            obs_emb_dim: int,
            action_dim: int,
            n_action_steps: int,
            base_policy: BaseImagePolicy,
            res_policy: TD3Policy,):
        self.base_policy = base_policy
        self.res_policy = res_policy
        self.res_scale = res_scale
        self.normalizer = base_policy.normalizer

        self.base_policy.eval()

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
            res_input).reshape(-1, self.n_action_steps, self.action_dim)

        sum_naction = self.res_scale * res_naction + base_naction
        sum_action = normalizer['action'].unnormalize(sum_naction)

        assert_shape(sum_action, (None, self.n_action_steps, self.action_dim))

        return {
            'action': sum_action,
        }

    @torch.no_grad()
    def predict_train_action(self,
            base_naction: torch.Tensor,
            obs_emb: torch.Tensor,
            stddev: float = 0.0,
            res_mask: Union[torch.Tensor, None] = None
            ) -> Dict[str, torch.Tensor]:
        res_input = obs_emb
        if self.res_policy.actor_input == 'obs_action':
            res_input = torch.cat(
                [obs_emb, base_naction.flatten(start_dim=1)], dim=-1)
        res_naction_flat = self.res_policy.predict_res_naction(
            res_input, stddev=stddev)
        if res_mask is not None:
            res_naction_flat[res_mask] = 0.0

        res_naction = res_naction_flat.reshape_as(base_naction)
        sum_naction = self.res_scale * res_naction + base_naction
        sum_action = self.normalizer['action'].unnormalize(sum_naction)

        return {
            'action': sum_action,
            'naction': sum_naction,
            'res_naction_flat': res_naction_flat,
        }
