from typing import Dict, Union
import logging
import torch
import torch.nn.functional as F
from stable_baselines3.common.type_aliases import ReplayBufferSamples

from zprl.policy.base_image_policy import BaseImagePolicy
from zprl.model.common.module_attr_mixin import ModuleAttrMixin
from zprl.model.common.shape_util import assert_shape
from zprl.model.online_td3 import Actor
from zprl.model.online import BatchedSoftQNet

logger = logging.getLogger(__name__)


class DirectPolicy(ModuleAttrMixin):
    def __init__(self,
            # network params
            obs_dim: int,
            action_dim: int,
            hidden_dim: int = 256,
            # training params
            gamma: float = 0.97,
            tau: float = 0.01,
            bc_loss_weight: float = 0.0,
            base_action_dropout: bool = True,
            base_action_dropout_prob: float = 0.5,
            fixed_std: float = 0.03,
            stddev_clip: float = 0.1,
            # batched-q params
            num_qs: int = 2,
            num_subset: int = 2,):
        super().__init__()

        actor = Actor(
            obs_dim=obs_dim + action_dim,
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
        self.bc_loss_weight = bc_loss_weight
        self.base_action_dropout = base_action_dropout
        self.base_action_dropout_prob = base_action_dropout_prob
        self.fixed_std = fixed_std
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

    def _make_actor_input(self, obs_emb, base_naction):
        return torch.cat([obs_emb, base_naction], dim=-1)

    def compute_critic_loss(self, batch: ReplayBufferSamples, stddev: Union[float, None] = None):
        bs = batch.rewards.shape[0]
        naction, _, base_next_naction = \
            torch.split(batch.actions, self.action_dim, dim=-1)

        # compute targets
        with torch.no_grad():
            actor_input = self._make_actor_input(
                batch.next_observations, base_next_naction)
            next_naction = self.predict_naction(actor_input, stddev=stddev)

            target_q_all = self.q_targets(batch.next_observations, next_naction)
            subset_indices = torch.randperm(self.num_qs, device=target_q_all.device)[:self.num_subset]
            target_q_subset = target_q_all[subset_indices]

            target_q_next = torch.min(target_q_subset, dim=0).values  # (B,1)
            assert_shape(target_q_next, (bs, 1))
            target_q = batch.rewards.flatten() + (1 - batch.dones.flatten()) * self.gamma * target_q_next.view(-1) # (B,)
            target_q = target_q.unsqueeze(0).expand(self.num_qs, -1) # broadcast to (num_qs, B)
            assert_shape(target_q, (self.num_qs, bs))

        # compute current Q values
        all_q_preds = self.qs(batch.observations, naction).squeeze(-1)  # (num_qs, B)

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

    def compute_actor_loss(self, batch: ReplayBufferSamples, stddev: Union[float, None] = None):
        bs = batch.rewards.shape[0]
        _, base_naction, _ = \
            torch.split(batch.actions, self.action_dim, dim=-1)

        base_naction_input = base_naction
        dropout_mask = None
        if self.base_action_dropout and self.base_action_dropout_prob > 0:
            dropout_mask = torch.rand(bs, device=base_naction.device) < self.base_action_dropout_prob
            base_naction_input = base_naction.clone()
            base_naction_input[dropout_mask] = 0.0

        actor_input = self._make_actor_input(batch.observations, base_naction_input)
        na_mean = self.actor(actor_input)
        if stddev is not None and stddev > 0:
            naction = self.actor.sample(actor_input, stddev=stddev, clip=self.stddev_clip)
        else:
            naction = na_mean

        all_q_preds = self.qs(batch.observations, naction)  # (num_qs, B, 1)
        predicted_q = torch.mean(all_q_preds, dim=0)  # (B, 1)
        assert_shape(predicted_q, (bs, 1))

        actor_rl_loss = -predicted_q.mean()
        bc_loss = ((naction - base_naction) ** 2).mean(dim=-1).mean()
        actor_loss = actor_rl_loss + self.bc_loss_weight * bc_loss

        info = {
            'n_rms': torch.sqrt((naction ** 2).mean()).item(),
            'base_n_rms': torch.sqrt((base_naction ** 2).mean()).item(),
            'delta_n_rms': torch.sqrt(((naction - base_naction) ** 2).mean()).item(),
            'delta_mean_rms': torch.sqrt(((na_mean - base_naction) ** 2).mean()).item(),
            'bc_loss': bc_loss.item(),
            'actor_rl_loss': actor_rl_loss.item(),
        }

        return actor_loss, info

    def target_update(self,):
        for param, target_param in zip(self.qs.parameters(), self.q_targets.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def predict_naction(
            self, actor_input: torch.Tensor,
            stddev: Union[float, None] = None) -> torch.Tensor:
        if stddev is None or stddev == 0:
            return self.actor.get_eval_action(actor_input)
        else:
            return self.actor.sample(actor_input, stddev=stddev, clip=self.stddev_clip)


class DirectActionPolicy:
    def __init__(self,
            # dimensions
            obs_emb_dim: int,
            action_dim: int,
            n_action_steps: int,
            # policies
            base_policy: BaseImagePolicy,
            direct_policy: DirectPolicy,):
        self.base_policy = base_policy
        self.direct_policy = direct_policy
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
        self.direct_policy.eval()

    def train(self):
        self.direct_policy.train()

    @property
    def device(self) -> torch.device:
        return self.direct_policy.device

    @property
    def dtype(self) -> torch.dtype:
        return self.direct_policy.dtype

    @torch.no_grad()
    def predict_action(self,
            obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        normalizer = self.base_policy.normalizer
        base_res = self.base_policy.predict_action(obs_dict)
        obs_seq_emb = base_res['obs_emb']
        base_naction = base_res['naction']
        obs_emb = obs_seq_emb[:, -self.obs_emb_dim:]

        direct_input = torch.cat(
            [obs_emb, base_naction.flatten(start_dim=1)], dim=-1)
        naction = self.direct_policy.predict_naction(
            direct_input).reshape(-1, self.n_action_steps, self.action_dim)

        action = normalizer['action'].unnormalize(naction)

        # check shapes
        assert_shape(action, (None, self.n_action_steps, self.action_dim))

        return {
            'action': action,
        }

    @torch.no_grad()
    def predict_train_action(self,
            base_naction: torch.Tensor,
            obs_emb: torch.Tensor,
            stddev: Union[float, None] = None,
            action_mask: Union[torch.Tensor, None] = None
            ) -> Dict[str, torch.Tensor]:
        """
        Compute the final action and intermediate results given base info and masks.

        Args:
            base_naction (torch.Tensor): normalized act seq by base, (B, Ta, da)
            obs_emb(torch.Tensor): o_t embedding by base, (B,do)
            stddev (float): fixed exploration stddev for the TD3 actor.
            action_mask (torch.Tensor): True where direct action is masked, (B,)

        Returns:
            Dict[str, torch.Tensor]
        """
        # 1. forward the direct policy
        direct_input = torch.cat(
            [obs_emb, base_naction.flatten(start_dim=1)], dim=-1)
        naction_flat = self.direct_policy.predict_naction(direct_input, stddev=stddev)
        base_naction_flat = base_naction.flatten(start_dim=1)
        if action_mask is not None:
            naction_flat[action_mask] = base_naction_flat[action_mask]

        # 2. compute the env action
        naction = naction_flat.reshape_as(base_naction)
        action = self.normalizer['action'].unnormalize(naction)

        return {
            'action': action, # (B,Ta,da), for env step
            'naction': naction,
            'naction_flat': naction_flat, # (B,Ta*da)
        }
