if __name__ == "__main__":
    import sys
    import pathlib

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

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
            q_ent: bool = True,
            # batched-q params
            num_qs: int = 2,
            num_subset: int = 2,
            critic_action_input: str = 'exec',
            target_clip: list = None,
            # action selection
            num_res_candidates: int = 1,
            include_base_candidate: bool = False,
            select_rollout: bool = False,
            select_backup: bool = False,
            num_base_candidates: int = 1,):
        super().__init__()

        for name, value in [('num_base_candidates', num_base_candidates),
                            ('num_res_candidates', num_res_candidates)]:
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if num_base_candidates > 1 and (num_res_candidates != 1 or not include_base_candidate):
            raise ValueError('Multiple bases require num_res_candidates=1 and include_base_candidate=True')
        if select_rollout or select_backup or num_base_candidates > 1:
            if critic_action_input != 'exec' or actor_input != 'obs_action' or q_ent:
                raise ValueError(
                    "Residual selection requires critic_action_input='exec', "
                    "actor_input='obs_action' and q_ent=False")
            if not 1 <= num_subset <= num_qs:
                raise ValueError('Residual selection requires 1 <= num_subset <= num_qs')

        if target_clip is not None:
            if len(target_clip) != 2:
                raise ValueError('target_clip must contain [lb, ub]')
            target_clip = tuple(float(bound) for bound in target_clip)
            if not target_clip[0] <= target_clip[1]:
                raise ValueError('target_clip lb must not exceed the ub')

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

        critic_action_dim = action_dim * (2 if critic_action_input == 'base_res' else 1)
        qs = BatchedSoftQNet(obs_dim=obs_dim, action_dim=critic_action_dim, num_qs=num_qs, hidden_dim=hidden_dim)
        q_targets = BatchedSoftQNet(obs_dim=obs_dim, action_dim=critic_action_dim, num_qs=num_qs, hidden_dim=hidden_dim)
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
        self.q_ent = q_ent
        self.critic_action_input = critic_action_input
        self.target_clip = target_clip
        self.num_res_candidates = num_res_candidates
        self.include_base_candidate = include_base_candidate
        self.select_rollout = select_rollout
        self.select_backup = select_backup
        self.num_base_candidates = num_base_candidates

        # dimensions
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_qs = num_qs
        self.num_subset = num_subset

        logger.info(
            "Qs parameters: %.2f M", sum(p.numel() for p in self.qs.parameters()) / 1e6
        )
        logger.info(
            "Actor parameters: %.2f M", sum(p.numel() for p in self.actor.parameters()) / 1e6
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

        return res['sample'], res['log_prob'], res['log_std']

    def _critic_values(self, critic, obs, base_naction, res_naction):
        if self.critic_action_input == 'base_res':
            action = torch.cat([base_naction, res_naction], dim=-1)
        else:
            action = base_naction + self.res_scale * res_naction
        return critic(obs, action)

    @torch.no_grad()
    def _select_res_naction(self, obs, base_naction):
        """Single-base compatibility interface: return unscaled residual and Q."""
        result = self.select_base_res_pair(obs, base_naction[:, None, :])
        return result['res_naction'], result['q']

    @torch.no_grad()
    def select_base_res_pair(self, obs, base_bank):
        """Select among [base_0,...,base_K-1,edited_0,...,edited_K-1].

        Ties prefer the first original base. K=1 also supports stage-one K_res.
        Candidate type is determined by index, even if a sampled edit is zero.
        A one-element bank also supports the legacy single-base interface.
        """
        bs = obs.shape[0]
        k = base_bank.shape[1]
        assert k in (1, self.num_base_candidates)
        assert_shape(obs, (bs, self.obs_dim))
        assert_shape(base_bank, (bs, k, self.action_dim))
        actor_input = torch.cat([
            obs[:, None, :].expand(-1, k, -1), base_bank], dim=-1).reshape(bs * k, -1)
        residuals = self.actor.sample_candidates(actor_input, self.num_res_candidates)
        if k == 1:
            bases = base_bank.expand(-1, self.num_res_candidates, -1)
            if self.include_base_candidate:
                residuals = torch.cat([torch.zeros_like(residuals[:, :1]), residuals], dim=1)
                bases = base_bank.expand_as(residuals)
            num_original = int(self.include_base_candidate)
        else:
            residuals = residuals.reshape(bs, k, self.action_dim)
            residuals = torch.cat([torch.zeros_like(residuals), residuals], dim=1)
            bases = torch.cat([base_bank, base_bank], dim=1)
            num_original = k
        # Score every candidate with the same target-Q subset, then gather its pair.
        num_candidates = residuals.shape[1]
        actions = bases + self.res_scale * residuals
        target_q_all = self.q_targets(
            obs[:, None, :].expand(-1, num_candidates, -1).reshape(-1, self.obs_dim),
            actions.reshape(-1, self.action_dim))
        subset_indices = torch.randperm(self.num_qs, device=target_q_all.device)[:self.num_subset]
        scores = target_q_all[subset_indices].min(dim=0).values.reshape(bs, num_candidates)
        best_q, indices = scores.max(dim=1, keepdim=True)
        gather_indices = indices.unsqueeze(-1).expand(-1, -1, self.action_dim)
        indices = indices.squeeze(1)
        return {
            'base_naction': bases.gather(1, gather_indices).squeeze(1),
            'res_naction': residuals.gather(1, gather_indices).squeeze(1),
            'q': best_q,
            'selected_index': indices,
            'selected_is_res': indices >= num_original,
            'base_index': indices % k,
        }

    def _split_replay_actions(self, actions):
        return torch.split(
            actions, 
            [
                self.action_dim,
                self.action_dim,
                self.num_base_candidates * self.action_dim
            ],
            dim=-1
        )

    def compute_critic_loss(self, batch: ReplayBufferSamples, dist=None):
        bs = batch.rewards.shape[0]
        res_naction, base_naction, base_next_naction = \
            self._split_replay_actions(batch.actions)

        alpha = self.init_alpha
        if self.auto_alpha:
            alpha = self.log_alpha.detach().exp()

        # compute targets
        selection_info = {}
        with torch.no_grad():
            if self.num_base_candidates > 1 and self.select_backup:
                valid = batch.dones.flatten() == 0
                target_q_next = batch.rewards.new_zeros((bs, 1))
                ratio = batch.rewards.new_zeros(())
                if valid.any():
                    bank = base_next_naction.reshape(bs, self.num_base_candidates, self.action_dim)
                    selected = self.select_base_res_pair(batch.next_observations[valid], bank[valid])
                    target_q_next[valid] = selected['q']
                    ratio = selected['selected_is_res'].float().mean()
                selection_info = {'backup_res_selection_ratio': ratio}
            elif self.select_backup:
                _, target_q_next = self._select_res_naction(batch.next_observations, base_next_naction)
            else:
                base_next_naction = base_next_naction[:, :self.action_dim]
                actor_input = batch.next_observations
                if self.actor_input == 'obs_action':
                    actor_input = torch.cat([batch.next_observations, base_next_naction], dim=-1)
                res_next_naction, next_log_prob, _ = self._sample_naction_log_prob(actor_input)
                target_q_all = self._critic_values(
                    self.q_targets, batch.next_observations, base_next_naction, res_next_naction)
                subset_indices = torch.randperm(self.num_qs, device=target_q_all.device)[:self.num_subset]
                target_q_subset = target_q_all[subset_indices]

                target_q_next = torch.min(target_q_subset, dim=0).values  # (B,1)
                if self.q_ent:
                    target_q_next -= alpha * next_log_prob
            assert_shape(target_q_next, (bs, 1))
            discounts = self.gamma if batch.discounts is None else batch.discounts.flatten()
            if isinstance(discounts, torch.Tensor):
                assert_shape(discounts, (bs,))
            target_q = batch.rewards.flatten() + (1 - batch.dones.flatten()) * discounts * target_q_next.view(-1) # (B,)
            if self.target_clip is not None:
                target_q = target_q.clamp(*self.target_clip)
            target_q = target_q.unsqueeze(0).expand(self.num_qs, -1) # broadcast to (num_qs, B)
            assert_shape(target_q, (self.num_qs, bs))

        # compute current Q values
        all_q_preds = self._critic_values(
            self.qs, batch.observations, base_naction, res_naction).squeeze(-1)  # (num_qs, B)

        # compute critic loss
        critic_loss = F.mse_loss(
            all_q_preds, target_q, reduction='none'
        ).mean(dim=-1).sum()

        info = {
            'q_target': target_q.mean().detach(),
            'q_predicted': all_q_preds.mean().detach(),
            'q_predicted_min': all_q_preds.mean(dim=0).min().detach(),
            'q_predicted_max': all_q_preds.mean(dim=0).max().detach(),
            "rewards": batch.rewards.mean().detach(),
            "dones": batch.dones.float().mean().detach(),
        }
        info.update(selection_info)

        return critic_loss, info

    def compute_actor_loss(self, batch: ReplayBufferSamples):
        bs = batch.rewards.shape[0]
        _, base_naction, _ = \
            self._split_replay_actions(batch.actions)

        if self.auto_alpha:
            alpha = self.log_alpha.detach().exp()
        else:
            alpha = self.init_alpha

        actor_input = batch.observations
        if self.actor_input == 'obs_action':
            actor_input = torch.cat([batch.observations, base_naction], dim=-1)
        res_naction, log_prob, log_std = self._sample_naction_log_prob(actor_input)

        all_q_preds = self._critic_values(
            self.qs, batch.observations, base_naction, res_naction)  # (num_qs, B, 1)
        predicted_q = torch.mean(all_q_preds, dim=0)  # (B, 1)
        assert_shape(predicted_q, (bs, 1))

        actor_loss = (alpha * log_prob - predicted_q).mean()

        info = {
            'actor_entropy': -log_prob.mean().detach(),
            'res_naction_norm': torch.norm(res_naction, dim=-1).mean().detach(),
            'res_n_rms': torch.sqrt((res_naction ** 2).mean()).detach(),
            'base_n_rms': torch.sqrt((base_naction ** 2).mean()).detach(),
            'log_std': log_std.mean().detach(),
        }

        return actor_loss, info

    def compute_alpha_loss(self, batch):
        bs = batch.rewards.shape[0]
        _, base_naction, _ = \
            self._split_replay_actions(batch.actions)

        actor_input = batch.observations
        if self.actor_input == 'obs_action':
            actor_input = torch.cat([batch.observations, base_naction], dim=-1)

        with torch.no_grad():
            _, log_prob, _ = self._sample_naction_log_prob(actor_input)

        alpha_loss = (-self.log_alpha * (log_prob + self.target_entropy).detach()).mean()

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
            argmax (bool): With selection disabled, True returns mean and False a sample.
                With select_rollout enabled, both training and evaluation select among
                random candidates; argmax does not change the sampling mode.
            
        Returns
            res_naction (torch.Tensor): The predicted residual action.
        """
        if self.select_rollout:
            obs, base_naction = torch.split(actor_input, [self.obs_dim, self.action_dim], dim=-1)
            return self._select_res_naction(obs, base_naction)[0]
        if argmax:
            return self.actor.get_eval_action(actor_input)
        else:
            return self.actor.get_action(actor_input)['sample']


class SumPolicy:
    def __init__(self,
            # dimensions
            res_scale: float,
            base_obs_emb_dim: int,
            subtask_dim: int,
            action_dim: int,
            n_action_steps: int,
            # policies
            base_policy: BaseImagePolicy,
            res_policy: ResiduePolicy,):
        self.base_policy = base_policy
        self.res_policy = res_policy
        self.res_scale = res_scale
        self.normalizer = base_policy.normalizer
        self.num_base_candidates = getattr(res_policy, 'num_base_candidates', 1)
        if self.num_base_candidates > 1:
            self.base_policy.requires_grad_(False)

        # make sure the base policy is in eval mode
        self.base_policy.eval()

        # store dimensions
        self.base_obs_emb_dim = base_obs_emb_dim
        self.subtask_dim = subtask_dim
        self.obs_emb_dim = base_obs_emb_dim + subtask_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        assert self.res_policy.obs_dim == self.obs_emb_dim

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
    def encode_obs(self, obs_dict: Dict[str, torch.Tensor]):
        base_obs_dict = obs_dict
        if self.subtask_dim > 0:
            stage_mask = obs_dict['completed_stage_mask'][:, -1]
            assert_shape(stage_mask, (None, self.subtask_dim))
            base_obs_dict = dict(obs_dict)
            del base_obs_dict['completed_stage_mask']

        if self.num_base_candidates > 1:
            decoder_emb = self.base_policy.encode_obs(base_obs_dict)
            base_res = self.base_policy.conditional_predict_k(decoder_emb, self.num_base_candidates)
        else:
            base_res = self.base_policy.predict_action(base_obs_dict)
        obs_seq_emb = base_res['obs_emb']
        obs_emb = obs_seq_emb[:, -self.base_obs_emb_dim:]
        if self.subtask_dim > 0:
            obs_emb = torch.cat([obs_emb, stage_mask], dim=-1)
        assert_shape(obs_emb, (None, self.obs_emb_dim))
        return base_res, obs_emb

    @torch.no_grad()
    def predict_action(self,
            obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        normalizer = self.base_policy.normalizer
        base_res, obs_emb = self.encode_obs(obs_dict)
        base_naction = base_res['naction']
        if self.num_base_candidates > 1:
            return {'action': self._predict_bank_action(base_naction, obs_emb, argmax=True)['action']}

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
            base_naction (torch.Tensor): normalized act seq by base, (B, Ta, da),
                or (B, K, Ta, da) when num_base_candidates > 1.
            obs_emb(torch.Tensor): augmented residual state, (B, self.obs_emb_dim)
            res_masks (torch.Tensor): True where res is masked, (B,)

        Returns:
            Dict[str, torch.Tensor]: K>1 also returns base_naction_flat, the
                actual selected base after applying res_mask, for replay.
        """
        assert_shape(obs_emb, (None, self.obs_emb_dim))
        if self.num_base_candidates > 1:
            return self._predict_bank_action(base_naction, obs_emb, res_mask=res_mask)

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

    @torch.no_grad()
    def _predict_bank_action(self, base_bank, obs_emb, argmax=False, res_mask=None):
        bs = obs_emb.shape[0]
        assert_shape(base_bank, (bs, self.num_base_candidates, self.n_action_steps, self.action_dim))
        bank_flat = base_bank.flatten(start_dim=2)
        if self.res_policy.select_rollout:
            selected = self.res_policy.select_base_res_pair(obs_emb, bank_flat)
            base_flat, res_flat = selected['base_naction'], selected['res_naction']
        else:
            base_flat = bank_flat[:, 0].clone()
            res_flat = self.res_policy.predict_res_naction(
                torch.cat([obs_emb, base_flat], dim=-1), argmax=argmax)
        if res_mask is not None:
            # Warmup/progressive exploration executes the ordinary first proposal.
            base_flat[res_mask] = bank_flat[res_mask, 0]
            res_flat[res_mask] = 0.0
        sum_naction = (base_flat + self.res_scale * res_flat).reshape(bs, self.n_action_steps, self.action_dim)
        return {
            'action': self.normalizer['action'].unnormalize(sum_naction),
            'naction': sum_naction,
            'res_naction_flat': res_flat,
            'base_naction_flat': base_flat,
        }


def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    ResiduePolicy(
        obs_dim=521,
        action_dim=20,
        actor_input='obs_action',
        hidden_dim=512,
        log_std_min=-10.0,
        log_std_max=2.0,
        gamma=0.97,
        tau=0.01,
        init_alpha=0.01,
        auto_alpha=True,
        res_scale=0.05,
        q_ent=True,
        num_qs=10,
        num_subset=2
    )


if __name__ == "__main__":
    main()
