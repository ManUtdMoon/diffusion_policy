"""
CondResPolicy + CondSumPolicy

Instead of outputting action residuals (like ResiduePolicy/SumPolicy),
CondResPolicy outputs obs_emb residuals. The perturbed embedding
    sum_obs_emb = obs_emb + emb_scale * res_emb
is then fed to the frozen flow model (conditional_predict) to generate actions.

Q function signature:  Q(obs_emb, sum_obs_emb)
  — second argument is the *summed* embedding, not the raw residual,
    so the critic directly sees where in embedding space the policy moved to.

Replay buffer layout (differs from ResiduePolicy):
    observations:      obs_emb       (Do,)   Do = To * do
    actions:           res_emb       (Do,)   same dim as obs  (raw residual, pre-scale)
    next_observations: next_obs_emb  (Do,)
"""

from typing import Dict, Optional
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.type_aliases import ReplayBufferSamples

from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from diffusion_policy.model.common.shape_util import assert_shape
from diffusion_policy.model.online import Actor, BatchedSoftQNet
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.policy.flow_match_unet_image_policy import FlowMatchUnetImagePolicy

logger = logging.getLogger(__name__)


class CondResPolicy(ModuleAttrMixin):
    """
    SAC-style RL policy operating in obs_emb space.

    State  : obs_emb      (B, Do)  — full global conditioning embedding from the base policy
    Action : res_emb      (B, Do)  — additive residual, tanh-bounded to [-1, 1]
    Q      : Q(obs_emb, sum_obs_emb)
               where sum_obs_emb = obs_emb + emb_scale * res_emb

    The critic sees (where we came from, where we landed), which gives it a richer
    signal than just seeing the raw residual offset.

    The flow model itself is NOT part of this class; CondSumPolicy owns the coupling.
    """

    def __init__(self,
            obs_dim: int,           # Do = To * do (full global cond dim)
            emb_scale: float = 0.1,
            hidden_dim: int = 256,
            log_std_min: float = -10.0,
            log_std_max: float = 2.0,
            gamma: float = 0.97,
            tau: float = 0.01,
            init_alpha: float = 0.01,
            auto_alpha: bool = True,
            num_qs: int = 2,
            num_subset: int = 2,):
        super().__init__()

        # Actor: obs_emb → res_emb  (same dim, tanh-squashed)
        actor = Actor(
            obs_dim=obs_dim,
            action_dim=obs_dim,
            hidden_dim=hidden_dim,
            log_std_min=log_std_min,
            log_std_max=log_std_max,
        )

        # Critic: Q(obs_emb, sum_obs_emb) → scalar value in [0, 1]
        # Both inputs have shape (Do,), so action_dim=obs_dim here means sum_obs_emb dim.
        qs = BatchedSoftQNet(
            obs_dim=obs_dim, action_dim=obs_dim,
            num_qs=num_qs, hidden_dim=hidden_dim)
        q_targets = BatchedSoftQNet(
            obs_dim=obs_dim, action_dim=obs_dim,
            num_qs=num_qs, hidden_dim=hidden_dim)
        q_targets.load_state_dict(qs.state_dict())
        q_targets.requires_grad_(False)

        log_alpha = nn.Parameter(
            torch.log(torch.tensor(init_alpha, dtype=torch.float32)))
        target_entropy = 0.0  # heuristic: half the embedding dim

        self.actor = actor
        self.qs = qs
        self.q_targets = q_targets
        self.log_alpha = log_alpha

        self.gamma = gamma
        self.tau = tau
        self.init_alpha = init_alpha
        self.auto_alpha = auto_alpha
        self.target_entropy = target_entropy
        self.emb_scale = emb_scale

        self.obs_dim = obs_dim
        self.num_qs = num_qs
        self.num_subset = num_subset

        logger.info(
            "CondResPolicy number of parameters: %.2f M",
            sum(p.numel() for p in self.parameters()) / 1e6,
        )

    # ========= optimizers =========

    def get_optimizer(self, policy_lr, q_lr):
        return {
            'actor_optimizer': torch.optim.Adam(self.actor.parameters(), lr=policy_lr),
            'q_optimizer':     torch.optim.Adam(self.qs.parameters(), lr=q_lr),
            'alpha_optimizer': torch.optim.Adam([self.log_alpha], lr=q_lr),
        }

    # ========= shared forward =========

    def _sample_res_emb_log_prob(self, obs_emb: torch.Tensor):
        """
        Sample res_emb ~ π(·|obs_emb) and return its log-probability.

        Args:
            obs_emb: (B, Do)
        Returns:
            res_emb:  (B, Do)  tanh-bounded in [-1, 1]
            log_prob: (B, 1)
        """
        bs = obs_emb.shape[0]
        res = self.actor.get_action(obs_emb)
        assert_shape(res['sample'],   (bs, self.obs_dim))
        assert_shape(res['log_prob'], (bs, 1))
        return res['sample'], res['log_prob']

    # ========= loss functions =========

    def compute_critic_loss(self, batch: ReplayBufferSamples, dist=None):
        """
        SAC critic loss.

        Buffer layout assumed:
            batch.observations      = obs_emb      (B, Do)
            batch.actions           = res_emb      (B, Do)  raw residual (pre-scale)
            batch.next_observations = next_obs_emb (B, Do)

        Q inputs:
            current:  Q(obs_emb,      obs_emb + emb_scale * res_emb)
            target:   Q(next_obs_emb, next_obs_emb + emb_scale * next_res_emb)
        """
        bs = batch.rewards.shape[0]
        res_emb = batch.actions  # (B, Do)

        alpha = self.init_alpha
        if self.auto_alpha:
            alpha = self.log_alpha.exp().item()

        # ---------- Bellman targets ----------
        with torch.no_grad():
            next_res_emb, next_log_prob = self._sample_res_emb_log_prob(
                batch.next_observations)
            next_sum_obs_emb = batch.next_observations + self.emb_scale * next_res_emb

            target_q_all = self.q_targets(batch.next_observations, next_sum_obs_emb)
            subset_idx = torch.randperm(
                self.num_qs, device=target_q_all.device)[:self.num_subset]
            target_q_next = torch.min(target_q_all[subset_idx], dim=0).values  # (B, 1)
            assert_shape(target_q_next, (bs, 1))

            target_q = (
                batch.rewards.flatten()
                + (1 - batch.dones.flatten()) * self.gamma * target_q_next.view(-1)
            )  # (B,)
            target_q = target_q.unsqueeze(0).expand(self.num_qs, -1)  # (num_qs, B)
            assert_shape(target_q, (self.num_qs, bs))

        # ---------- current Q ----------
        sum_obs_emb = batch.observations + self.emb_scale * res_emb
        all_q_preds = self.qs(batch.observations, sum_obs_emb).squeeze(-1)  # (num_qs, B)
        critic_loss = F.mse_loss(
            all_q_preds, target_q, reduction='none').mean(dim=-1).sum()

        info = {
            'q_target':         target_q.mean().item(),
            'q_predicted':      all_q_preds.mean().item(),
            'q_predicted_min':  all_q_preds.mean(dim=0).min().item(),
            'q_predicted_max':  all_q_preds.mean(dim=0).max().item(),
            'rewards':          batch.rewards.mean().item(),
            'dones':            batch.dones.float().mean().item(),
        }
        return critic_loss, info

    def compute_actor_loss(self, batch: ReplayBufferSamples):
        bs = batch.rewards.shape[0]
        alpha = self.log_alpha.exp().item() if self.auto_alpha else self.init_alpha

        res_emb, log_prob = self._sample_res_emb_log_prob(batch.observations)
        sum_obs_emb = batch.observations + self.emb_scale * res_emb
        all_q_preds = self.qs(batch.observations, sum_obs_emb)  # (num_qs, B, 1)
        predicted_q = torch.mean(all_q_preds, dim=0)             # (B, 1)
        assert_shape(predicted_q, (bs, 1))

        actor_loss = (alpha * log_prob - predicted_q).mean()

        info = {
            'actor_entropy': -log_prob.mean().item(),
            'res_emb_rms':  res_emb.pow(2).mean(dim=-1).sqrt().mean().item(),
            'base_emb_rms': batch.observations.pow(2).mean(dim=-1).sqrt().mean().item(),
        }
        return actor_loss, info

    def compute_alpha_loss(self, batch: ReplayBufferSamples):
        with torch.no_grad():
            _, log_prob = self._sample_res_emb_log_prob(batch.observations)
        return (-self.log_alpha.exp() * (log_prob + self.target_entropy)).mean()

    # ========= target network =========

    def target_update(self):
        for param, target in zip(self.qs.parameters(), self.q_targets.parameters()):
            target.data.copy_(self.tau * param.data + (1.0 - self.tau) * target.data)

    # ========= inference =========

    def predict_res_emb(self,
            obs_emb: torch.Tensor,
            argmax: bool = False) -> torch.Tensor:
        """
        Args:
            obs_emb: (B, Do)
            argmax:  True → tanh(mean), False → stochastic sample
        Returns:
            res_emb: (B, Do) bounded in [-1, 1]
        """
        if argmax:
            return self.actor.get_eval_action(obs_emb)
        else:
            return self.actor.get_action(obs_emb)['sample']


# ---------------------------------------------------------------------------

class CondSumPolicy:
    """
    Combined inference policy: frozen base FlowMatchUnetImagePolicy + CondResPolicy.

    Forward pass:
        1. obs → obs_emb  (via base_policy.encode_obs)
        2. res_emb = cond_res_policy(obs_emb)
        3. perturbed_emb = obs_emb + emb_scale * res_emb
        4. action = base_policy.conditional_predict(perturbed_emb)

    For eval (predict_action): argmax actor, single forward.
    For train (predict_train_action): stochastic actor, supports prog-explore masking,
        returns res_emb for replay-buffer storage.
    """

    def __init__(self,
            obs_emb_dim: int,       # Do = To * do
            action_dim: int,        # da
            n_action_steps: int,    # Ta
            base_policy: FlowMatchUnetImagePolicy,
            cond_res_policy: CondResPolicy):
        self.base_policy = base_policy
        self.cond_res_policy = cond_res_policy
        # emb_scale lives in cond_res_policy so critic and inference stay in sync
        self.normalizer = base_policy.normalizer

        self.base_policy.eval()

        self.obs_emb_dim = obs_emb_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps

    def reset(self):
        pass

    def eval(self):
        self.cond_res_policy.eval()

    def train(self):
        self.cond_res_policy.train()

    @property
    def device(self) -> torch.device:
        return self.cond_res_policy.device

    @property
    def dtype(self) -> torch.dtype:
        return self.cond_res_policy.dtype

    @torch.no_grad()
    def predict_action(self,
            obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Full pipeline: raw obs → action.
        Uses argmax actor (deterministic, no exploration).
        """
        obs_emb = self.base_policy.encode_obs(obs_dict)           # (B, Do)
        res_emb = self.cond_res_policy.predict_res_emb(obs_emb, argmax=True)  # (B, Do)
        perturbed_emb = obs_emb + self.cond_res_policy.emb_scale * res_emb
        result = self.base_policy.conditional_predict(perturbed_emb)
        assert_shape(result['action'], (None, self.n_action_steps, self.action_dim))
        return {'action': result['action']}

    @torch.no_grad()
    def predict_train_action(self,
            obs_emb: torch.Tensor,
            res_mask: Optional[torch.Tensor] = None,
            ) -> Dict[str, torch.Tensor]:
        """
        Predict action for the online training loop.
        Accepts pre-computed obs_emb (already cached from base_policy.predict_action)
        to avoid redundant encoder forward passes.

        Args:
            obs_emb:  (B, Do) — full obs embedding, from base_dict['obs_emb']
            res_mask: (B,) bool — True → zero out residual (use base embedding only)

        Returns:
            action:  (B, Ta, da)  for env.step
            res_emb: (B, Do)      for replay buffer storage (post-mask)
        """
        res_emb = self.cond_res_policy.predict_res_emb(obs_emb)   # (B, Do), stochastic
        if res_mask is not None:
            res_emb[res_mask] = 0.0                                 # masked envs: base only

        perturbed_emb = obs_emb + self.cond_res_policy.emb_scale * res_emb
        result = self.base_policy.conditional_predict(perturbed_emb)

        return {
            'action':  result['action'],   # (B, Ta, da)
            'res_emb': res_emb,            # (B, Do) — store as 'action' in replay buffer
        }
