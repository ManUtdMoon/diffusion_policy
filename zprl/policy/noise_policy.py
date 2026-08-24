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
from zprl.model.online import SquashedNormal, BatchedLayerNorm, BatchedLinear


logger = logging.getLogger(__name__)


class DSRLActor(nn.Module):
    def __init__(self,
            obs_dim: int,
            action_dim: int,
            hidden_dim: int = 256,
            log_std_min: float = -20.0,
            log_std_max: float = 2.0,):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
        )
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, x):
        x = self.net(x)
        mean = self.mean(x)
        log_std = torch.clamp(
            self.log_std(x), self.log_std_min, self.log_std_max)
        return mean, log_std

    def get_eval_action(self, x):
        mean, _ = self.forward(x)
        return torch.tanh(mean)

    def get_action(self, x):
        mean, log_std = self.forward(x)
        dist = SquashedNormal(mean, log_std.exp())

        sample = dist.rsample()
        log_prob = dist.log_prob(sample).sum(dim=-1, keepdim=True)

        assert torch.all(torch.isfinite(sample))
        assert torch.all(torch.isfinite(log_prob))

        return {
            'sample': sample,
            'mean': dist.mean,
            'log_prob': log_prob,
            'log_std': log_std,
        }

class DSRLQNet(nn.Module):
    def __init__(self, obs_dim, action_dim, num_qs, hidden_dim=256):
        super().__init__()
        self.num_qs = num_qs
        self.net = nn.Sequential(
            BatchedLinear(num_qs, obs_dim + action_dim, hidden_dim),
            BatchedLayerNorm(num_qs, hidden_dim),
            nn.Tanh(),
            BatchedLinear(num_qs, hidden_dim, hidden_dim),
            BatchedLayerNorm(num_qs, hidden_dim),
            nn.Tanh(),
            BatchedLinear(num_qs, hidden_dim, hidden_dim),
            BatchedLayerNorm(num_qs, hidden_dim),
            nn.Tanh(),
            BatchedLinear(num_qs, hidden_dim, 1),
        )

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        # expand for batch
        x = x.unsqueeze(0).expand(self.num_qs, -1, -1) # (num_qs, B, D_in)
        q_logits = self.net(x) # (num_qs, B, 1)
        # q_values = 0.5 * (torch.tanh(q_logits) + 1.)  # scale to [0, 1]
        return q_logits


class NoisePolicy(ModuleAttrMixin):
    def __init__(self,
            # network params
            obs_dim: int,
            action_dim: int,
            hidden_dim: int = 256,
            log_std_min: float = -20.0,
            log_std_max: float = 2.0,
            # training params
            gamma: float = 0.97,
            tau: float = 0.01,
            init_alpha: float = 1.0,
            auto_alpha: bool = True,
            target_entropy: float = 0.0,
            # batched-q params
            num_qs: int = 2,
            num_subset: int = 2,
            q_entropy: bool = True,):
        super().__init__()

        # create models

        actor = DSRLActor(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            log_std_min=log_std_min,
            log_std_max=log_std_max,
        )

        qs = DSRLQNet(obs_dim=obs_dim, action_dim=action_dim, num_qs=num_qs, hidden_dim=hidden_dim)
        q_targets = DSRLQNet(obs_dim=obs_dim, action_dim=action_dim, num_qs=num_qs, hidden_dim=hidden_dim)
        q_targets.load_state_dict(qs.state_dict())
        q_targets.requires_grad_(False)

        log_alpha = nn.Parameter(
            torch.log(torch.tensor(init_alpha, dtype=torch.float32))
        )
        target_entropy = -action_dim / 2. if target_entropy is None else target_entropy

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
        self.q_entropy = q_entropy

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
    
    def _sample_log_prob(self, actor_input):
        """
        shared computation between loss functions
        
        Args:
            actor_input (torch.Tensor): The input to the actor network, obs_emb.
        
        Returns:
            nnoise (torch.Tensor): Predicted init normalized noise.
            log_prob (torch.Tensor): The log probability of the action.
        """
        bs = actor_input.shape[0]
        res = self.actor.get_action(actor_input)

        assert_shape(res['sample'], (bs, self.action_dim))
        assert_shape(res['log_prob'], (bs, 1))

        return res['sample'], res['log_prob'], res['log_std']

    def compute_critic_loss(self, batch: ReplayBufferSamples):
        bs = batch.rewards.shape[0]

        alpha = self.init_alpha
        if self.auto_alpha:
            alpha = self.log_alpha.exp().item()

        # compute targets
        with torch.no_grad():
            next_nnoise, next_log_prob, _ = self._sample_log_prob(batch.next_observations)

            target_q_all = self.q_targets(batch.next_observations, next_nnoise)
            subset_indices = torch.randperm(self.num_qs, device=target_q_all.device)[:self.num_subset]
            target_q_subset = target_q_all[subset_indices]

            target_q_next = torch.min(target_q_subset, dim=0).values
            if self.q_entropy:
                target_q_next = target_q_next - alpha * next_log_prob
            assert_shape(target_q_next, (bs, 1))
            target_q = batch.rewards.flatten() + (1 - batch.dones.flatten()) * self.gamma * target_q_next.view(-1) # (B,)
            target_q = target_q.unsqueeze(0).expand(self.num_qs, -1) # broadcast to (num_qs, B)
            assert_shape(target_q, (self.num_qs, bs))

        # compute current Q values
        all_q_preds = self.qs(batch.observations, batch.actions).squeeze(-1)  # (num_qs, B)

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

        if self.auto_alpha:
            alpha = self.log_alpha.exp().item()
        else:
            alpha = self.init_alpha

        nnoise, log_prob, log_std = self._sample_log_prob(batch.observations)
        all_q_preds = self.qs(batch.observations, nnoise)  # (num_qs, B, 1)
        predicted_q = torch.min(all_q_preds, dim=0).values  # (B, 1)
        assert_shape(predicted_q, (bs, 1))

        actor_loss = (alpha * log_prob - predicted_q).mean()

        info = {
            'actor_entropy': -log_prob.mean().item(),
            'nnoise_norm': torch.norm(nnoise, dim=-1).mean().item(),
            'log_std': log_std.mean().item(),
        }

        return actor_loss, info

    def compute_alpha_loss(self, batch):
        with torch.no_grad():
            _, log_prob, _ = self._sample_log_prob(batch.observations)

        alpha_loss = (
            -self.log_alpha.exp() * (log_prob + self.target_entropy).detach()
        ).mean()

        return alpha_loss

    def target_update(self,):
        for param, target_param in zip(self.qs.parameters(), self.q_targets.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def predict_noise(
            self, actor_input: torch.Tensor, argmax: bool = False) -> torch.Tensor:
        """
        Predict the normalized noise based on the current actor input.
        
        Args:
            actor_input (torch.Tensor): The input to the actor network, obs_emb or [obs_emb, base_naction].
            argmax (bool): If True, return mean; if False, return a sample.
            
        Returns
            nnoise (torch.Tensor): The predicted normalized noise.
        """
        if argmax:
            return self.actor.get_eval_action(actor_input)
        else:
            return self.actor.get_action(actor_input)['sample']



class SumPolicy:
    def __init__(self,
            # dimensions
            noise_scale: float,
            n_noise_steps: int,
            # policies
            base_policy: BaseImagePolicy,
            noise_policy: NoisePolicy,):
        # verify and init policies
        assert hasattr(base_policy, 'encode_obs')
        assert hasattr(base_policy, 'conditional_predict_from_noise')
        self.base_policy = base_policy
        self.noise_policy = noise_policy
        self.base_policy_has_vib = hasattr(base_policy, 'vib_forward')
        assert noise_scale > 0
        assert n_noise_steps > 0
        assert base_policy.horizon % n_noise_steps == 0
        assert noise_policy.action_dim == n_noise_steps * base_policy.action_dim
        self.noise_scale = noise_scale
        self.n_noise_steps = n_noise_steps
        self.normalizer = base_policy.normalizer

        # make sure the base policy is in eval mode
        self.base_policy.eval()

    def _get_base_cond(self, obs_emb: torch.Tensor, deterministic: bool):
        if not self.base_policy_has_vib:
            return obs_emb, None
        mod_obs_emb, z_mean, _, _ = self.base_policy.vib_forward(
            obs_emb, deterministic=deterministic)
        return mod_obs_emb, z_mean

    def _repeat_noise(self, noise: torch.Tensor) -> torch.Tensor:
        B = noise.shape[0]
        da = self.base_policy.action_dim
        noise = noise.reshape(B, self.n_noise_steps, da)
        return noise.repeat(
            1, self.base_policy.horizon // self.n_noise_steps, 1)

    def reset(self):
        pass

    def eval(self):
        self.noise_policy.eval()

    def train(self):
        self.noise_policy.train()

    @property
    def device(self) -> torch.device:
        return self.noise_policy.device

    @property
    def dtype(self) -> torch.dtype:
        return self.noise_policy.dtype

    @torch.no_grad()
    def predict_action(self,
            obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # 1. Get condition from base policy
        obs_emb = self.base_policy.encode_obs(obs_dict)
        base_cond, _ = self._get_base_cond(obs_emb, deterministic=True)

        # 2. Get init noise from rl policy
        noise = self.noise_policy.predict_noise(obs_emb)
        noise = self._repeat_noise(noise)

        # 3. Apply noise and obs_emb decoding
        result = self.base_policy.conditional_predict_from_noise(
            base_cond, noise * self.noise_scale
        )

        return result

    @torch.no_grad()
    def predict_train_action(self,
            obs_emb: torch.Tensor,
            perturb: bool = True
            ) -> Dict[str, torch.Tensor]:
        """
        Used for training data collection.
        Computes the final action and returns intermediate results for the replay buffer.

        Args:
            obs_emb(torch.Tensor): Observation embedding from base_policy.encode_obs, (B, Do=To*do)

        Returns:
            Dict[str, torch.Tensor]
        """
        # 1. Get condition from base policy
        base_cond, z_mean = self._get_base_cond(
            obs_emb, deterministic=False)

        # 2. Get init noise from rl policy
        nnoise = self.noise_policy.predict_noise(obs_emb)
        if perturb:
            init_noise = nnoise * self.noise_scale
        else:
            init_noise = torch.clamp(torch.randn_like(nnoise), -self.noise_scale, self.noise_scale)
            nnoise = init_noise / self.noise_scale
        # 3. Apply noise and obs_emb decoding
        result = self.base_policy.conditional_predict_from_noise(
            base_cond, self._repeat_noise(init_noise)
        )

        # 4. Return values for env step and replay buffer
        # 'obs_emb' is already available
        # 'z_mean' is possibly obs for the RL agent
        # 'noise' is the 'action' for the RL agent
        if z_mean is not None:
            result['z_mean'] = z_mean
        result['nnoise'] = nnoise

        return result
