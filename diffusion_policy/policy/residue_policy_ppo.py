from typing import Dict, Optional, Union
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from stable_baselines3.common.type_aliases import RolloutBufferSamples

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from diffusion_policy.model.common.shape_util import assert_shape
from diffusion_policy.model.online import ValueNet, Actor

logger = logging.getLogger(__name__)


class RunningMeanStd(nn.Module):
    """
    A module to compute running mean and standard deviation for input tensors.
    Uses buffers to ensure state is saved and moved with the parent module.
    """
    def __init__(self, shape):
        super().__init__()
        self.register_buffer('mean', torch.zeros(shape, dtype=torch.float32))
        self.register_buffer('var', torch.ones(shape, dtype=torch.float32))
        self.register_buffer('count', torch.tensor(1e-4, dtype=torch.float32))

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        # Use float64 for update step to improve precision
        x_64 = x.to(torch.float64)
        
        batch_mean = torch.mean(x_64, dim=0)
        batch_var = torch.var(x_64, dim=0, correction=0)
        batch_count = x_64.shape[0]

        delta = batch_mean - self.mean.to(torch.float64)
        tot_count = self.count.to(torch.float64) + batch_count

        new_mean = self.mean.to(torch.float64) + delta * batch_count / tot_count
        m_a = self.var.to(torch.float64) * self.count.to(torch.float64)
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + torch.square(delta) * self.count.to(torch.float64) * batch_count / tot_count
        new_var = m_2 / tot_count
        
        # In-place update of buffers, casting back to float32
        self.mean.copy_(new_mean.to(torch.float32))
        self.var.copy_(new_var.to(torch.float32))
        self.count.copy_(tot_count.to(torch.float32))

    @torch.no_grad()
    def normalize(self, x: torch.Tensor, clip_range: float = 10.0) -> torch.Tensor:
        std = torch.sqrt(self.var + 1e-8)
        normalized_x = (x - self.mean) / std
        return torch.clamp(normalized_x, -clip_range, clip_range)


class ResiduePolicyPPO(ModuleAttrMixin):
    def __init__(self,
            # network params
            obs_dim: int,
            action_dim: int,
            log_std_max: float = 2.0,
            log_std_min: float = -10.0,
            actor_input: str = 'obs_action',
            # training params
            gamma: float = 0.97,
            gae_lambda: float = 0.95,
            clip_range: float = 0.2,
            vf_coef: float = 0.5,
            res_scale: float = 0.05,):
        super().__init__()

        # create models
        actor = Actor(
            obs_dim=obs_dim,
            action_dim=action_dim,
            log_std_min=log_std_min,
            log_std_max=log_std_max,
            input_type=actor_input)
        vf = ValueNet(obs_dim=obs_dim, action_dim=action_dim, input_type=actor_input)

        self.actor = actor
        self.vf = vf

        # This policy instance should receive rms unnormalized input.
        # Then rms normalizes them, sends n_inputs to actor & vf.
        input_dim = obs_dim
        if actor_input == 'obs_action':
            input_dim += action_dim
        self.input_rms = RunningMeanStd(input_dim)

        # training params
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.vf_coef = vf_coef
        self.res_scale = res_scale
        self.actor_input = actor_input

        # dimensions
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        logger.info(
            "number of parameters: %.2f M", sum(p.numel() for p in self.parameters()) / 1e6
        )

    # training
    def get_optimizer(self, policy_lr):
        return torch.optim.Adam(self.parameters(), lr=policy_lr, eps=1e-5)

    def _sample_naction_log_prob(self, nactor_input):
        """
        shared computation between loss functions
        
        Args:
            nactor_input (torch.Tensor): The input to the actor network, rms normalized obs_emb or [obs_emb, base_naction].

        Returns:
            res_naction (torch.Tensor): Predicted residual action.
            log_prob (torch.Tensor): The log probability of the residual action.
        """
        bs = nactor_input.shape[0]
        res = self.actor.get_action(nactor_input)

        assert_shape(res['sample'], (bs, self.action_dim))
        assert_shape(res['log_prob'], (bs, 1))

        return res['sample'], res['log_prob']

    def _evaluate_action(self, npolicy_input, res_naction):
        """
        Evaluate the rollout res_naction in buffer with current policy.

        For training only (loss).
        
        Args:
            npolicy_input (torch.Tensor): The input to the actor network, rms normalized obs_emb or [obs_emb, base_naction].
            res_naction (torch.Tensor): The residual action to evaluate.
        
        Returns:
            value (torch.Tensor): The value of the state-action pair.
            log_prob (torch.Tensor): The log probability of the residual action.
            entropy (torch.Tensor): The entropy of the action distribution.
        """
        # get value
        value = self.vf(npolicy_input)  # (B,1)

        # get current normal dist
        log_prob = self.actor.log_prob_action(npolicy_input, res_naction)  # (B,1)

        return value, log_prob.squeeze(-1), -log_prob.squeeze(-1)

    def compute_loss(self, batch: RolloutBufferSamples):
        bs = batch.observations.shape[0]
        # 1. extract data
        ## 1.1 naction
        # res_naction, base_naction = torch.split(batch.actions, self.action_dim, dim=-1)
        res_naction, base_naction = batch.actions, None  # for mujoco test only

        ## 1.2 build policy input (the actual state of res_policy)
        policy_input = batch.observations
        if self.actor_input == 'obs_action':
            policy_input = torch.cat([batch.observations, base_naction], dim=-1)
        npolicy_input = self.input_rms.normalize(policy_input)

        ## 1.3 get variables of shape (B,)
        old_log_probs = batch.old_log_prob
        advantages = batch.advantages
        returns = batch.returns
        assert_shape(old_log_probs, (bs,))

        # 2. re-evaluate actions w/ current policy
        new_values, log_probs, entropies = self._evaluate_action(npolicy_input, res_naction)
        assert_shape(log_probs, (bs,))

        # 3. normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 4. compute loss
        ## 4.1 policy loss: clipped surrogate
        ratio = torch.exp(log_probs - old_log_probs)
        actor_loss = -torch.min(
            advantages * ratio,
            advantages * torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
        ).mean()

        ## 4.2 value loss
        new_values = new_values.view(-1)  # (B,)
        assert_shape(new_values, (bs,))
        critic_loss_unclipped = F.mse_loss(new_values, returns)
        ## clipped value loss
        v_clipped = batch.old_values + torch.clamp(
            new_values - batch.old_values,
            -self.clip_range,
            self.clip_range
        )
        critic_loss_clipped = F.mse_loss(v_clipped, returns)
        critic_loss = torch.max(critic_loss_unclipped, critic_loss_clipped).mean()

        ## 4.3 entropy loss
        entropy_loss = -entropies.mean()

        ## 4.4 total
        entropy_coef = 0.0
        total_loss = actor_loss + critic_loss * self.vf_coef # + entropy_loss * entropy_coef

        # 5. logging info
        with torch.no_grad():
            log_ratio = log_probs - old_log_probs
            approx_kl = ((torch.exp(log_ratio) - 1) - log_ratio).mean()
            clip_frac = (torch.abs(ratio - 1.0) > self.clip_range).float().mean()

        return total_loss, {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "approx_kl": approx_kl.item(),
            "clip_frac": clip_frac.item(),
            "value": new_values.mean().item(),
            "actor_entropy": entropies.mean().item(),
            "res_naction_norm": torch.norm(res_naction, dim=-1).mean().item(),
        }

    def predict_res_naction(
            self, actor_input: torch.Tensor, argmax: bool = False) -> torch.Tensor:
        """
        Predict the next residual action based on the current actor input.
        
        Args:
            actor_input (torch.Tensor): The input to the actor network, obs_emb or [obs_emb, base_naction]. (w/o rms normalization yet)
            argmax (bool): If True, return mean; if False, return a sample.
            
        Returns
            res_naction (torch.Tensor): The predicted residual action.
        """
        assert argmax, print("Argmax should be True. Because when we explore w/ PPO, we always need value and log_prob.")
        nactor_input = self.input_rms.normalize(actor_input)
        return self.actor.get_eval_action(nactor_input)

    def predict_value(self, policy_input: torch.Tensor) -> torch.Tensor:
        """
        Predict the value v given policy input.

        Args:
            policy_input (torch.Tensor): The input to policy, obs_emb or [obs_emb, base_naction].
                Shape: (B, do + Da)

        Returns:
            value (torch.Tensor): The predicted value of the current state. 
                Shape: (B, 1)
        """
        npolicy_input = self.input_rms.normalize(policy_input)
        return self.vf(npolicy_input)

    def predict_all(self, policy_input: torch.Tensor):
        """
        Predict next residual action, its log_prob and value given current policy input.

        For training only (exploration), evaluation should use `predict_res_naction` with `argmax=True`.

        Returns:
            res_naction (torch.Tensor): The predicted residual action.
            log_prob (torch.Tensor): The log probability of the residual action.
            value (torch.Tensor): The predicted value of the current state.
        """
        if self.training:
            self.input_rms.update(policy_input)

        npolicy_input = self.input_rms.normalize(policy_input)
        res_naction, log_prob = self._sample_naction_log_prob(npolicy_input)
        value = self.vf(npolicy_input)  # (B,1)
        return res_naction, log_prob, value
