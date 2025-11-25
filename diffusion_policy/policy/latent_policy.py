from typing import Dict, Optional, Union
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from stable_baselines3.common.type_aliases import ReplayBufferSamples

from diffusion_policy.policy.flow_match_vib_unet_image_policy import FlowMatchVibUnetImagePolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from diffusion_policy.model.common.shape_util import assert_shape
from diffusion_policy.model.online import Actor, BatchedSoftQNet


logger = logging.getLogger(__name__)


class ResiduePolicy(ModuleAttrMixin):
    def __init__(self,
            # network params
            obs_dim: int,
            z_dim: int,
            action_dim: int,
            actor_input_type: str = 'obs_action',  # 'obs' or 'obs_action'
            hidden_dim: int = 256,
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
        # The actor learns a residue of vib z, conditioned on a rich observation
        # The Q-function estimates the value of decoding a combined z
        actor = Actor(
            obs_dim=obs_dim,
            action_dim=z_dim,
            input_type=actor_input_type,
            hidden_dim=hidden_dim,)

        qs = BatchedSoftQNet(
            obs_dim=obs_dim,
            action_dim=z_dim,
            num_qs=num_qs, 
            hidden_dim=hidden_dim)
        q_targets = BatchedSoftQNet(
            obs_dim=obs_dim,
            action_dim=z_dim,
            num_qs=num_qs, 
            hidden_dim=hidden_dim)
        q_targets.load_state_dict(qs.state_dict())
        q_targets.requires_grad_(False)

        log_alpha = nn.Parameter(
            torch.log(torch.tensor(init_alpha, dtype=torch.float32))
        )
        target_entropy = -z_dim / 2 # heuristic target entropy

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
        self.actor_input_type = actor_input_type

        # dimensions
        self.obs_dim = obs_dim
        self.z_dim = z_dim
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
            actor_input (torch.Tensor): The input to the actor network, obs_emb or [obs_emb, z_mean].
        
        Returns:
            res_z (torch.Tensor): Predicted residual latent z.
            log_prob (torch.Tensor): The log probability of the residue.
        """
        bs = actor_input.shape[0]
        res = self.actor.get_action(actor_input)

        assert_shape(res['sample'], (bs, self.z_dim))
        assert_shape(res['log_prob'], (bs, 1))

        return res['sample'], res['log_prob']

    def compute_critic_loss(self, batch: ReplayBufferSamples):
        bs = batch.rewards.shape[0]
        obs_z = batch.observations
        res_z = batch.actions
        obs_z_next = batch.next_observations

        obs, z_mean = torch.split(obs_z, [self.obs_dim, self.z_dim], dim=-1)
        obs_next, z_mean_next = torch.split(obs_z_next, [self.obs_dim, self.z_dim], dim=-1)

        alpha = self.init_alpha
        if self.auto_alpha:
            alpha = self.log_alpha.exp().item()

        # compute targets
        with torch.no_grad():
            actor_input_next = _get_actor_input(self.actor_input_type, obs_next, z_mean_next)
            res_z_next, _ = self._sample_log_prob(actor_input_next)
            z_next = res_z_next * self.res_scale + z_mean_next

            target_q_all = self.q_targets(obs_next, z_next)
            subset_indices = torch.randperm(self.num_qs, device=target_q_all.device)[:self.num_subset]
            target_q_subset = target_q_all[subset_indices]

            target_q_next = torch.min(target_q_subset, dim=0).values  # (B,1)
            assert_shape(target_q_next, (bs, 1))
            target_q = batch.rewards.flatten() + (1 - batch.dones.flatten()) * self.gamma * target_q_next.view(-1) # (B,)
            target_q = target_q.unsqueeze(0).expand(self.num_qs, -1) # broadcast to (num_qs, B)
            assert_shape(target_q, (self.num_qs, bs))

        # compute current Q values
        z_curr = res_z * self.res_scale + z_mean
        all_q_preds = self.qs(obs, z_curr).squeeze(-1)  # (num_qs, B)

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
        obs_z = batch.observations
        obs, z_mean = torch.split(obs_z, [self.obs_dim, self.z_dim], dim=-1)

        if self.auto_alpha:
            alpha = self.log_alpha.exp().item()
        else:
            alpha = self.init_alpha

        actor_input = _get_actor_input(self.actor_input_type, obs, z_mean)
        res_z, log_prob = self._sample_log_prob(actor_input)

        z_curr = res_z * self.res_scale + z_mean
        all_q_preds = self.qs(obs, z_curr)  # (num_qs, B, 1)
        predicted_q = torch.mean(all_q_preds, dim=0)  # (B, 1)
        assert_shape(predicted_q, (bs, 1))

        actor_loss = (alpha * log_prob - predicted_q).mean()

        info = {
            'actor_entropy': -log_prob.mean().item(),
            'res_z_norm': torch.norm(res_z, dim=-1).mean().item(),
            'z_mean_norm': torch.norm(z_mean, dim=-1).mean().item(),
        }

        return actor_loss, info

    def compute_alpha_loss(self, batch):
        obs_z = batch.observations
        obs, z_mean = torch.split(obs_z, [self.obs_dim, self.z_dim], dim=-1)

        actor_input = _get_actor_input(self.actor_input_type, obs, z_mean)

        with torch.no_grad():
            _, log_prob = self._sample_log_prob(actor_input)

        alpha_loss = (-self.log_alpha.exp() * (log_prob + self.target_entropy)).mean()

        return alpha_loss

    def target_update(self,):
        for param, target_param in zip(self.qs.parameters(), self.q_targets.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def predict_res_z(
            self, actor_input: torch.Tensor, argmax: bool = False) -> torch.Tensor:
        """
        Predict the next residual for z based on the current actor_input.
        
        Args:
            actor_input (torch.Tensor): The input to the actor network, obs_emb or [obs_emb, base_naction].
            argmax (bool): If True, return mean; if False, return a sample.
            
        Returns
            res_z (torch.Tensor): The predicted residual for z.
        """
        if argmax:
            return self.actor.get_eval_action(actor_input)
        else:
            return self.actor.get_action(actor_input)['sample']



class SumPolicy:
    def __init__(self,
            # dimensions
            res_scale: float,
            # policies
            base_policy: FlowMatchVibUnetImagePolicy,
            res_policy: ResiduePolicy,):
        # verify and init policies
        assert hasattr(base_policy, 'vib_encoder') and hasattr(base_policy, 'vib_decoder')
        self.base_policy = base_policy
        self.res_policy = res_policy
        self.res_scale = res_scale
        self.normalizer = base_policy.normalizer
        self.actor_input_type = res_policy.actor_input_type

        # make sure the base policy is in eval mode
        self.base_policy.eval()

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
        # 1. Get latent mean from base policy's encoders
        obs_emb = self.base_policy.encode_obs(obs_dict)
        z_mean, _ = self.base_policy.vib_encoder(obs_emb)

        # 2. Construct input for res_policy and predict residual (deterministically)
        res_input = _get_actor_input(self.actor_input_type, obs_emb, z_mean)
        res_z = self.res_policy.predict_res_z(res_input, argmax=True)

        # 3. Apply residual and decode back to conditional embedding
        perturbed_z = z_mean + self.res_scale * res_z
        modified_obs_emb = self.base_policy.vib_decoder(perturbed_z)

        # 4. Generate action from the new conditional embedding
        result = self.base_policy.conditional_predict(modified_obs_emb)

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
        # 1. Get latent mean from base policy's VIB encoder
        z_mean, _ = self.base_policy.vib_encoder(obs_emb)

        # 2. Build input for res_policy and predict res z (stochastically)
        res_input = _get_actor_input(self.actor_input_type, obs_emb, z_mean)
        if perturb:
            res_z = self.res_policy.predict_res_z(res_input, argmax=False)
        else:
            res_z = torch.zeros_like(z_mean, device=z_mean.device)

        # 3. Apply residual and decode back to conditional embedding
        perturbed_z = z_mean + self.res_scale * res_z
        modified_obs_emb = self.base_policy.vib_decoder(perturbed_z)

        # 4. Generate action from the new conditional embedding
        result = self.base_policy.conditional_predict(modified_obs_emb)

        # 5. Return values for env step and replay buffer
        # 'obs_emb' is already available
        # 'z_mean' is possibly obs for the RL agent
        # 'res_z' is the 'action' for the RL agent
        result['z_mean'] = z_mean
        result['res_z'] = res_z

        return result



def _get_actor_input(
        actor_input_type: str,
        obs_emb: torch.Tensor,
        z_mean: torch.Tensor
    ) -> torch.Tensor:
    if actor_input_type == 'obs':
        return obs_emb
    elif actor_input_type == 'obs_action':
        return torch.cat([obs_emb, z_mean], dim=-1)
    else:
        raise NotImplementedError("Invalid actor_input_type")