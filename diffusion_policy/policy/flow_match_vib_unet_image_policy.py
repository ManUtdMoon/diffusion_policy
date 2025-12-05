from typing import Dict
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from diffusion_policy.common.pytorch_util import dict_apply

logger = logging.getLogger(__name__)

class VIBEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim=256, alpha=1.0):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU()
        )
        self.mean_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)
        self.alpha = alpha

        logger.info(
            "number of parameters: %.2f M", sum(p.numel() for p in self.parameters()) / 1e6
        )

    def forward(self, x):
        features = self.network(x)
        mean = self.mean_head(features)
        base_logvar = self.logvar_head(features)
        # Apply alpha scaling here
        final_logvar = base_logvar + 2 * torch.log(torch.tensor(self.alpha, device=base_logvar.device))
        return mean, final_logvar

class VIBDecoder(nn.Module):
    def __init__(self, latent_dim, output_dim, hidden_dim=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )

        logger.info(
            "number of parameters: %.2f M", sum(p.numel() for p in self.parameters()) / 1e6
        )

    def forward(self, z):
        return self.network(z)

class FlowMatchVibUnetImagePolicy(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: FlowMatchEulerDiscreteScheduler,
            obs_encoder: MultiImageObsEncoder,
            horizon, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            obs_as_global_cond=True,
            diffusion_step_embed_dim=256,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
            vib_latent_dim=16,
            vib_alpha=2.0,
            vib_beta=1e-3,
            vib_recon=0.1,
            vib_hidden_dim=256,
            # parameters passed to step
            **kwargs):
        super().__init__()

        # parse shapes
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        # get feature dim
        obs_feature_dim = obs_encoder.output_shape()[0]

        # create diffusion model
        assert obs_as_global_cond
        input_dim = action_dim
        global_cond_dim = obs_feature_dim * n_obs_steps

        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale
        )

        self.obs_encoder = obs_encoder
        self.model = model
        
        # VIB components
        self.vib_encoder = VIBEncoder(
            input_dim=global_cond_dim, 
            latent_dim=vib_latent_dim,
            hidden_dim=vib_hidden_dim,
            alpha=vib_alpha)
        self.vib_decoder = VIBDecoder(
            latent_dim=vib_latent_dim,
            output_dim=global_cond_dim,
            hidden_dim=vib_hidden_dim)
        self.vib_beta = vib_beta

        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs

        self.vib_alpha = vib_alpha
        self.vib_beta = vib_beta
        self.vib_recon = vib_recon
        self.vib_latent_dim = vib_latent_dim

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        self.timesteps = self.noise_scheduler.timesteps.clone()
    
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
    
        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(trajectory, t, 
                local_cond=local_cond, global_cond=global_cond)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample
        
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        

        return trajectory
    
    def encode_obs(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        nobs = self.normalizer.normalize(obs_dict)
        B, To = next(iter(nobs.values())).shape[:2]
        batched_nobs = dict_apply(
            nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
        obs_emb = self.obs_encoder(batched_nobs) # (B*To, do)
        obs_emb = obs_emb.reshape(B, -1) # (B, Do=To*do)
        return obs_emb
    
    def vib_forward(self, global_cond, deterministic=False):
        # 1. Get final logvar directly from encoder
        z_mean, z_logvar = self.vib_encoder(global_cond)
        
        if deterministic:
            z = z_mean
        else:
            # 2. Sample using the final variance (reparameterization trick)
            std = torch.exp(0.5 * z_logvar)
            eps = torch.randn_like(std)
            z = z_mean + eps * std

        modified_global_cond = self.vib_decoder(z)
        
        # 3. Return the final logvar for KL loss calculation
        return modified_global_cond, z_mean, z_logvar, z

    def conditional_predict(self, obs_emb: torch.Tensor) -> Dict[str, torch.Tensor]:
        B = obs_emb.shape[0]
        T = self.horizon
        Da = self.action_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # condition through global feature
        global_cond = obs_emb
        # empty data for action
        cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        # run sampling
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            local_cond=None,
            global_cond=global_cond,
            **self.kwargs)
        
        # unnormalize prediction
        naction_pred = nsample
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred,
            'naction': naction_pred[:,start:end],
            'naction_pred': naction_pred,
        }
        return result

    def conditional_predict_from_noise(self, obs_emb: torch.Tensor, noise: torch.Tensor) -> Dict[str, torch.Tensor]:
        B = obs_emb.shape[0]
        T = self.horizon
        Da = self.action_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # run sampling from noise
        model = self.model
        scheduler = self.noise_scheduler

        # Check if noise shape is correct
        assert noise.shape == (B, T, Da)
        trajectory = noise.to(device=device, dtype=dtype)

        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. predict model output
            model_output = model(trajectory, t,
                local_cond=None, global_cond=obs_emb)

            # 2. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory,
                **self.kwargs
            ).prev_sample

        # unnormalize prediction
        naction_pred = trajectory
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred,
            'naction': naction_pred[:,start:end],
            'naction_pred': naction_pred,
        }
        return result

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert self.obs_as_global_cond, "VIB policy only supports obs_as_global_cond=True"
        
        # 1. encode observation
        obs_emb = self.encode_obs(obs_dict)
        
        # 2. pass through VIB module
        modified_obs_emb, _, _, _ = self.vib_forward(obs_emb, deterministic=True)
        
        # 3. predict action
        result = self.conditional_predict(modified_obs_emb)
        
        # 4. append embeddings to result for debugging
        result['obs_emb'] = obs_emb
        result['modified_obs_emb'] = modified_obs_emb
        
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # handle different ways of passing observation
        trajectory = nactions
        cond_data = trajectory
        assert self.obs_as_global_cond, "VIB policy only supports obs_as_global_cond=True"
        
        # reshape B, T, ... to B*T
        this_nobs = dict_apply(nobs, 
            lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs)
        # reshape back to B, Do
        global_cond = nobs_features.reshape(batch_size, -1)

        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        # Sample a random timestep for each image
        if self.timesteps.device != trajectory.device:
            self.timesteps = self.timesteps.to(trajectory.device)
        timestep_idxs = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (batch_size,), device=trajectory.device
        ).long()
        timesteps = self.timesteps[timestep_idxs]
        # Add noise to the clean images by linear interpolation
        noisy_trajectory = self.noise_scheduler.scale_noise(
            trajectory, timesteps, noise)

        # apply conditioning
        noisy_trajectory[condition_mask] = cond_data[condition_mask]
        
        # target for flow matching
        target = noise - trajectory

        # --- VIB Forward Pass ---
        # Decouple il and vib training
        modified_global_cond, z_mean, z_logvar, _ = self.vib_forward(global_cond.detach(), deterministic=False)

        # --- IL Flow Loss (still conditioned on original obs_emb) ---
        pred_il = self.model(noisy_trajectory, timesteps, global_cond=global_cond)
        il_loss = F.mse_loss(pred_il, target)

        # --- VIB KL Loss ---
        # KL divergence loss to regularize the latent space
        vib_kl_loss = -0.5 * torch.mean(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())
        # --- VIB IL Loss ---
        pred_vib_il = self.model(noisy_trajectory, timesteps, global_cond=modified_global_cond)
        vib_il_loss = F.mse_loss(pred_vib_il, target)
        # --- VIB reconstruction loss ---, disabled for now
        # vib_recon_loss = F.mse_loss(modified_global_cond, global_cond.detach())
        
        # Total loss
        loss = il_loss
        vib_loss = vib_il_loss + self.vib_beta * vib_kl_loss # + self.vib_recon * vib_recon_loss

        # logging
        info = {
            # 'il_loss': il_loss.item(),
            'vib_loss': vib_loss.item(),
            'vib_il_loss': vib_il_loss.item(),
            'vib_kl_loss': vib_kl_loss.item(),
            # 'vib_recon_loss': vib_recon_loss.item(),
        }

        return loss, vib_loss, info