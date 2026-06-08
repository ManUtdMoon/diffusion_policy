from typing import Dict, Literal
import math
import torch
import torch.nn.functional as F
from torch.func import vjp
from torch.func import functional_call
from einops import rearrange, reduce

from zprl.model.common.normalizer import LinearNormalizer
from zprl.model.vib import VIBDecoder, VIBEncoder
from zprl.policy.base_image_policy import BaseImagePolicy
from zprl.model.diffusion.conditional_unet1d import ConditionalUnet1D
from zprl.model.diffusion.mask_generator import LowdimMaskGenerator
from zprl.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from zprl.common.pytorch_util import dict_apply
from zprl.policy.prev_action_util import append_prev_action_cond, get_prev_action_cond, split_prev_action

PrefixAttentionSchedule = Literal["linear", "exp", "ones", "zeros"]


def get_prefix_weights_torch(
        start: int,
        end: int,
        total: int,
        schedule: PrefixAttentionSchedule,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
    start = min(start, end)
    idx = torch.arange(total, device=device)

    if schedule == "ones":
        w = torch.ones(total, device=device, dtype=dtype)
    elif schedule == "zeros":
        w = (idx < start).to(dtype)
    elif schedule in ("linear", "exp"):
        w = torch.clamp((start - 1 - idx.to(dtype)) / (end - start + 1) + 1, 0, 1)
        if schedule == "exp":
            w = w * torch.expm1(w) / (math.e - 1)
    else:
        raise ValueError(f"Invalid schedule: {schedule}")

    return torch.where(idx >= end, torch.zeros((), device=device, dtype=dtype), w)


class FlowMatchVibUnetImagePolicy(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
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
            n_prev_action_steps=0):
        super().__init__()

        # parse shapes
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        # get feature dim
        obs_feature_dim = obs_encoder.output_shape()[0]
        obs_cond_dim = obs_feature_dim * n_obs_steps
        prev_action_cond_dim = int(n_prev_action_steps) * action_dim
        model_global_cond_dim = obs_cond_dim + prev_action_cond_dim

        # create diffusion model
        assert obs_as_global_cond
        input_dim = action_dim

        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=model_global_cond_dim,
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
            input_dim=obs_cond_dim, 
            latent_dim=vib_latent_dim,
            hidden_dim=vib_hidden_dim,
            alpha=vib_alpha)
        self.vib_decoder = VIBDecoder(
            latent_dim=vib_latent_dim,
            output_dim=obs_cond_dim,
            hidden_dim=vib_hidden_dim)

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
        self.n_prev_action_steps = n_prev_action_steps
        self.obs_as_global_cond = obs_as_global_cond

        self.vib_alpha = vib_alpha
        self.vib_beta = vib_beta
        self.vib_recon = vib_recon
        self.vib_latent_dim = vib_latent_dim

        if num_inference_steps is None:
            num_inference_steps = 100
        self.num_inference_steps = num_inference_steps


    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            generator=None
            ):
        model = self.model

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
    
        timesteps = torch.linspace(
            1.0, 0.0, self.num_inference_steps + 1,
            device=trajectory.device,
            dtype=torch.float32)
        dt = -1.0 / self.num_inference_steps

        for t in timesteps[:-1]:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(trajectory, t, 
                local_cond=local_cond, global_cond=global_cond)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = trajectory + dt * model_output
        
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        

        return trajectory
    
    def encode_obs(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        obs_dict, _, _ = split_prev_action(obs_dict)
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
            global_cond=global_cond)
        
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
            'action_pred_all': action_pred[:,start:],  # B,H-To+1,Da
            'naction_pred_all': naction_pred[:,start:],  # B,H-To+1,Da
        }
        return result

    def conditional_sample_rtc(self,
            global_cond: torch.Tensor,
            prev_naction_chunk: torch.Tensor,
            inference_delay: int,
            prefix_attention_horizon: int,
            prefix_attention_schedule: PrefixAttentionSchedule,
            max_guidance_weight: float,
            sigma: float = 1.0,
            generator=None,
            ) -> torch.Tensor:
        B = global_cond.shape[0]
        T = self.horizon
        Da = self.action_dim
        To = self.n_obs_steps
        start = To - 1
        future_len = T - start
        device = self.device
        dtype = self.dtype

        if prev_naction_chunk.shape != (B, future_len, Da):
            raise ValueError(
                f"prev_naction_chunk must have shape {(B, future_len, Da)}, "
                f"got {tuple(prev_naction_chunk.shape)}")
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}.")

        prev_naction_chunk = prev_naction_chunk.to(device=device, dtype=dtype)
        prefix_full = torch.zeros((B, T, Da), device=device, dtype=dtype)
        prefix_full[:, start:] = prev_naction_chunk
        weights_future = get_prefix_weights_torch(
            inference_delay,
            prefix_attention_horizon,
            future_len,
            prefix_attention_schedule,
            device=device,
            dtype=dtype,
        )
        weights_full = torch.zeros(T, device=device, dtype=dtype)
        weights_full[start:] = weights_future

        trajectory = torch.randn(
            size=(B, T, Da),
            dtype=dtype,
            device=device,
            generator=generator)
        dt = 1.0 / self.num_inference_steps
        time = torch.ones((), device=device, dtype=dtype)
        model = self.model

        def pinv_corrected_velocity(obs_emb, x_t, y, t):
            def denoiser(x):
                v = model(x, t, local_cond=None, global_cond=obs_emb)
                return x - v * t, v

            x0, vjp_fn, v = vjp(denoiser, x_t.detach(), has_aux=True)
            error = (y - x0) * weights_full[None, :, None]
            pinv_correction = vjp_fn(error)[0]
            sigma2 = sigma ** 2
            inv_r2 = (t**2 + sigma2 * (1 - t) ** 2) / (t**2 * sigma2)
            c = torch.nan_to_num(t / (1 - t), posinf=max_guidance_weight)
            guidance_weight = torch.minimum(
                c * inv_r2,
                torch.as_tensor(max_guidance_weight, device=device, dtype=dtype))
            return v.detach() - guidance_weight * pinv_correction.detach()

        with torch.enable_grad():
            for _ in range(self.num_inference_steps):
                model_output = pinv_corrected_velocity(
                    global_cond, trajectory, prefix_full, time)
                trajectory = trajectory - dt * model_output
                time = time - dt

        return trajectory

    def conditional_predict_rtc(self,
            obs_emb: torch.Tensor,
            prev_naction_chunk: torch.Tensor,
            inference_delay: int,
            prefix_attention_horizon: int,
            prefix_attention_schedule: PrefixAttentionSchedule,
            max_guidance_weight: float,
            sigma: float = 1.0,
            generator=None,
            ) -> Dict[str, torch.Tensor]:
        naction_pred = self.conditional_sample_rtc(
            global_cond=obs_emb,
            prev_naction_chunk=prev_naction_chunk,
            inference_delay=inference_delay,
            prefix_attention_horizon=prefix_attention_horizon,
            prefix_attention_schedule=prefix_attention_schedule,
            max_guidance_weight=max_guidance_weight,
            sigma=sigma,
            generator=generator,
        )
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        To = self.n_obs_steps
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]

        return {
            'action': action,
            'action_pred': action_pred,
            'naction': naction_pred[:,start:end],
            'naction_pred': naction_pred,
            'action_pred_all': action_pred[:,start:],
            'naction_pred_all': naction_pred[:,start:],
        }

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

        # Check if noise shape is correct
        assert noise.shape == (B, T, Da)
        trajectory = noise.to(device=device, dtype=dtype)

        timesteps = torch.linspace(
            1.0, 0.0, self.num_inference_steps + 1,
            device=trajectory.device,
            dtype=torch.float32)
        dt = -1.0 / self.num_inference_steps

        for t in timesteps[:-1]:
            # 1. predict model output
            model_output = model(trajectory, t,
                local_cond=None, global_cond=obs_emb)

            # 2. compute previous image: x_t -> x_t-1
            trajectory = trajectory + dt * model_output

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
            'action_pred_all': action_pred[:,start:],
            'naction_pred_all': naction_pred[:,start:],
        }
        return result

    def predict_action(self, obs_dict: Dict[str, torch.Tensor], rtc_context=None) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert self.obs_as_global_cond, "VIB policy only supports obs_as_global_cond=True"
        
        # 1. encode observation
        _, prev_action, prev_action_valid_mask = split_prev_action(obs_dict)
        prev_action_cond = get_prev_action_cond(prev_action, prev_action_valid_mask, self.n_prev_action_steps, self.normalizer['action'])
        obs_emb = self.encode_obs(obs_dict)
        
        # 2. pass through VIB module
        modified_obs_emb, _, _, _ = self.vib_forward(obs_emb, deterministic=True)
        model_global_cond = append_prev_action_cond(modified_obs_emb, prev_action_cond)
        
        # 3. predict action
        if rtc_context is None:
            result = self.conditional_predict(model_global_cond)
        else:
            result = self.conditional_predict_rtc(
                model_global_cond,
                prev_naction_chunk=rtc_context['prev_naction_chunk'],
                inference_delay=rtc_context['inference_delay'],
                prefix_attention_horizon=rtc_context['prefix_attention_horizon'],
                prefix_attention_schedule=rtc_context.get('prefix_attention_schedule', 'exp'),
                max_guidance_weight=rtc_context['max_guidance_weight'],
                sigma=rtc_context.get('sigma', 1.0),
            )
        
        # 4. append embeddings to result for debugging
        result['obs_emb'] = obs_emb
        result['modified_obs_emb'] = modified_obs_emb
        
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def forward(self, batch):
        return self.compute_loss(batch)

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        obs_dict, prev_action, prev_action_valid_mask = split_prev_action(batch['obs'])
        prev_action_cond = get_prev_action_cond(prev_action, prev_action_valid_mask, self.n_prev_action_steps, self.normalizer['action'])
        nobs = self.normalizer.normalize(obs_dict)
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
        model_global_cond = append_prev_action_cond(global_cond, prev_action_cond)

        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn_like(trajectory)
        # Sample a random continuous flow time. t=1 is noise, t=0 is sample.
        timesteps = torch.rand(
            (batch_size,), device=trajectory.device, dtype=trajectory.dtype)
        timesteps_b = timesteps[:, None, None]
        noisy_trajectory = (1 - timesteps_b) * trajectory + timesteps_b * noise

        # apply conditioning
        noisy_trajectory[condition_mask] = cond_data[condition_mask]
        
        # target for flow matching
        target = noise - trajectory

        # --- VIB Forward Pass ---
        # Decouple il and vib training
        modified_global_cond, z_mean, z_logvar, z = self.vib_forward(global_cond.detach(), deterministic=False)

        # --- IL Flow Loss (still conditioned on original obs_emb) ---
        pred_il = self.model(noisy_trajectory, timesteps, global_cond=model_global_cond)
        il_loss = F.mse_loss(pred_il, target)

        # --- VIB KL Loss ---
        # KL divergence loss to regularize the latent space
        vib_kl_loss = -0.5 * torch.mean(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())
        # --- VIB IL Loss ---
        frozen_model_params = {
            key: value.detach() for key, value in self.model.named_parameters()
        }
        frozen_model_buffers = {
            key: value.detach() for key, value in self.model.named_buffers()
        }
        pred_vib_il = functional_call(
            self.model,
            (frozen_model_params, frozen_model_buffers),
            (noisy_trajectory, timesteps),
            {'global_cond': append_prev_action_cond(modified_global_cond, prev_action_cond)}
        )
        vib_il_loss = F.mse_loss(pred_vib_il, target)
        # --- VIB reconstruction loss ---, disabled for now
        vib_recon_loss = F.mse_loss(modified_global_cond, global_cond.detach())
        
        vib_loss = (
            vib_il_loss
            + self.vib_beta * vib_kl_loss
            + self.vib_recon * vib_recon_loss
        )
        loss = il_loss + vib_loss

        with torch.no_grad():
            delta_rms = (modified_global_cond - global_cond).pow(2).mean().sqrt()
            base_rms = global_cond.pow(2).mean().sqrt()
            # logging
            info = {
                'il_loss': il_loss.item(),
                'vib_loss': vib_loss.item(),
                'vib_il_loss': vib_il_loss.item(),
                'vib_kl_loss': vib_kl_loss.item(),
                'cond_base_rms': base_rms.item(),
                'cond_delta_rms': delta_rms.item(),
                'z_mean_rms': z_mean.pow(2).mean().sqrt().item(),
                'z_std_rms': (z_logvar * 0.5).exp().pow(2).mean().sqrt().item(),
                'z_rms': z.pow(2).mean().sqrt().item(),
                'vib_recon_loss': vib_recon_loss.item(),
            }

        return loss, info
