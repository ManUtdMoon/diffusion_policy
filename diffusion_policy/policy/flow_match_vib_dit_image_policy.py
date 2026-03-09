from typing import Dict
import torch
import torch.nn.functional as F
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.dit import DiTNoiseNet
from diffusion_policy.model.vib import VIBDecoder, VIBEncoder
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from diffusion_policy.common.pytorch_util import dict_apply

class FlowMatchVibDitImagePolicy(BaseImagePolicy):
    def __init__(self,
            shape_meta: dict,
            noise_scheduler: FlowMatchEulerDiscreteScheduler,
            obs_encoder: MultiImageObsEncoder,
            horizon,
            n_action_steps,
            n_obs_steps,
            num_inference_steps=None,
            # model parameters
            num_blocks=6,
            hidden_dim=384,
            time_dim=128,
            dropout=0.1,
            dim_feedforward=2048,
            nhead=8,
            activation="gelu",
            # VIB parameters
            vib_latent_dim=16,
            vib_alpha=2.0,
            vib_beta=1e-3,
            vib_recon=0.1,
            vib_hidden_dim=256,
            # parameters passed to step
            **kwargs):
        super().__init__()

        # parse shapes
        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        # get feature dim
        obs_feature_dim = obs_encoder.output_shape()[0]

        # create diffusion model
        input_dim = action_dim
        cond_dim = obs_feature_dim * n_obs_steps

        model = DiTNoiseNet(
            input_dim=input_dim,
            input_length=horizon,
            cond_dim=cond_dim,
            time_dim=time_dim,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            dropout=dropout,
            dim_feedforward=dim_feedforward,
            nhead=nhead,
            activation=activation
        )

        self.obs_encoder = obs_encoder
        self.model = model

        self.vib_encoder = VIBEncoder(
            input_dim=cond_dim,
            latent_dim=vib_latent_dim,
            hidden_dim=vib_hidden_dim,
            alpha=vib_alpha,
        )
        self.vib_decoder = VIBDecoder(
            latent_dim=vib_latent_dim,
            output_dim=cond_dim,
            hidden_dim=vib_hidden_dim,
        )

        self.noise_scheduler = noise_scheduler
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.kwargs = kwargs

        self.vib_alpha = vib_alpha
        self.vib_beta = vib_beta
        self.vib_recon = vib_recon
        self.vib_latent_dim = vib_latent_dim

        assert num_inference_steps is not None, (
            "num_inference_steps must be specified for FlowMatchVibDitImagePolicy"
        )
        self.num_inference_steps = num_inference_steps

        self.timesteps = self.noise_scheduler.timesteps.clone()

    # ========= inference  ============
    def conditional_sample(self,
            condition_data,
            global_cond=None,
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
            generator=generator,
        )

        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            model_output = model(trajectory, t, global_cond)
            trajectory = scheduler.step(
                model_output, t, trajectory, generator=generator, **kwargs
            ).prev_sample

        return trajectory

    def encode_obs(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        nobs = self.normalizer.normalize(obs_dict)
        B, To = next(iter(nobs.values())).shape[:2]
        batched_nobs = dict_apply(
            nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:])
        )
        obs_emb = self.obs_encoder(batched_nobs)
        obs_emb = obs_emb.reshape(B, -1)
        return obs_emb

    def vib_forward(self, global_cond, deterministic=False):
        z_mean, z_logvar = self.vib_encoder(global_cond)

        if deterministic:
            z = z_mean
        else:
            std = torch.exp(0.5 * z_logvar)
            eps = torch.randn_like(std)
            z = z_mean + eps * std

        modified_global_cond = self.vib_decoder(z)
        return modified_global_cond, z_mean, z_logvar, z

    def conditional_predict(self, obs_emb: torch.Tensor) -> Dict[str, torch.Tensor]:
        B = obs_emb.shape[0]
        T = self.horizon
        Da = self.action_dim
        To = self.n_obs_steps

        device = self.device
        dtype = self.dtype

        global_cond = obs_emb
        cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)

        nsample = self.conditional_sample(
            cond_data,
            global_cond=global_cond,
            **self.kwargs,
        )

        naction_pred = nsample
        action_pred = self.normalizer["action"].unnormalize(naction_pred)

        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]

        result = {
            "action": action,
            "action_pred": action_pred,
            "naction": naction_pred[:, start:end],
            "naction_pred": naction_pred,
        }
        return result

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        obs_emb = self.encode_obs(obs_dict)
        modified_obs_emb, _, _, _ = self.vib_forward(obs_emb, deterministic=True)

        result = self.conditional_predict(modified_obs_emb)
        result["obs_emb"] = obs_emb
        result["modified_obs_emb"] = modified_obs_emb
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        nobs = self.normalizer.normalize(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])
        batch_size = nactions.shape[0]

        trajectory = nactions

        this_nobs = dict_apply(
            nobs, lambda x: x[:, : self.n_obs_steps, ...].reshape(-1, *x.shape[2:])
        )
        nobs_features = self.obs_encoder(this_nobs)
        global_cond = nobs_features.reshape(batch_size, -1)

        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]

        if self.timesteps.device != trajectory.device:
            self.timesteps = self.timesteps.to(trajectory.device)
        timestep_idxs = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bsz,),
            device=trajectory.device,
        ).long()
        timesteps = self.timesteps[timestep_idxs]

        noisy_trajectory = self.noise_scheduler.scale_noise(trajectory, timesteps, noise)
        target = noise - trajectory

        # Decouple IL and VIB training; VIB should not update obs encoder.
        modified_global_cond, z_mean, z_logvar, z = self.vib_forward(
            global_cond.detach(), deterministic=False
        )

        pred_il = self.model(noisy_trajectory, timesteps, global_cond)
        il_loss = F.mse_loss(pred_il, target)

        # vib loss
        vib_kl_loss = -0.5 * torch.mean(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())

        pred_vib_il = self.model(noisy_trajectory, timesteps, modified_global_cond)
        vib_il_loss = F.mse_loss(pred_vib_il, target)

        loss = il_loss
        vib_loss = vib_il_loss + self.vib_beta * vib_kl_loss

        with torch.no_grad():
            delta_rms = (modified_global_cond - global_cond).pow(2).mean().sqrt()
            base_rms = global_cond.pow(2).mean().sqrt()
            info = {
                "vib_loss": vib_loss.item(),
                "vib_il_loss": vib_il_loss.item(),
                "vib_kl_loss": vib_kl_loss.item(),
                "cond_base_rms": base_rms.item(),
                "cond_delta_rms": delta_rms.item(),
                "z_mean_rms": z_mean.pow(2).mean().sqrt().item(),
                "z_std_rms": (z_logvar * 0.5).exp().pow(2).mean().sqrt().item(),
                "z_rms": z.pow(2).mean().sqrt().item(),
            }

        return loss, vib_loss, info
