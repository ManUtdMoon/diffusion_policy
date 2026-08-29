import torch
import torch.nn.functional as F
from torch.func import functional_call

from zprl.model.vib import AEEncoder
from zprl.policy.flow_match_vib_unet_image_policy import FlowMatchVibUnetImagePolicy


class FlowMatchAeUnetImagePolicy(FlowMatchVibUnetImagePolicy):
    def __init__(self,
            *args,
            vib_latent_dim=16,
            vib_alpha=1.0,
            vib_beta=0.0,
            vib_recon=0.1,
            vib_hidden_dim=256,
            **kwargs):
        super().__init__(
            *args,
            vib_latent_dim=vib_latent_dim,
            vib_alpha=vib_alpha,
            vib_beta=vib_beta,
            vib_recon=vib_recon,
            vib_hidden_dim=vib_hidden_dim,
            **kwargs)

        global_cond_dim = self.obs_feature_dim * self.n_obs_steps
        self.vib_encoder = AEEncoder(
            input_dim=global_cond_dim,
            latent_dim=vib_latent_dim,
            hidden_dim=vib_hidden_dim)

    def vib_forward(self, global_cond, deterministic=False):
        z, z_logvar = self.vib_encoder(global_cond)
        modified_global_cond = self.vib_decoder(z)
        return modified_global_cond, z, z_logvar, z

    def _compute_vib_loss(self, noisy_trajectory, timesteps, target, global_cond):
        # --- VIB Forward Pass ---
        # Decouple il and vib training
        modified_global_cond, z_mean, _, z = self.vib_forward(
            global_cond.detach(), deterministic=False)

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
            {'global_cond': modified_global_cond}
        )
        vib_il_loss = F.mse_loss(pred_vib_il, target)
        # --- VIB reconstruction loss ---
        vib_recon_loss = F.mse_loss(modified_global_cond, global_cond.detach())

        vib_loss = (
            vib_il_loss
            + self.vib_recon * vib_recon_loss
        )

        with torch.no_grad():
            delta_rms = (modified_global_cond - global_cond).pow(2).mean().sqrt()
            base_rms = global_cond.pow(2).mean().sqrt()
            # logging
            info = {
                'vib_loss': vib_loss.item(),
                'vib_il_loss': vib_il_loss.item(),
                'cond_base_rms': base_rms.item(),
                'cond_delta_rms': delta_rms.item(),
                'z_mean_rms': z_mean.pow(2).mean().sqrt().item(),
                'z_rms': z.pow(2).mean().sqrt().item(),
                'vib_recon_loss': vib_recon_loss.item(),
            }

        return vib_loss, info
