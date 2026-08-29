import unittest

import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra

from zprl.model.vib import AEEncoder
from zprl.policy.flow_match_ae_unet_image_policy import (
    FlowMatchAeUnetImagePolicy,
)


class _ObsEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(2, 4)

    def output_shape(self):
        return (4,)

    def forward(self, obs):
        return self.proj(obs['state'])


def _make_policy():
    policy = FlowMatchAeUnetImagePolicy(
        shape_meta={'action': {'shape': [2]}},
        noise_scheduler=FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=10),
        obs_encoder=_ObsEncoder(),
        horizon=4,
        n_action_steps=2,
        n_obs_steps=1,
        num_inference_steps=2,
        diffusion_step_embed_dim=8,
        down_dims=(8, 16),
        kernel_size=3,
        n_groups=4,
        vib_latent_dim=2,
        vib_hidden_dim=8,
        vib_beta=123.0,
        vib_recon=0.3,
    )
    policy.normalizer.fit({
        'state': torch.randn(16, 2),
        'action': torch.randn(16, 4, 2),
    }, last_n_dims=1, mode='gaussian')
    return policy


class FlowMatchAeTest(unittest.TestCase):
    def test_ae_encoder_returns_zero_logvar_without_variance_parameters(self):
        encoder = AEEncoder(input_dim=4, latent_dim=2, hidden_dim=8)
        inputs = torch.randn(3, 4, dtype=torch.float64)
        encoder = encoder.to(dtype=inputs.dtype)

        z, z_logvar = encoder(inputs)

        self.assertEqual(z.shape, (3, 2))
        self.assertEqual(z_logvar.shape, z.shape)
        self.assertEqual(z_logvar.device, z.device)
        self.assertEqual(z_logvar.dtype, z.dtype)
        self.assertEqual(torch.count_nonzero(z_logvar).item(), 0)
        self.assertFalse(any(
            'logvar' in name for name, _ in encoder.named_parameters()))

    def test_policy_is_deterministic_and_loss_has_no_kl(self):
        torch.manual_seed(1)
        policy = _make_policy()
        global_cond = torch.randn(3, 4)

        deterministic = policy.vib_forward(global_cond, deterministic=True)
        stochastic = policy.vib_forward(global_cond, deterministic=False)

        for actual, expected in zip(stochastic, deterministic):
            torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(deterministic[1], deterministic[3])
        self.assertEqual(torch.count_nonzero(deterministic[2]).item(), 0)
        self.assertFalse(any(
            'logvar_head' in key for key in policy.state_dict()))

        batch = {
            'obs': {'state': torch.randn(3, 1, 2)},
            'action': torch.randn(3, 4, 2),
        }
        loss, info = policy.compute_loss(batch)

        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(set(info), {
            'il_loss',
            'vib_loss',
            'vib_il_loss',
            'cond_base_rms',
            'cond_delta_rms',
            'z_mean_rms',
            'z_rms',
            'vib_recon_loss',
        })
        expected_vib_loss = (
            info['vib_il_loss']
            + policy.vib_recon * info['vib_recon_loss']
        )
        self.assertAlmostEqual(
            info['vib_loss'], expected_vib_loss, places=5)

        loss.backward()
        for module in (policy.vib_encoder, policy.vib_decoder):
            self.assertTrue(all(
                param.grad is not None and torch.isfinite(param.grad).all()
                for param in module.parameters()
            ))

    def test_config_composes_for_robomimic_tasks(self):
        for task in ('can_image_abs', 'square_image_abs'):
            GlobalHydra.instance().clear()
            with initialize(version_base=None, config_path='../zprl/config'):
                cfg = compose(
                    config_name='train_flow_match_ae_unet_image_workspace',
                    overrides=[f'task={task}'])

            self.assertEqual(
                cfg._target_,
                'zprl.workspace.train_flow_match_vib_unet_image_workspace.'
                'TrainFlowMatchVibUnetImageWorkspace')
            self.assertEqual(
                cfg.policy._target_,
                'zprl.policy.flow_match_ae_unet_image_policy.'
                'FlowMatchAeUnetImagePolicy')
            self.assertEqual(cfg.policy.vib_latent_dim, 16)
            self.assertEqual(cfg.policy.vib_alpha, 1.0)
            self.assertEqual(cfg.policy.vib_beta, 0.0)
            self.assertEqual(cfg.policy.vib_recon, 0.01)
            self.assertEqual(cfg.policy.vib_hidden_dim, 256)


if __name__ == '__main__':
    unittest.main()
