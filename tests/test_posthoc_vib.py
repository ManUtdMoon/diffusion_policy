import copy
import pathlib
import tempfile
import unittest

import dill
import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from torch.func import functional_call

from zprl.common.pytorch_util import dict_apply
from zprl.model.diffusion.ema_model import EMAModel
from zprl.policy.flow_match_vib_unet_image_policy import (
    FlowMatchVibUnetImagePolicy,
)
from zprl.workspace.train_flow_match_posthoc_vib_unet_image_workspace import (
    TrainFlowMatchPosthocVibUnetImageWorkspace,
)
from zprl.workspace.train_flow_match_vib_unet_image_workspace import (
    TrainFlowMatchVibUnetImageWorkspace,
)


CONFIG_DIR = (pathlib.Path(__file__).parent.parent / "zprl" / "config").resolve()


class _ObsEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(2, 4)

    def output_shape(self):
        return (4,)

    def forward(self, obs):
        return self.proj(obs['state'])


def _make_policy():
    policy = FlowMatchVibUnetImagePolicy(
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
        vib_beta=0.2,
        vib_recon=0.3,
    )
    policy.normalizer.fit({
        'state': torch.randn(16, 2),
        'action': torch.randn(16, 4, 2),
    }, last_n_dims=1, mode='gaussian')
    return policy


def _make_workspace_cfg(base_ckpt):
    shape_meta = {'action': {'shape': [2]}}
    policy = {
        '_target_': (
            'zprl.policy.flow_match_vib_unet_image_policy.'
            'FlowMatchVibUnetImagePolicy'),
        'shape_meta': shape_meta,
        'noise_scheduler': {
            '_target_': (
                'diffusers.schedulers.scheduling_flow_match_euler_discrete.'
                'FlowMatchEulerDiscreteScheduler'),
            'num_train_timesteps': 10,
        },
        'obs_encoder': {
            '_target_': f'{__name__}._ObsEncoder',
        },
        'horizon': 4,
        'n_action_steps': 2,
        'n_obs_steps': 1,
        'num_inference_steps': 2,
        'obs_as_global_cond': True,
        'diffusion_step_embed_dim': 8,
        'down_dims': [8, 16],
        'kernel_size': 3,
        'n_groups': 4,
        'cond_predict_scale': True,
        'vib_latent_dim': 2,
        'vib_hidden_dim': 8,
        'vib_beta': 0.2,
        'vib_recon': 0.3,
    }
    return OmegaConf.create({
        'name': 'train_flow_match_posthoc_vib_unet_image',
        '_target_': (
            'zprl.workspace.train_flow_match_posthoc_vib_unet_image_workspace.'
            'TrainFlowMatchPosthocVibUnetImageWorkspace'),
        'base_ckpt': str(base_ckpt),
        'task_name': 'unit_task',
        'shape_meta': shape_meta,
        'horizon': 4,
        'n_obs_steps': 1,
        'n_action_steps': 2,
        'n_latency_steps': 0,
        'dataset_obs_steps': 1,
        'policy': policy,
        'task': {
            'task_name': 'unit',
            'dataset_type': 'unit_type',
            'abs_action': True,
            'dataset': {
                '_target_': 'unit.Dataset',
                'shape_meta': shape_meta,
                'dataset_path': '/tmp/unit.hdf5',
                'horizon': 4,
                'pad_before': 0,
                'pad_after': 1,
                'n_obs_steps': 1,
                'abs_action': True,
                'rotation_rep': 'rotation_6d',
                'use_legacy_normalizer': False,
                'seed': 3,
                'num_demo': 4,
            },
        },
        'optimizer': {
            '_target_': 'torch.optim.AdamW',
            'lr': 1e-3,
        },
        'training': {
            'device': 'cpu',
            'seed': 3,
            'resume': False,
            'use_ema': True,
        },
    })


def _make_source_payload(target_cfg):
    source_cfg = copy.deepcopy(target_cfg)
    source_cfg._target_ = (
        'zprl.workspace.train_diffusion_unet_image_workspace.'
        'TrainDiffusionUnetImageWorkspace')
    source_cfg.policy._target_ = (
        'zprl.policy.flow_match_unet_image_policy.FlowMatchUnetImagePolicy')
    for key in (
            'vib_latent_dim', 'vib_hidden_dim', 'vib_beta', 'vib_recon'):
        del source_cfg.policy[key]

    source_policy = hydra.utils.instantiate(source_cfg.policy)
    source_policy.normalizer.fit({
        'state': torch.randn(16, 2),
        'action': torch.randn(16, 4, 2),
    }, last_n_dims=1, mode='gaussian')
    with torch.no_grad():
        for param in source_policy.parameters():
            param.add_(0.25)
    source_state = copy.deepcopy(source_policy.state_dict())
    return {
        'cfg': source_cfg,
        'state_dicts': {
            'model': copy.deepcopy(source_state),
            'ema_model': source_state,
        },
        'pickles': {},
    }


def _make_batch():
    return {
        'obs': {'state': torch.randn(3, 1, 2)},
        'action': torch.randn(3, 4, 2),
    }


def _legacy_compute_loss(policy, batch):
    nobs = policy.normalizer.normalize(batch['obs'])
    trajectory = policy.normalizer['action'].normalize(batch['action'])
    batch_size = trajectory.shape[0]
    this_nobs = dict_apply(
        nobs,
        lambda x: x[:, :policy.n_obs_steps, ...].reshape(
            -1, *x.shape[2:]))
    global_cond = policy.obs_encoder(this_nobs).reshape(batch_size, -1)

    condition_mask = policy.mask_generator(trajectory.shape)
    noise = torch.randn(trajectory.shape, device=trajectory.device)
    if policy.timesteps.device != trajectory.device:
        policy.timesteps = policy.timesteps.to(trajectory.device)
    timestep_idxs = torch.randint(
        0, policy.noise_scheduler.config.num_train_timesteps,
        (batch_size,), device=trajectory.device).long()
    timesteps = policy.timesteps[timestep_idxs]
    noisy_trajectory = policy.noise_scheduler.scale_noise(
        trajectory, timesteps, noise)
    noisy_trajectory[condition_mask] = trajectory[condition_mask]
    target = noise - trajectory

    modified_global_cond, z_mean, z_logvar, z = policy.vib_forward(
        global_cond.detach(), deterministic=False)
    pred_il = policy.model(
        noisy_trajectory, timesteps, global_cond=global_cond)
    il_loss = F.mse_loss(pred_il, target)
    vib_kl_loss = -0.5 * torch.mean(
        1 + z_logvar - z_mean.pow(2) - z_logvar.exp())
    frozen_model_params = {
        key: value.detach() for key, value in policy.model.named_parameters()
    }
    frozen_model_buffers = {
        key: value.detach() for key, value in policy.model.named_buffers()
    }
    pred_vib_il = functional_call(
        policy.model,
        (frozen_model_params, frozen_model_buffers),
        (noisy_trajectory, timesteps),
        {'global_cond': modified_global_cond})
    vib_il_loss = F.mse_loss(pred_vib_il, target)
    vib_recon_loss = F.mse_loss(
        modified_global_cond, global_cond.detach())
    vib_loss = (
        vib_il_loss
        + policy.vib_beta * vib_kl_loss
        + policy.vib_recon * vib_recon_loss
    )

    with torch.no_grad():
        info = {
            'il_loss': il_loss.item(),
            'vib_loss': vib_loss.item(),
            'vib_il_loss': vib_il_loss.item(),
            'vib_kl_loss': vib_kl_loss.item(),
            'cond_base_rms': global_cond.pow(2).mean().sqrt().item(),
            'cond_delta_rms': (
                modified_global_cond - global_cond).pow(2).mean().sqrt().item(),
            'z_mean_rms': z_mean.pow(2).mean().sqrt().item(),
            'z_std_rms': (
                z_logvar * 0.5).exp().pow(2).mean().sqrt().item(),
            'z_rms': z.pow(2).mean().sqrt().item(),
            'vib_recon_loss': vib_recon_loss.item(),
        }
    return il_loss + vib_loss, info


class _LossModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))

    def forward(self, batch):
        return self.weight * batch, {'metric': 1.0}

    def compute_vib_loss(self, batch):
        return self.forward(batch)


class PosthocVibTest(unittest.TestCase):
    def test_joint_loss_refactor_preserves_values(self):
        torch.manual_seed(1)
        policy = _make_policy()
        batch = _make_batch()

        torch.manual_seed(7)
        expected_loss, expected_info = _legacy_compute_loss(policy, batch)
        torch.manual_seed(7)
        actual_loss, actual_info = policy.compute_loss(batch)

        torch.testing.assert_close(actual_loss, expected_loss)
        self.assertEqual(set(actual_info), set(expected_info))
        for key in expected_info:
            self.assertAlmostEqual(actual_info[key], expected_info[key], places=6)

    def test_compute_vib_loss_formula_and_gradient_boundary(self):
        torch.manual_seed(2)
        policy = _make_policy()
        batch = _make_batch()
        policy.requires_grad_(False)
        policy.vib_encoder.requires_grad_(True)
        policy.vib_decoder.requires_grad_(True)

        loss, info = policy.compute_vib_loss(batch)
        expected = (
            info['vib_il_loss']
            + policy.vib_beta * info['vib_kl_loss']
            + policy.vib_recon * info['vib_recon_loss'])
        self.assertNotIn('il_loss', info)
        self.assertAlmostEqual(loss.item(), expected, places=6)

        loss.backward()
        for module in (policy.vib_encoder, policy.vib_decoder):
            self.assertTrue(all(
                param.grad is not None and torch.isfinite(param.grad).all()
                for param in module.parameters()))
        for module in (policy.obs_encoder, policy.model):
            self.assertTrue(all(param.grad is None for param in module.parameters()))

    def test_compute_vib_loss_restores_scheduler_after_inference(self):
        policy = _make_policy().eval()
        policy.noise_scheduler.set_timesteps(policy.num_inference_steps)
        self.assertEqual(
            len(policy.noise_scheduler.timesteps), policy.num_inference_steps)

        torch.manual_seed(0)
        loss, _ = policy.compute_vib_loss(_make_batch())

        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(
            len(policy.noise_scheduler.timesteps),
            policy.noise_scheduler.config.num_train_timesteps)
        torch.testing.assert_close(
            policy.timesteps, policy.noise_scheduler.timesteps)

    def test_workspace_default_hooks_preserve_joint_behavior(self):
        workspace = TrainFlowMatchVibUnetImageWorkspace.__new__(
            TrainFlowMatchVibUnetImageWorkspace)
        workspace.model = _LossModel()

        self.assertEqual(
            {id(param) for param in workspace.get_optimizer_parameters()},
            {id(param) for param in workspace.model.parameters()})
        loss, info = workspace.compute_batch_loss(torch.tensor(3.0))
        self.assertEqual(loss.item(), 3.0)
        self.assertEqual(info, {'metric': 1.0})

        eval_model = _LossModel()
        eval_model.weight.data.fill_(2.0)
        loss, info = workspace.compute_batch_loss(
            torch.tensor(3.0), model=eval_model)
        self.assertEqual(loss.item(), 6.0)
        self.assertEqual(info, {'metric': 1.0})

        posthoc_workspace = TrainFlowMatchPosthocVibUnetImageWorkspace.__new__(
            TrainFlowMatchPosthocVibUnetImageWorkspace)
        posthoc_workspace.model = workspace.model
        loss, info = posthoc_workspace.compute_batch_loss(
            torch.tensor(3.0), model=eval_model)
        self.assertEqual(loss.item(), 6.0)
        self.assertEqual(info, {'metric': 1.0})


    def test_posthoc_config_composes_with_full_policy_and_task(self):
        with initialize_config_dir(
                version_base=None, config_dir=str(CONFIG_DIR)):
            cfg = compose(
                config_name=(
                    'train_flow_match_posthoc_vib_unet_image_workspace'),
                overrides=[
                    'task=square_image_abs',
                    'base_ckpt=/tmp/base.ckpt',
                ])

        self.assertEqual(
            cfg._target_,
            'zprl.workspace.train_flow_match_posthoc_vib_unet_image_workspace.'
            'TrainFlowMatchPosthocVibUnetImageWorkspace')
        self.assertEqual(
            cfg.policy._target_,
            'zprl.policy.flow_match_vib_unet_image_policy.'
            'FlowMatchVibUnetImagePolicy')
        self.assertEqual(cfg.task.name, 'square_image')
        self.assertFalse(cfg.training.resume)
        self.assertTrue(cfg.training.use_ema)
        self.assertEqual(
            cfg.ema._target_, 'zprl.model.diffusion.ema_model.EMAModel')

    def test_source_initialization_loads_base_freezes_and_syncs_full_ema(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = pathlib.Path(temp_dir) / 'source.ckpt'
            cfg = _make_workspace_cfg(source_path)
            payload = _make_source_payload(cfg)
            torch.save(payload, source_path, pickle_module=dill)
            workspace = TrainFlowMatchPosthocVibUnetImageWorkspace(
                cfg, output_dir=temp_dir)
            vib_before = copy.deepcopy({
                key: value for key, value in workspace.model.state_dict().items()
                if key.startswith('vib_encoder.')
                or key.startswith('vib_decoder.')
            })

            workspace._initialize_from_base_checkpoint()

            source_state = payload['state_dicts']['ema_model']
            target_state = workspace.model.state_dict()
            for key, value in source_state.items():
                torch.testing.assert_close(target_state[key], value)
            for key, value in vib_before.items():
                torch.testing.assert_close(target_state[key], value)
            self.assertTrue(all(
                param.requires_grad
                for param in workspace.model.vib_encoder.parameters()))
            self.assertTrue(all(
                param.requires_grad
                for param in workspace.model.vib_decoder.parameters()))
            self.assertTrue(all(
                not param.requires_grad
                for name, param in workspace.model.named_parameters()
                if not name.startswith('vib_encoder.')
                and not name.startswith('vib_decoder.')))
            optimizer_ids = {
                id(param)
                for group in workspace.optimizer.param_groups
                for param in group['params']
            }
            trainable_ids = {
                id(param) for param in workspace.model.parameters()
                if param.requires_grad
            }
            self.assertEqual(optimizer_ids, trainable_ids)
            self.assertEqual(
                workspace.ema_model.state_dict().keys(),
                workspace.model.state_dict().keys())
            for key, value in workspace.model.state_dict().items():
                torch.testing.assert_close(
                    workspace.ema_model.state_dict()[key], value)

    def test_source_config_and_state_mismatches_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cfg = _make_workspace_cfg(
                pathlib.Path(temp_dir) / 'source.ckpt')
            workspace = TrainFlowMatchPosthocVibUnetImageWorkspace(
                cfg, output_dir=temp_dir)
            payload = _make_source_payload(cfg)

            cases = (
                ('task_name', 'other_task', 'task_name'),
                ('shape_meta.action.shape', [3], 'shape_meta'),
                ('policy.down_dims', [8, 32], 'policy.down_dims'),
                ('task.dataset.use_legacy_normalizer', True,
                 'task.dataset.use_legacy_normalizer'),
            )
            for path, value, message in cases:
                with self.subTest(path=path):
                    source_cfg = copy.deepcopy(payload['cfg'])
                    OmegaConf.update(source_cfg, path, value, merge=False)
                    with self.assertRaisesRegex(ValueError, message):
                        workspace._validate_source_config(source_cfg)

            source_cfg = copy.deepcopy(payload['cfg'])
            source_cfg.policy._target_ = 'unsupported.Policy'
            with self.assertRaisesRegex(
                    ValueError, 'Unsupported source policy target'):
                workspace._validate_source_config(source_cfg)

            invalid_payload = copy.deepcopy(payload)
            missing_key = next(
                key for key in invalid_payload['state_dicts']['ema_model']
                if key.startswith('model.'))
            del invalid_payload['state_dicts']['ema_model'][missing_key]
            with self.assertRaisesRegex(
                    RuntimeError, 'Unexpected source state dict keys'):
                workspace._initialize_from_source_payload(
                    invalid_payload, 'invalid.ckpt')

            invalid_payload = copy.deepcopy(payload)
            invalid_payload['state_dicts']['ema_model'][
                'unexpected.weight'] = torch.ones(1)
            with self.assertRaisesRegex(
                    RuntimeError, 'Unexpected source state dict keys'):
                workspace._initialize_from_source_payload(
                    invalid_payload, 'invalid.ckpt')

    def test_source_without_ema_selects_model_state(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cfg = _make_workspace_cfg(
                pathlib.Path(temp_dir) / 'source.ckpt')
            workspace = TrainFlowMatchPosthocVibUnetImageWorkspace(
                cfg, output_dir=temp_dir)
            payload = _make_source_payload(cfg)
            payload['cfg'].training.use_ema = False
            key = next(
                key for key in payload['state_dicts']['model']
                if key.startswith('model.')
                and payload['state_dicts']['model'][key].is_floating_point())
            payload['state_dicts']['model'][key].fill_(3)
            payload['state_dicts']['ema_model'][key].fill_(4)

            workspace._initialize_from_source_payload(
                payload, 'source.ckpt')

            self.assertEqual(workspace.source_state_key, 'model')
            torch.testing.assert_close(
                workspace.model.state_dict()[key],
                torch.full_like(workspace.model.state_dict()[key], 3))

    def test_posthoc_optimizer_step_changes_only_vib_and_full_ema_tracks_base(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cfg = _make_workspace_cfg(
                pathlib.Path(temp_dir) / 'source.ckpt')
            workspace = TrainFlowMatchPosthocVibUnetImageWorkspace(
                cfg, output_dir=temp_dir)
            workspace._initialize_from_source_payload(
                _make_source_payload(cfg), 'source.ckpt')
            base_before = copy.deepcopy({
                key: value for key, value in workspace.model.state_dict().items()
                if not key.startswith('vib_encoder.')
                and not key.startswith('vib_decoder.')
            })
            vib_before = copy.deepcopy({
                key: value for key, value in workspace.model.state_dict().items()
                if key.startswith('vib_encoder.')
                or key.startswith('vib_decoder.')
            })

            loss, info = workspace.compute_batch_loss(_make_batch())
            loss.backward()
            workspace.optimizer.step()
            workspace.optimizer.zero_grad()

            current_state = workspace.model.state_dict()
            self.assertTrue(any(
                not torch.equal(current_state[key], value)
                for key, value in vib_before.items()))
            for key, value in base_before.items():
                self.assertTrue(torch.equal(current_state[key], value), key)
            ema = EMAModel(workspace.ema_model)
            ema.step(workspace.model)
            for key, value in workspace.model.state_dict().items():
                if not key.startswith('vib_encoder.') \
                        and not key.startswith('vib_decoder.'):
                    self.assertTrue(torch.equal(
                        workspace.ema_model.state_dict()[key], value), key)

    def test_checkpoint_round_trip_is_independent_of_source_and_online_compatible(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = pathlib.Path(temp_dir)
            source_path = temp_path / 'source.ckpt'
            cfg = _make_workspace_cfg(source_path)
            source_payload = _make_source_payload(cfg)
            torch.save(source_payload, source_path, pickle_module=dill)
            workspace = TrainFlowMatchPosthocVibUnetImageWorkspace(
                cfg, output_dir=temp_dir)
            workspace._initialize_from_base_checkpoint()
            checkpoint_path = temp_path / 'posthoc.ckpt'
            workspace.save_checkpoint(
                path=checkpoint_path, use_thread=False)
            source_path.unlink()

            payload = torch.load(
                checkpoint_path, pickle_module=dill, weights_only=False)
            self.assertIn('model', payload['state_dicts'])
            self.assertIn('ema_model', payload['state_dicts'])
            self.assertEqual(
                payload['cfg'].policy._target_,
                'zprl.policy.flow_match_vib_unet_image_policy.'
                'FlowMatchVibUnetImagePolicy')
            self.assertEqual(
                set(payload['state_dicts']['model']),
                set(payload['state_dicts']['ema_model']))

            restored = TrainFlowMatchPosthocVibUnetImageWorkspace(
                payload['cfg'], output_dir=temp_dir)
            restored.load_payload(payload)
            for key, value in payload['state_dicts']['ema_model'].items():
                torch.testing.assert_close(
                    restored.ema_model.state_dict()[key], value)

            online_policy = hydra.utils.instantiate(payload['cfg'].policy)
            online_policy.load_state_dict(payload['state_dicts']['ema_model'])


if __name__ == '__main__':
    unittest.main()
