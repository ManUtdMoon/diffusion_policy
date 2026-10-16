import importlib.util
import json
import os
import pathlib
import tempfile
import unittest
from unittest import mock

import dill
from omegaconf import OmegaConf
import torch

from zprl.common.training_budget import configure_training_budget
from zprl.dataset.base_dataset import BaseImageDataset
from zprl.env_runner.base_image_runner import BaseImageRunner
from zprl.model.common.normalizer import LinearNormalizer
from zprl.model.diffusion.ema_model import EMAModel


class TinyDataset(BaseImageDataset):
    def __len__(self):
        return 4

    def __getitem__(self, idx):
        return {'obs': {'state': torch.ones(1, 1)}, 'action': torch.ones(1, 1)}

    def get_validation_dataset(self):
        return self

    def get_normalizer(self):
        normalizer = LinearNormalizer()
        normalizer.fit({'action': torch.ones(4, 1), 'state': torch.ones(4, 1)})
        return normalizer


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.normalizer = LinearNormalizer()
        self.train_calls = 0

    def set_normalizer(self, normalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def forward(self, batch):
        return self.compute_loss(batch)

    def compute_loss(self, batch):
        self.train_calls += 1
        return (self.weight - batch['action']).square().mean()

    def predict_action(self, obs):
        return {'action_pred': self.weight.expand(len(obs['state']), 1, 1)}


class RecordingEMA(EMAModel):
    steps = []

    def step(self, model):
        RecordingEMA.steps.append(self.optimization_step)
        super().step(model)


class TinyRunner(BaseImageRunner):
    calls = 0

    def run(self, policy):
        TinyRunner.calls += 1
        return {'test/n_rollouts': 1}


def tiny_config(gradient_clip_val=None, use_ema=False):
    return OmegaConf.create({
        'policy': {'_target_': __name__ + '.TinyPolicy'},
        'task': {'dataset': {'_target_': __name__ + '.TinyDataset'},
            'action_mse_groups': {}, 'env_runner': {'_target_': __name__ + '.TinyRunner'}},
        'dataloader': {'batch_size': 2, 'num_workers': 0, 'shuffle': False},
        'val_dataloader': {'batch_size': 2, 'num_workers': 0, 'shuffle': False},
        'optimizer': {'_target_': 'torch.optim.SGD', 'lr': 0.1},
        'ema': {'_target_': __name__ + '.RecordingEMA'},
        'training': {'seed': 0, 'device': 'cpu', 'debug': False,
            'resume': False, 'num_epochs': None, 'max_train_steps': 5,
            'max_val_steps': 1, 'gradient_accumulate_every': 1,
            'gradient_clip_val': gradient_clip_val,
            'lr_scheduler': 'constant', 'lr_warmup_steps': 0,
            'lr_scheduler_kwargs': {},
            'use_ema': use_ema, 'freeze_encoder': False, 'enable_rollout': True,
            'mixed_precision': 'no', 'ddp_find_unused_parameters': False,
            'val_every': 1, 'sample_every': 2, 'checkpoint_every': 1,
            'rollout_every': 2, 'tqdm_interval_sec': 100},
        'logging': {},
        'checkpoint': {'topk': {'monitor_key': 'val_action_mse', 'mode': 'min',
            'k': 3, 'format_str': 'epoch_{epoch}.ckpt'},
            'save_last_ckpt': True, 'save_last_snapshot': False},
    })


class OneLegTrainingBudgetTest(unittest.TestCase):
    def test_budget_derives_epochs_and_preserves_epoch_fallback(self):
        cfg = OmegaConf.create({'max_train_steps': 200000, 'num_epochs': None})
        self.assertEqual(configure_training_budget(cfg, 1796), 200000)
        self.assertEqual(cfg.num_epochs, 112)
        cfg.max_train_steps = None
        cfg.num_epochs = 3
        self.assertEqual(configure_training_budget(cfg, 2), 6)
        with self.assertRaises(ValueError):
            configure_training_budget(cfg, 0)

    def _check_workspace(self, cls, gradient_clip_val=None, use_ema=False):
        TinyRunner.calls = 0
        RecordingEMA.steps = []
        with tempfile.TemporaryDirectory() as directory:
            cfg = tiny_config(gradient_clip_val, use_ema)
            module = cls.__module__
            with mock.patch(module + '.wandb.init'), mock.patch(module + '.wandb.config'):
                workspace = cls(cfg, output_dir=directory)
                workspace.run()
                if workspace._saving_thread is not None:
                    workspace._saving_thread.join()
                self.assertEqual(workspace.global_step, 5)
                self.assertEqual(workspace.epoch, 3)
                self.assertEqual(workspace.model.train_calls, 5)
                self.assertEqual(TinyRunner.calls, 1)
                self.assertTrue(workspace.model.training)
                self.assertEqual(len(list((pathlib.Path(directory) / 'checkpoints').glob('epoch_*.ckpt'))), 3)
                logs = [json.loads(line) for line in (pathlib.Path(directory) / 'logs.json.txt').read_text().splitlines()]
                self.assertEqual([row['epoch'] for row in logs if 'test/n_rollouts' in row], [2])
                self.assertEqual([row['epoch'] for row in logs if 'train_action_mse' in row], [0, 2])
                torch.testing.assert_close(workspace.model.weight.detach(), torch.tensor([0.125 if gradient_clip_val else 0.67232]))
                if gradient_clip_val:
                    self.assertEqual(logs[0]['grad_norm'], 2.0)
                checkpoint = pathlib.Path(directory) / 'checkpoints/latest.ckpt'
                payload = torch.load(checkpoint, pickle_module=dill, map_location='cpu')
                self.assertEqual(dill.loads(payload['pickles']['global_step']), 5)
                self.assertEqual(dill.loads(payload['pickles']['epoch']), 2)
                self.assertEqual(payload['pickles'].keys(), {'global_step', 'epoch', '_output_dir'})
                self.assertEqual(logs[-1]['epoch'], 2)
                self.assertEqual([row['global_step'] for row in logs], list(range(1, 6)))

                cfg.training.resume = True
                cfg.training.max_train_steps = 7
                resumed = cls(cfg, output_dir=directory)
                resumed.run()
                if resumed._saving_thread is not None:
                    resumed._saving_thread.join()
                self.assertEqual(resumed.global_step, 7)
                self.assertEqual(resumed.model.train_calls, 2)
                self.assertEqual(TinyRunner.calls, 2)
                self.assertEqual(len(list(checkpoint.parent.glob('epoch_*.ckpt'))), 4)
                completed = cls(cfg, output_dir=directory)
                completed.run()
                self.assertEqual(completed.global_step, 7)
                self.assertEqual(completed.model.train_calls, 0)
                if use_ema:
                    self.assertEqual(RecordingEMA.steps, list(range(7)))
                self.assertEqual(TinyRunner.calls, 2)

    @unittest.skipUnless(importlib.util.find_spec('accelerate'), 'requires accelerate')
    def test_accelerate_stops_mid_epoch_and_resumes_without_resetting_budget(self):
        from zprl.workspace.train_diffusion_image_accelerate_workspace import TrainDiffusionImageAccelerateWorkspace
        with mock.patch.dict(os.environ, {'ACCELERATE_USE_CPU': 'true'}):
            self._check_workspace(TrainDiffusionImageAccelerateWorkspace)


    @unittest.skipUnless(importlib.util.find_spec('accelerate'), 'requires accelerate')
    def test_gradient_clipping_limits_actual_optimizer_updates(self):
        from zprl.workspace.train_diffusion_image_accelerate_workspace import TrainDiffusionImageAccelerateWorkspace
        with mock.patch.dict(os.environ, {'ACCELERATE_USE_CPU': 'true'}):
            self._check_workspace(TrainDiffusionImageAccelerateWorkspace, gradient_clip_val=0.25)


    @unittest.skipUnless(importlib.util.find_spec('accelerate'), 'requires accelerate')
    def test_resume_preserves_ema_update_count(self):
        from zprl.workspace.train_diffusion_image_accelerate_workspace import TrainDiffusionImageAccelerateWorkspace
        with mock.patch.dict(os.environ, {'ACCELERATE_USE_CPU': 'true'}):
            self._check_workspace(TrainDiffusionImageAccelerateWorkspace, use_ema=True)


    @unittest.skipUnless(importlib.util.find_spec('accelerate'), 'requires accelerate')
    def test_periodic_events_share_zero_based_epoch_and_resume(self):
        from zprl.workspace.train_diffusion_image_accelerate_workspace import TrainDiffusionImageAccelerateWorkspace
        cls = TrainDiffusionImageAccelerateWorkspace
        cfg = tiny_config()
        cfg.training.max_train_steps = 22
        cfg.training.val_every = 5
        cfg.training.sample_every = 5
        cfg.training.checkpoint_every = 5
        cfg.training.rollout_every = 10
        TinyRunner.calls = 0
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.dict(os.environ, {'ACCELERATE_USE_CPU': 'true'}), \
                mock.patch(cls.__module__ + '.wandb.init'), \
                mock.patch(cls.__module__ + '.wandb.config'):
            workspace = cls(cfg, output_dir=directory)
            workspace.run()
            workspace._saving_thread.join()
            logs = [json.loads(line) for line in (pathlib.Path(directory) / 'logs.json.txt').read_text().splitlines()]
            for key in ('val_action_mse', 'train_action_mse'):
                self.assertEqual([row['epoch'] for row in logs if key in row], [0, 5, 10])
            self.assertEqual([row['epoch'] for row in logs if 'test/n_rollouts' in row], [10])
            checkpoints = pathlib.Path(directory) / 'checkpoints'
            self.assertEqual({path.name for path in checkpoints.glob('epoch_*.ckpt')},
                {'epoch_0.ckpt', 'epoch_5.ckpt', 'epoch_10.ckpt'})
            self.assertEqual(workspace.epoch, 11)
            cfg.training.resume = True
            cfg.training.max_train_steps = 24
            resumed = cls(cfg, output_dir=directory)
            resumed.run()
            resumed._saving_thread.join()
            self.assertEqual(resumed.epoch, 12)
            self.assertEqual(resumed.model.train_calls, 2)
            self.assertEqual(TinyRunner.calls, 2)
            logs = [json.loads(line) for line in (pathlib.Path(directory) / 'logs.json.txt').read_text().splitlines()]
            self.assertEqual(logs[-1]['epoch'], 11)
            checkpoint = checkpoints / 'latest.ckpt'
            payload = torch.load(checkpoint, pickle_module=dill, map_location='cpu')
            self.assertEqual(payload['pickles'].keys(), {'global_step', 'epoch', '_output_dir'})
            self.assertEqual(dill.loads(payload['pickles']['epoch']), logs[-1]['epoch'])
            self.assertEqual(dill.loads(payload['pickles']['global_step']), logs[-1]['global_step'])
            cfg.training.max_train_steps = 26
            legacy = cls(cfg, output_dir=directory)
            legacy.run()
            legacy._saving_thread.join()
            self.assertEqual(legacy.epoch, 13)
            self.assertEqual(legacy.model.train_calls, 2)
            logs = [json.loads(line) for line in (pathlib.Path(directory) / 'logs.json.txt').read_text().splitlines()]
            self.assertEqual(logs[-1]['epoch'], 12)


    @unittest.skipUnless(importlib.util.find_spec('accelerate'), 'requires accelerate')
    def test_checkpoint_validation_and_final_rollout_use_the_same_record(self):
        from zprl.workspace.train_diffusion_image_accelerate_workspace import TrainDiffusionImageAccelerateWorkspace
        cls = TrainDiffusionImageAccelerateWorkspace
        cfg = tiny_config(use_ema=True)
        cfg.training.max_train_steps = 24
        cfg.training.val_every = 7
        cfg.training.checkpoint_every = 5
        cfg.training.rollout_every = 10
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.dict(os.environ, {'ACCELERATE_USE_CPU': 'true'}), \
                mock.patch(cls.__module__ + '.wandb.init') as init, \
                mock.patch(cls.__module__ + '.wandb.config'):
            workspace = cls(cfg, output_dir=directory)
            workspace.run()
            workspace._saving_thread.join()
            logs = [json.loads(line) for line in (pathlib.Path(directory) / 'logs.json.txt').read_text().splitlines()]
            self.assertEqual([row['epoch'] for row in logs if 'val_action_mse' in row], [0, 7, 11])
            self.assertEqual([row['epoch'] for row in logs if 'test/n_rollouts' in row], [10, 11])
            self.assertEqual([row['global_step'] for row in logs], list(range(1, 25)))
            self.assertTrue(workspace.ema_model.training)
            self.assertEqual({path.name for path in (pathlib.Path(directory) / 'checkpoints').glob('epoch_*.ckpt')},
                {'epoch_0.ckpt', 'epoch_11.ckpt'})
            for path in (pathlib.Path(directory) / 'checkpoints').glob('*.ckpt'):
                with path.open('rb') as file:
                    payload = torch.load(file, pickle_module=dill, map_location='cpu')
                epoch = dill.loads(payload['pickles']['epoch'])
                step = dill.loads(payload['pickles']['global_step'])
                record = next(row for row in logs if row['global_step'] == step)
                self.assertEqual(record['epoch'], epoch)
                self.assertIn('val_action_mse', record)
                self.assertIn('ema_model', payload['state_dicts'])
            for call in init.return_value.log.call_args_list:
                self.assertEqual(call.kwargs['step'], call.args[0]['global_step'])


class AccelerateDeviceMappingTest(unittest.TestCase):
    def test_default_mapping_is_unchanged(self):
        from zprl.workspace.train_diffusion_image_accelerate_workspace import _configure_cuda_devices
        with mock.patch('torch.cuda.set_device') as set_device:
            before = dict(os.environ)
            _configure_cuda_devices(None)
            set_device.assert_not_called()
            self.assertEqual(dict(os.environ), before)

    def test_explicit_mapping_preserves_visible_devices_and_local_rank(self):
        from zprl.workspace.train_diffusion_image_accelerate_workspace import _configure_cuda_devices
        for rank in (0, 1):
            with self.subTest(rank=rank), mock.patch.dict(os.environ, {
                    'CUDA_VISIBLE_DEVICES': '1,0,2,3', 'LOCAL_WORLD_SIZE': '2', 'LOCAL_RANK': str(rank)}), \
                    mock.patch('torch.cuda.device_count', return_value=4), \
                    mock.patch('torch.cuda.set_device') as set_device:
                _configure_cuda_devices(OmegaConf.create([2, 3]))
                set_device.assert_called_once_with(rank + 2)
                self.assertEqual(os.environ['ACCELERATE_TORCH_DEVICE'], f'cuda:{rank + 2}')
                self.assertEqual(os.environ['ACCELERATE_BYPASS_DEVICE_MAP'], 'true')
                self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], '1,0,2,3')
                self.assertEqual(os.environ['LOCAL_RANK'], str(rank))

    def test_invalid_devices_are_rejected_before_setting_cuda(self):
        from zprl.workspace.train_diffusion_image_accelerate_workspace import _configure_cuda_devices
        for ids in ([], [2], [2, 3, 0], [2, 2], [-1, 3], [2, 4], ['2', 3], [True, 3], 2, '2,3'):
            with self.subTest(gpu_ids=ids), \
                    mock.patch.dict(os.environ, {'LOCAL_WORLD_SIZE': '2', 'LOCAL_RANK': '0'}), \
                    mock.patch('torch.cuda.device_count', return_value=4), \
                    mock.patch('torch.cuda.set_device') as set_device:
                with self.assertRaises(ValueError):
                    _configure_cuda_devices(ids)
                set_device.assert_not_called()


if __name__ == '__main__':
    unittest.main()
