import copy
import json
import pathlib
import tempfile
import unittest
from unittest.mock import Mock, patch

import dill
import hydra
import torch
from hydra import compose, initialize_config_dir

from zprl.workspace.base_workspace import BaseWorkspace
from zprl.workspace.train_soe_workspace import TrainSoeWorkspace
from zprl.model.common.lr_scheduler import get_scheduler


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base = torch.nn.Parameter(torch.tensor(1.0))
        self.vib = torch.nn.Parameter(torch.tensor(1.0))

    def set_normalizer(self, normalizer):
        pass

    def forward(self, batch):
        loss = self.base.square() + self.vib.square()
        return loss, {'il_loss': self.base.square().item()}

    def encode_obs(self, obs):
        return obs['x']

    def conditional_predict(self, obs):
        return {'action_pred': obs * self.base}

    def predict_action(self, obs):
        raise AssertionError('Evaluation must bypass VIB')


class TinyDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, index):
        return {'obs': {'x': torch.ones(1, 1)}, 'action': torch.zeros(1, 1)}

    def get_normalizer(self):
        return None

    def get_validation_dataset(self):
        return self


class SoeTrainingTest(unittest.TestCase):
    def test_update_limit_and_final_hooks(self):
        for limit, epochs, expected_steps, eval_steps in [
                (3, 2, 3, [2, 3]), (4, 3, 4, [2, 4]),
                (1, 1, 1, [1]), (None, 2, 4, [2, 4])]:
            with self.subTest(limit=limit), tempfile.TemporaryDirectory() as tmp:
                config_dir = pathlib.Path(__file__).resolve().parents[1] / 'zprl/config'
                with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
                    cfg = compose(config_name='train_soe_workspace')
                cfg.training.device = 'cpu'
                cfg.task.env_runner.render_device_id = 0
                cfg.training.max_grad_steps = limit
                cfg.training.num_epochs = 2
                cfg.training.lr_warmup_steps = 0
                cfg.training.rollout_every = 100
                cfg.training.checkpoint_every = 3
                cfg.training.val_every = 1
                cfg.training.sample_every = 1
                cfg.dataloader.batch_size = 1
                cfg.dataloader.num_workers = 0
                cfg.val_dataloader.num_workers = 0
                cfg.checkpoint.topk.k = 0
                cfg.checkpoint.save_last_ckpt = False
                workspace = TrainSoeWorkspace.__new__(TrainSoeWorkspace)
                BaseWorkspace.__init__(workspace, cfg, output_dir=tmp)
                workspace.model = TinyPolicy()
                workspace.ema_model = copy.deepcopy(workspace.model)
                workspace.optimizer = torch.optim.SGD(workspace.model.parameters(), lr=0.1)
                workspace.global_step = workspace.epoch = workspace.optimizer_step = 0
                runner = Mock()
                observed_steps = []

                def rollout(policy):
                    self.assertIs(policy.policy, workspace.ema_model)
                    self.assertFalse(policy.policy.training)
                    policy.predict_action({'x': torch.ones(1, 1, 1)})
                    observed_steps.append(workspace.optimizer_step)
                    return {'test/mean_score': 0.5}

                runner.run.side_effect = rollout
                instantiate = hydra.utils.instantiate

                def make(config, **kwargs):
                    if config is cfg.task.dataset:
                        return TinyDataset()
                    if config is cfg.task.env_runner:
                        return runner
                    return instantiate(config, **kwargs)

                run = Mock()
                with patch('hydra.utils.instantiate', side_effect=make), \
                        patch('zprl.workspace.train_soe_workspace.wandb.init', return_value=run), \
                        patch('zprl.workspace.train_soe_workspace.wandb.config'), \
                        patch('zprl.workspace.train_soe_workspace.get_scheduler', wraps=get_scheduler) as scheduler:
                    workspace.run()
                self.assertEqual(workspace.global_step, expected_steps)
                self.assertEqual(cfg.training.num_epochs, epochs)
                self.assertEqual(observed_steps, eval_steps)
                self.assertEqual(scheduler.call_args.kwargs['num_training_steps'], expected_steps)
                self.assertLess(workspace.model.base.item(), 1)
                self.assertLess(workspace.model.vib.item(), 1)
                runner.close.assert_called_once()
                run.finish.assert_called_once()
                logs = [json.loads(line) for line in pathlib.Path(tmp, 'logs.json.txt').read_text().splitlines()]
                self.assertEqual([row['global_step'] for row in logs], list(range(expected_steps)))
                self.assertEqual([row['optimizer_step'] for row in logs], list(range(1, expected_steps + 1)))
                self.assertEqual([call.kwargs['step'] for call in run.log.call_args_list], list(range(expected_steps)))
                run.define_metric.assert_not_called()
                self.assertIn('val_loss', logs[-1])
                checkpoint = pathlib.Path(tmp, 'checkpoints', f'step_{expected_steps:06d}.ckpt')
                payload = torch.load(checkpoint, pickle_module=dill, map_location='cpu')
                self.assertEqual(dill.loads(payload['pickles']['global_step']), expected_steps - 1)
                self.assertEqual(dill.loads(payload['pickles']['optimizer_step']), expected_steps)
                self.assertEqual(payload['cfg'].training.num_epochs, epochs)
                self.assertIn('ema_model', payload['state_dicts'])


if __name__ == '__main__':
    unittest.main()
