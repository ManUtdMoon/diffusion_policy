import io
import json
import pathlib
import tempfile
import unittest
from unittest.mock import Mock, patch

import hydra
import numpy as np
import torch
import tqdm
from omegaconf import OmegaConf

from tests.test_oneleg_online_history import FakeBase
from tests.test_oneleg_online_wrapper import ScriptedOneLegEnv
from zprl.common.oneleg_episode_replay_buffer import OneLegEpisodeCollector
from zprl.env.furniturebench.oneleg_online_env import OneLegOnlineEnv
from zprl.workspace.train_online_reactive_history_oneleg_workspace import TrainOnlineReactiveHistoryOneLegWorkspace
from zprl.workspace.train_online_reactive_history_noise_oneleg_workspace import TrainOnlineReactiveHistoryNoiseOneLegWorkspace


def config(method='resrl', **overrides):
    OmegaConf.register_new_resolver('eval', eval, replace=True)
    with hydra.initialize_config_dir(config_dir=str(pathlib.Path(__file__).resolve().parents[1] / 'zprl/config'), version_base=None):
        cfg = hydra.compose(config_name='train_online_reactive_history' + ('_noise' if method == 'dsrl' else '') + '_oneleg_workspace')
    values = dict({'training.device': 'cpu', 'training.debug': True,
        'training.num_steps': 13, 'training.learning_start': 11,
        'training.training_freq': 2, 'training.utd': 1., 'training.batch_size': 1,
        'training.buffer_size': 50, 'training.log_every': 3,
        'online_task.base_ckpt': 'mock-base.ckpt',
        'online_task.n_envs': 2, 'online_task.max_steps': 18,
        'n_action_steps': 8, 'n_rl_steps': 4,
        'res_policy.history_length': 5, 'res_policy.sub_ratio': 2,
        'res_policy.hidden_dim': 8, 'res_policy.gru_input_dim': 8,
        'res_policy.gru_hidden_dim': 8, 'res_policy.num_qs': 2}, **overrides)
    if method == 'dsrl':
        values.pop('n_rl_steps')
        values['n_action_steps'] = 4
        values['n_noise_steps'] = 2
        values = {key.replace('res_policy.', 'noise_policy.'): value for key, value in values.items()}
    for key, value in values.items():
        OmegaConf.update(cfg, key, value)
    return cfg


class OneLegWorkspaceTest(unittest.TestCase):
    def test_progress_clears_between_sampling_updates_and_on_exit(self):
        real_tqdm = tqdm.tqdm
        for method in ('resrl', 'dsrl'):
            for fail_at in (None, 6):
                with self.subTest(method=method, fail_at=fail_at):
                    output = io.StringIO()
                    workspace_class = (TrainOnlineReactiveHistoryOneLegWorkspace if method == 'resrl'
                                       else TrainOnlineReactiveHistoryNoiseOneLegWorkspace)
                    original_update = workspace_class._update
                    updates = []

                    def progress(*a, **kw):
                        return real_tqdm(*a, file=output, **kw)

                    def checked_update(workspace, cfg, optimizers):
                        self.assertEqual(output.getvalue().rsplit('\r', 1)[-1].strip(), '')
                        result = original_update(workspace, cfg, optimizers)
                        self.assertEqual(output.getvalue().rsplit('\r', 1)[-1].strip(), '')
                        updates.append(workspace.global_update)
                        return result

                    with patch('tqdm.tqdm', side_effect=progress), \
                         patch.object(workspace_class, '_update', checked_update):
                        self.run_workspace(config(method, **{'training.tqdm_interval_sec': 0.,
                            'training.training_freq': 1}), fail_at=fail_at)
                    self.assertIn('Sampling', output.getvalue())
                    if fail_at is None:
                        self.assertTrue(updates)
                        self.assertIn('Updating', output.getvalue())
                    self.assertNotIn('\x1b[A', output.getvalue())  # Both phases use one terminal row.
                    self.assertEqual(output.getvalue().rsplit('\r', 1)[-1].strip(), '')

    def test_update_threshold_crossings_preserve_remainder(self):
        workspace, logs = self.run_workspace(config(**{
            'training.training_freq': 1, 'training.utd': 1., 'training.log_every': 1}))
        self.assertEqual(workspace.global_update, 4)
        self.assertEqual([row['info/chunk_step'] for row in logs],
                         [11, 12, 14, 14])

    def test_logging_starts_at_warmup_even_before_log_every(self):
        for method in ('resrl', 'dsrl'):
            for warmup, budget, expected in ((11, 13, [11, 14]), (13, 13, [14, 14]), (11, 5, [])):
                with self.subTest(method=method, warmup=warmup, budget=budget):
                    workspace, logs = self.run_workspace(config(method, **{
                        'training.learning_start': warmup, 'training.num_steps': budget,
                        'training.log_every': 100}))
                    self.assertEqual([row['info/chunk_step'] for row in logs], expected)
                    self.assertTrue(all(row['info/chunk_step'] >= warmup for row in logs))

    def run_workspace(self, cfg, fail_at=None):
        noise = 'noise_policy' in cfg
        module = 'zprl.workspace.train_online_reactive_history' + ('_noise' if noise else '') + '_oneleg_workspace'
        workspace_class = TrainOnlineReactiveHistoryNoiseOneLegWorkspace if noise else TrainOnlineReactiveHistoryOneLegWorkspace
        interval = cfg.n_action_steps if noise else cfg.n_rl_steps
        base = FakeBase()
        before = {key: value.clone() for key, value in base.state_dict().items()}
        native = ScriptedOneLegEnv((3, None))
        native.fail_at = fail_at
        env = OneLegOnlineEnv(4, max_steps=cfg.online_task.max_steps, env=native)
        wb = Mock()
        with tempfile.TemporaryDirectory() as output:
            workspace = workspace_class(cfg, output)
            with patch(module + '.load_oneleg_base', return_value=(base, None)), \
                 patch(module + '.OneLegOnlineEnv', return_value=env), \
                 patch(module + '.wandb.init', return_value=wb), \
                 patch.object(env, 'reset', wraps=env.reset) as reset, \
                 patch.object(OneLegEpisodeCollector, 'discard_pending', autospec=True,
                              side_effect=OneLegEpisodeCollector.discard_pending) as discard:
                if fail_at:
                    with self.assertRaisesRegex(RuntimeError, 'native simulation failed'):
                        workspace.run()
                else:
                    workspace.run()
            self.assertTrue(reset.call_args_list)
            self.assertTrue(all(not call.args and not call.kwargs for call in reset.call_args_list))
            discard.assert_called_once_with(workspace.collector)
            logs = [json.loads(line) for line in pathlib.Path(output, 'logs.json.txt').read_text().splitlines()]
            self.assertEqual(list(pathlib.Path(output).glob('**/*.ckpt')), [])
        self.assertEqual(native.close_calls, 1)
        wb.finish.assert_called_once()
        self.assertEqual(len(wb.log.call_args_list), len(logs))
        for call in wb.log.call_args_list:
            self.assertEqual(call.kwargs['step'], call.args[0]['info/global_step'])
            self.assertEqual(call.args[0]['info/global_step'], call.args[0]['info/chunk_step'] * interval)
            self.assertNotIn('info/res_ratio', call.args[0])
            self.assertNotIn('info/ready_step', call.args[0])
            self.assertAlmostEqual(call.args[0]['info/idle_ratio'],
                call.args[0]['info/idle_steps'] / (call.args[0]['info/solid_steps'] + call.args[0]['info/idle_steps']))
        self.assertFalse(hasattr(env, 'global_step'))
        self.assertFalse(hasattr(env, 'executed_steps'))
        self.assertFalse(base.training)
        for key, value in base.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in base.parameters()))
        self.assertTrue(all(count == 2 for count in base.rows))
        self.assertTrue(all(np.isfinite(value) for record in logs for value in record.values()))
        return workspace, logs

    def test_real_updates_warmup_budget_and_incomplete_discard(self):
        workspace, logs = self.run_workspace(config())
        self.assertEqual(workspace.global_step, 14)  # Explicit small debug budget survives.
        self.assertEqual(workspace.global_update, 4)
        self.assertEqual(workspace.replay.committed_episodes, 5)
        self.assertEqual(workspace.replay.committed_transitions, 13)
        self.assertEqual(workspace.collector.pending_transitions, 0)
        self.assertEqual(workspace.global_step - workspace.cfg.training.num_steps, 1)
        self.assertEqual(logs[-1]['info/global_step'], 56)
        self.assertEqual(logs[-1]['info/chunk_step'], 14)
        self.assertEqual(logs[-1]['info/solid_steps'], 49)
        self.assertEqual(logs[-1]['info/idle_steps'], 31)
        self.assertAlmostEqual(logs[-1]['info/idle_ratio'], 31 / 80)
        self.assertEqual([row['info/chunk_step'] for row in logs], [11, 12, 14])
        self.assertIn('loss/actor_loss', logs[-1])
        self.assertIn('loss/alpha_loss', logs[-1])
        # Warmup collects the policy's residual before any optimizer update.
        storage = workspace.replay.storage
        self.assertTrue(np.any(storage.actions[0] != 0))
        self.assertTrue(np.any(storage.actions[1] != 0))
        self.assertTrue(np.any(storage.actions[2:storage.pos] != 0))
        self.assertFalse(workspace.res_policy.q_targets.training)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in workspace.res_policy.q_targets.parameters()))
        optimizers = workspace.res_policy.get_optimizer(1e-4, 1e-4, workspace.cfg.training.enc_lr)
        for key, module in (('actor_optimizer', workspace.res_policy.actor), ('q_optimizer', workspace.res_policy.qs)):
            groups = optimizers[key].param_groups
            self.assertEqual(len(groups), 1)
            self.assertEqual(groups[0]['lr'], 1e-4)
            self.assertEqual({id(p) for p in groups[0]['params']}, {id(p) for p in module.parameters()})
        self.assertTrue(any(not torch.equal(q, target) for q, target in zip(workspace.res_policy.qs.parameters(), workspace.res_policy.q_targets.parameters())))

    def test_sampling_with_replacement_does_not_require_batch_size_stored_transitions(self):
        workspace, logs = self.run_workspace(config(**{'training.policy_freq': 100,
            'training.batch_size': 30, 'training.num_steps': 11}))
        self.assertEqual(workspace.global_update, 2)
        self.assertEqual(workspace.replay.size(), 7)
        self.assertNotIn('info/replay_wait_decisions', logs[-1])
        self.assertNotIn('loss/actor_loss', logs[-1])
        self.assertNotIn('loss/alpha_loss', logs[-1])

    def test_failed_chunk_is_not_counted_or_committed_and_cleanup_runs(self):
        workspace, logs = self.run_workspace(config(), fail_at=6)
        self.assertEqual(workspace.global_step, 2)
        self.assertEqual(workspace.global_update, 0)
        self.assertEqual(workspace.replay.committed_transitions, 1)
        self.assertEqual(workspace.collector.pending_transitions, 0)
        self.assertEqual(logs, [])
        self.assertEqual(workspace.last_log['info/global_step'], 8)
        self.assertEqual(workspace.last_log['info/chunk_step'], 2)
        self.assertEqual(workspace.last_log['info/solid_steps'], 8)
        self.assertEqual(workspace.last_log['info/idle_steps'], 2)

    def test_dsrl_updates_and_counts_match_resrl_for_same_events(self):
        res, res_logs = self.run_workspace(config())
        noise, noise_logs = self.run_workspace(config('dsrl'))
        for key in ('info/global_step', 'info/chunk_step', 'info/global_update',
                    'info/num_episodes', 'info/solid_steps', 'info/idle_steps', 'info/idle_ratio'):
            self.assertEqual([row[key] for row in res_logs], [row[key] for row in noise_logs])
        self.assertEqual(noise.replay.committed_transitions, res.replay.committed_transitions)
        self.assertEqual(noise.replay.committed_episodes, res.replay.committed_episodes)
        self.assertIn('loss/actor_loss', noise_logs[-1])
        self.assertIn('loss/alpha_loss', noise_logs[-1])
        self.assertEqual(noise.noise_policy.action_dim, 20)
        self.assertEqual(noise.noise_policy.exec_dim, 40)
        storage = noise.replay.storage
        self.assertEqual(storage.actions.shape[-1], 20)
        self.assertTrue(np.any(storage.actions[0] != 0))  # Gaussian warmup, not base-only.
        self.assertTrue(np.all(np.abs(storage.actions[:storage.pos]) <= 1))
        first_noise = torch.tensor(storage.actions[0, 0]).reshape(1, 2, 10)
        torch.testing.assert_close(noise.base_policy.noises[0][:1], first_noise.repeat(1, 8, 1) * 1.5)
        self.assertFalse(noise.noise_policy.q_targets.training)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in noise.noise_policy.q_targets.parameters()))

    def test_dsrl_failure_and_sampling_with_replacement(self):
        workspace, logs = self.run_workspace(config('dsrl'), fail_at=6)
        self.assertEqual(logs, [])
        self.assertEqual((workspace.global_step, workspace.replay.committed_transitions), (2, 1))
        self.assertEqual(workspace.collector.pending_transitions, 0)
        workspace, logs = self.run_workspace(config('dsrl', **{'training.policy_freq': 100,
            'training.batch_size': 30, 'training.num_steps': 11}))
        self.assertEqual((workspace.global_update, workspace.replay.size()), (2, 7))
        self.assertNotIn('loss/actor_loss', logs[-1])


if __name__ == '__main__':
    unittest.main()
