import pathlib
import tempfile
import unittest
from unittest.mock import Mock, patch

import dill
import torch
from omegaconf import OmegaConf

from tests.test_oneleg_online_history import FakeBase, make_session
from tests.test_oneleg_online_workspace import config
from tests.test_oneleg_online_wrapper import ScriptedOneLegEnv
from zprl.common.oneleg_online import oneleg_online_snapshot, load_oneleg_online
from zprl.env.furniturebench.oneleg_online_env import OneLegOnlineEnv
from zprl.env_runner.oneleg_online_runner import OneLegOnlineRunner


class OneLegSnapshotTest(unittest.TestCase):
    def test_both_methods_roundtrip_external_base_and_strict_weights(self):
        for method in ('resrl', 'dsrl'):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as directory:
                cfg = config(method)
                if method == 'resrl':
                    cfg.res_policy.critic_action_input = 'exec'
                base_path = pathlib.Path(directory, 'base.ckpt')
                base_path.write_bytes(b'external offline checkpoint')
                cfg.online_task.base_ckpt = str(base_path)
                session, _, steering = make_session(method)
                snapshot = oneleg_online_snapshot(cfg, method, steering, 14, 4, 5)
                key = 'res_policy' if method == 'resrl' else 'noise_policy'
                self.assertEqual(set(snapshot), {'cfg', 'method', key, 'base_ckpt',
                    'base_sha256', 'global_step', 'global_update', 'num_episodes'})
                self.assertTrue(all(value.device.type == 'cpu' for value in snapshot[key].values()))
                self.assertFalse(any('base_policy' in name for name in snapshot[key]))
                path = pathlib.Path(directory, 'online.ckpt')
                torch.save(snapshot, path, pickle_module=dill)
                with patch('zprl.common.oneleg_online.load_oneleg_base',
                        return_value=(FakeBase(), OmegaConf.create({'task_name': 'oneleg_image'}))):
                    restored, loaded_cfg, metadata = load_oneleg_online(path, device='cpu')
                self.assertEqual(metadata['method'], method)
                self.assertEqual(metadata['global_step'], 14)
                self.assertEqual(loaded_cfg.online_task.base_ckpt, str(base_path))
                restored_steering = getattr(restored, key)
                self.assertTrue(all(not p.requires_grad for p in restored_steering.parameters()))
                env = OneLegOnlineEnv(4, env=ScriptedOneLegEnv())
                obs = env.reset()
                for combined in (session, restored):
                    combined.eval()
                    combined.reset(obs)
                torch.testing.assert_close(session.predict_action(obs, deterministic=True)['action'],
                                           restored.predict_action(obs, deterministic=True)['action'])
                base_path.write_bytes(b'changed offline checkpoint')
                with self.assertRaisesRegex(ValueError, 'differs from the snapshot'):
                    load_oneleg_online(path, device='cpu')

    def test_online_and_base_interface_must_match(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = config()
            cfg.online_task.base_ckpt = str(pathlib.Path(directory, 'base'))
            pathlib.Path(cfg.online_task.base_ckpt).write_bytes(b'base')
            _, _, steering = make_session('resrl')
            snapshot = oneleg_online_snapshot(cfg, 'resrl', steering, 0, 0, 0)
            snapshot['cfg'].horizon = 8
            path = pathlib.Path(directory, 'online.ckpt')
            torch.save(snapshot, path, pickle_module=dill)
            with patch('zprl.common.oneleg_online.load_oneleg_base',
                    return_value=(FakeBase(), OmegaConf.create({'task_name': 'oneleg_image'}))):
                with self.assertRaisesRegex(ValueError, 'interfaces differ'):
                    load_oneleg_online(path, device='cpu')


class OneLegRunnerTest(unittest.TestCase):
    def test_fixed_ids_wait_for_whole_batch_and_method_defaults(self):
        for method in ('resrl', 'dsrl'):
            with self.subTest(method=method):
                session, base, steering = make_session(method)
                native = ScriptedOneLegEnv((None, 1))
                env = OneLegOnlineEnv(4, max_steps=9, mode='eval', env=native)
                runner = OneLegOnlineRunner('/tmp', n_rollouts=3, n_envs=2,
                                           max_steps=9, n_action_steps=4, seed=100)
                with patch('zprl.env_runner.oneleg_online_runner.OneLegOnlineEnv', return_value=env), \
                     patch.object(env, 'reset', wraps=env.reset) as reset, \
                     patch.object(session, 'predict_action', wraps=session.predict_action) as predict:
                    metrics = runner.run(session, policy_seed=77)
                self.assertEqual([call.kwargs['seed'] for call in reset.call_args_list], [100, 102])
                self.assertEqual([result['rollout_id'] for result in runner.last_results], [0, 1, 2])
                self.assertEqual([result['seed'] for result in runner.last_results], [100, 101, 102])
                self.assertEqual([result['termination_reason'] for result in runner.last_results],
                                 ['timeout', 'success', 'timeout'])
                self.assertEqual((metrics['success_count'], metrics['timeout_count'], metrics['failure_count']), (1, 2, 0))
                self.assertTrue(all(call.kwargs['deterministic'] == (method == 'resrl') for call in predict.call_args_list))
                self.assertTrue(all(row == 2 for row in base.rows))
                self.assertEqual(native.close_calls, 1)
                self.assertEqual(native.steps, 9)
                self.assertEqual((metrics['solid_steps'], metrics['idle_steps']), (20, 16))
                self.assertFalse(base.training)

    def test_failed_batch_retains_success_once_then_recreates(self):
        session, _, _ = make_session('dsrl')
        first, second = ScriptedOneLegEnv((1, None)), ScriptedOneLegEnv((None, None))
        first.fail_at = 3
        envs = [OneLegOnlineEnv(4, max_steps=5, mode='eval', env=native) for native in (first, second)]
        runner = OneLegOnlineRunner('/tmp', n_rollouts=4, n_envs=2, n_action_steps=4, max_steps=5)
        with patch('zprl.env_runner.oneleg_online_runner.OneLegOnlineEnv', side_effect=envs):
            metrics = runner.run(session, deterministic=True)
        self.assertEqual([result['termination_reason'] for result in runner.last_results],
                         ['success', 'failure', 'timeout', 'timeout'])
        self.assertNotIn('error', runner.last_results[0])
        self.assertIn('native simulation failed', runner.last_results[1]['error'])
        self.assertEqual((metrics['success_count'], metrics['timeout_count'], metrics['failure_count']), (1, 2, 1))
        self.assertEqual((first.close_calls, second.close_calls), (1, 1))

    def test_reset_failure_and_program_errors(self):
        session, _, _ = make_session('resrl')
        native = ScriptedOneLegEnv()
        env = OneLegOnlineEnv(4, mode='eval', env=native)
        runner = OneLegOnlineRunner('/tmp', n_rollouts=2, n_envs=2, n_action_steps=4)
        with patch('zprl.env_runner.oneleg_online_runner.OneLegOnlineEnv', return_value=env), \
             patch.object(env, 'reset', side_effect=RuntimeError('reset failed')):
            metrics = runner.run(session)
        self.assertEqual(metrics['failure_count'], 2)
        self.assertEqual(native.close_calls, 1)
        for source in ('policy', 'native_protocol'):
            with self.subTest(source=source):
                native = ScriptedOneLegEnv()
                native.unexpected_done = source == 'native_protocol'
                env = OneLegOnlineEnv(4, mode='eval', env=native)
                with patch('zprl.env_runner.oneleg_online_runner.OneLegOnlineEnv', return_value=env):
                    if source == 'policy':
                        with patch.object(session, 'predict_action', side_effect=RuntimeError('policy error')):
                            with self.assertRaisesRegex(RuntimeError, 'policy error'):
                                runner.run(session)
                    else:
                        with self.assertRaisesRegex(RuntimeError, 'Native done'):
                            runner.run(session)
                self.assertEqual(native.close_calls, 1)

    def test_video_writers_close_on_simulation_failure(self):
        session, _, _ = make_session('resrl')
        native = ScriptedOneLegEnv((1, None))
        native.fail_at = 5
        env = OneLegOnlineEnv(4, mode='eval', env=native)
        writer = Mock()
        with tempfile.TemporaryDirectory() as output, \
             patch('zprl.env_runner.oneleg_online_runner.OneLegOnlineEnv', return_value=env), \
             patch('imageio.v2.get_writer', return_value=writer) as factory:
            runner = OneLegOnlineRunner(output, n_rollouts=2, n_envs=2,
                                       n_action_steps=4, record_video=True, n_vis=1)
            runner.run(session)
        factory.assert_called_once()
        writer.close.assert_called_once()
        self.assertEqual(writer.append_data.call_count, 2)  # Initial + terminal; no idle frames.


if __name__ == '__main__':
    unittest.main()
