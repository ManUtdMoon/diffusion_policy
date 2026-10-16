import math
import pathlib
import tempfile
import unittest
from unittest import mock

from click.testing import CliRunner
import numpy as np
import torch

from eval_oneleg import main
from zprl.env_runner.oneleg_image_runner import OneLegImageRunner
from zprl.policy.base_image_policy import BaseImagePolicy


class FakePolicy(BaseImagePolicy):
    horizon = 16
    n_obs_steps = 1

    def __init__(self):
        super().__init__()
        self.calls = 0
        self.resets = 0

    def reset(self):
        self.resets += 1

    def predict_action(self, obs):
        self.calls += 1
        batch = obs['robot_state'].shape[0]
        assert obs['robot_state'].shape == (batch, 1, 16)
        assert obs['color_image1'].shape == (batch, 1, 3, 240, 320)
        action = torch.zeros(batch, 16, 10)
        action[:, :, 3] = action[:, :, 7] = 1
        action[:, :, -1] = torch.arange(16)
        return {'action_pred': action}


class FakeEnv:
    def __init__(self, num_envs, max_env_steps, successes=None, early_done=None,
            fail_step=None, fail_reset=False, **kwargs):
        self.num_envs = num_envs
        self.max_steps = max_env_steps
        self.successes = successes or {}
        self.early_done = early_done or {}
        self.fail_step = fail_step
        self.fail_reset = fail_reset
        self.actions = []
        self.reset_seeds = []
        self.close_count = 0

    def observation(self):
        wrist = torch.zeros(self.num_envs, 3, 240, 320)
        front = torch.zeros_like(wrist)
        wrist[:, 0] = 1
        front[:, 1] = 1
        return {'color_image1': wrist,
                'color_image2': front,
                'robot_state': torch.full((self.num_envs, 16), float(self.step_count))}

    def reset(self, seed=None):
        self.reset_seeds.append(seed)
        if seed is not None:
            np.random.seed(seed)
        np.random.rand()
        if self.fail_reset:
            raise RuntimeError('reset error')
        self.step_count = 0
        self.assembled = torch.zeros(self.num_envs, dtype=torch.bool)
        return self.observation()

    def step(self, action):
        self.actions.append(action.clone())
        self.step_count += 1
        if self.step_count == self.fail_step:
            raise RuntimeError('singular matrix')
        reward = torch.zeros(self.num_envs, 1)
        done = torch.zeros(self.num_envs, 1, dtype=torch.bool)
        for i, step in self.successes.items():
            if self.step_count == step:
                self.assembled[i] = True
                reward[i] = 1
        for i, step in self.early_done.items():
            done[i] = self.step_count >= step
        return self.observation(), reward, done, {}

    def is_success(self):
        return self.assembled.clone()

    def close(self):
        self.close_count += 1


class Writer:
    def __init__(self):
        self.frames = []
        self.closed = False

    def append_data(self, frame):
        self.frames.append(frame)

    def close(self):
        self.closed = True


class OneLegRunnerTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.policy = FakePolicy()

    def runner(self, **kwargs):
        return OneLegImageRunner(output_dir=self.directory.name,
            n_rollouts=kwargs.pop('n_rollouts', 3), n_envs=kwargs.pop('n_envs', 3),
            max_steps=kwargs.pop('max_steps', 3), **kwargs)

    def test_success_timeout_early_failure_and_first_episode_length(self):
        env = FakeEnv(3, 3, successes={0: 1}, early_done={2: 2})
        runner = self.runner()
        with mock.patch('zprl.env_runner.oneleg_image_runner.OneLegEnv', return_value=env), \
                mock.patch.object(runner, '_open_videos', side_effect=AssertionError('offline video')):
            metrics = runner.run(self.policy)
        for name in ('success', 'timeout', 'failure'):
            self.assertEqual(metrics[f'test/{name}_count'], 1)
            self.assertAlmostEqual(metrics[f'test/{name}_rate'], 1 / 3)
        self.assertEqual(metrics['test/success_mean_epi_len'], 1)
        self.assertEqual([x['epi_len'] for x in runner.last_results], [1, 3, 2])
        self.assertEqual([x['result'] for x in runner.last_results], ['success', 'timeout', 'failure'])
        self.assertEqual(len(env.actions), 3)
        torch.testing.assert_close(env.actions[1][0], torch.tensor([0, 0, 0, 1, 0, 0, 0, 1, 0, 1.]))
        self.assertFalse((pathlib.Path(self.directory.name) / 'videos').exists())
        self.assertEqual(env.close_count, 1)

    def test_sim_error_preserves_finished_success_and_recreates_next_batch(self):
        first = FakeEnv(3, 3, successes={0: 1}, fail_step=2)
        second = FakeEnv(3, 3, successes={0: 3})
        runner = self.runner(n_rollouts=6)
        with mock.patch('zprl.env_runner.oneleg_image_runner.OneLegEnv',
                side_effect=[first, second]) as factory, self.assertLogs(level='ERROR'):
            metrics = runner.run(self.policy)
        self.assertEqual(metrics['test/success_count'], 2)
        self.assertEqual(metrics['test/failure_count'], 2)
        self.assertEqual(metrics['test/timeout_count'], 2)
        self.assertEqual(metrics['test/success_mean_epi_len'], 2)
        self.assertEqual([call.kwargs['num_envs'] for call in factory.call_args_list], [3, 3])
        self.assertEqual(runner.last_results[0]['result'], 'success')
        self.assertEqual(runner.last_results[1]['error'], 'singular matrix')
        self.assertEqual(first.close_count, 1)
        self.assertEqual(second.close_count, 1)

    def test_chunk_prefix_no_repeat_and_exact_timeout_budget(self):
        env = FakeEnv(3, 5)
        runner = self.runner(n_action_steps=2, max_steps=5)
        with mock.patch('zprl.env_runner.oneleg_image_runner.OneLegEnv', return_value=env):
            metrics = runner.run(self.policy)
        self.assertEqual(self.policy.calls, 3)
        self.assertEqual([a[1, -1].item() for a in env.actions], [0, 1, 0, 1, 0])
        self.assertEqual(metrics['test/timeout_rate'], 1)
        self.assertTrue(math.isnan(metrics['test/success_mean_epi_len']))
        self.assertEqual([x['epi_len'] for x in runner.last_results], [5, 5, 5])

    def test_normal_batches_reuse_simulator_but_reset_policy_and_layout(self):
        env = FakeEnv(2, 1)
        runner = self.runner(n_rollouts=4, n_envs=2, max_steps=1)
        with mock.patch('zprl.env_runner.oneleg_image_runner.OneLegEnv', return_value=env) as factory:
            metrics = runner.run(self.policy)
        self.assertEqual(factory.call_count, 1)
        self.assertEqual(factory.call_args.kwargs['render_size'], (640, 360))
        self.assertEqual(env.reset_seeds, [10000, 10002])
        self.assertEqual(self.policy.resets, 2)
        self.assertEqual(metrics['test/n_rollouts'], 4)

    def test_default_hundred_rollouts_use_five_batches_of_twenty(self):
        env = FakeEnv(20, 1)
        runner = OneLegImageRunner(output_dir=self.directory.name, max_steps=1)
        with mock.patch('zprl.env_runner.oneleg_image_runner.OneLegEnv', return_value=env):
            metrics = runner.run(self.policy)
        self.assertEqual(env.reset_seeds, [10000, 10020, 10040, 10060, 10080])
        self.assertEqual(metrics['test/n_rollouts'], 100)
        self.assertEqual(metrics['test/timeout_count'], 100)
        self.assertEqual(len(runner.last_results), 100)
        self.assertEqual(self.policy.calls, 5)

    def test_reset_error_marks_batch_failure(self):
        env = FakeEnv(3, 3, fail_reset=True)
        runner = self.runner()
        with mock.patch('zprl.env_runner.oneleg_image_runner.OneLegEnv', return_value=env), \
                self.assertLogs(level='ERROR'):
            metrics = runner.run(self.policy)
        self.assertEqual(metrics['test/failure_rate'], 1)
        self.assertTrue(all(record['epi_len'] == 0 for record in runner.last_results))

    def test_environment_seeds_increment_without_reseeding_policy_noise(self):
        env = FakeEnv(3, 3)
        runner = self.runner(n_rollouts=6)
        with mock.patch('zprl.env_runner.oneleg_image_runner.OneLegEnv', return_value=env), \
                mock.patch('torch.manual_seed', side_effect=AssertionError('Do not seed policy noise')):
            runner.run(self.policy)
        self.assertEqual(env.reset_seeds, [10000, 10003])
        self.assertEqual([record['seed'] for record in runner.last_results], list(range(10000, 10006)))
        self.assertTrue(self.policy.training)
        self.assertTrue(next(self.policy.parameters()).requires_grad)

    def test_video_frames_stop_at_each_episode_end_and_writers_close(self):
        writers = [Writer() for _ in range(3)]
        env = FakeEnv(3, 3, successes={0: 1}, early_done={2: 2})
        runner = self.runner(record_video=True)
        with mock.patch('zprl.env_runner.oneleg_image_runner.OneLegEnv', return_value=env), \
                mock.patch.object(runner, '_open_videos', return_value=(writers, ['a', 'b', 'c'])):
            runner.run(self.policy)
        self.assertEqual([len(writer.frames) for writer in writers], [2, 4, 3])
        self.assertTrue(all(writer.closed for writer in writers))
        self.assertEqual(writers[0].frames[0].shape, (240, 640, 3))
        np.testing.assert_array_equal(writers[0].frames[0][0, 0], [255, 0, 0])
        np.testing.assert_array_equal(writers[0].frames[0][0, 320], [0, 255, 0])

    def test_policy_error_is_not_misclassified_as_simulation_failure(self):
        env = FakeEnv(3, 3)
        runner = self.runner()
        with mock.patch('zprl.env_runner.oneleg_image_runner.OneLegEnv', return_value=env), \
                mock.patch.object(self.policy, 'predict_action', side_effect=RuntimeError('model bug')):
            with self.assertRaisesRegex(RuntimeError, 'model bug'):
                runner.run(self.policy)
        self.assertTrue(self.policy.training)
        self.assertEqual(env.close_count, 1)

    def test_video_limit_spans_batches_and_only_selected_records_have_paths(self):
        env = FakeEnv(2, 1)
        runner = self.runner(n_rollouts=6, n_envs=2, max_steps=1,
            record_video=True, n_vis=3)
        groups = [[Writer(), Writer()], [Writer()]]
        with mock.patch('zprl.env_runner.oneleg_image_runner.OneLegEnv', return_value=env), \
                mock.patch.object(runner, '_open_videos', side_effect=[
                    (groups[0], ['a', 'b']), (groups[1], ['c'])]) as open_videos:
            metrics = runner.run(self.policy)
        self.assertEqual([(call.args[0], call.args[1])
            for call in open_videos.call_args_list], [(0, 2), (2, 1)])
        self.assertEqual(['video' in record for record in runner.last_results],
            [True, True, True, False, False, False])
        self.assertTrue(all(writer.closed and len(writer.frames) == 2
            for group in groups for writer in group))
        self.assertEqual(metrics['test/timeout_count'], 6)

    def test_zero_video_limit_creates_no_writers_or_video_directory(self):
        runner = self.runner(record_video=True, n_vis=0)
        with mock.patch('zprl.env_runner.oneleg_image_runner.OneLegEnv', return_value=FakeEnv(3, 3)), \
                mock.patch.object(runner, '_open_videos', side_effect=AssertionError('no video')):
            metrics = runner.run(self.policy)
        self.assertFalse((pathlib.Path(self.directory.name) / 'videos').exists())
        self.assertEqual(metrics['test/n_rollouts'], 3)

    def test_video_limit_above_rollout_count_is_clamped(self):
        runner = self.runner(record_video=True, n_vis=10)
        writers = [Writer() for _ in range(3)]
        with mock.patch('zprl.env_runner.oneleg_image_runner.OneLegEnv', return_value=FakeEnv(3, 3)), \
                mock.patch.object(runner, '_open_videos', return_value=(writers, ['a', 'b', 'c'])) as open_videos:
            runner.run(self.policy)
        self.assertEqual(open_videos.call_args.args[1], 3)

    def test_cli_help_and_limits(self):
        cli = CliRunner()
        result = cli.invoke(main, ['--help'])
        self.assertEqual(result.exit_code, 0)
        self.assertIn('--record-video', result.output)
        self.assertIn('--n-vis', result.output)
        self.assertIn('--render-size', result.output)
        self.assertIn('100', result.output)
        self.assertIn('20', result.output)
        result = cli.invoke(main, ['-c', __file__, '-o', self.directory.name,
            '--n-rollouts', '5', '--n-envs', '3'])
        self.assertEqual(result.exit_code, 2)
        self.assertIn('divisible', result.output)
        with self.assertRaisesRegex(ValueError, 'divisible'):
            self.runner(n_rollouts=5)
        with self.assertRaises(ValueError):
            self.runner(n_action_steps=17)
        with self.assertRaises(ValueError):
            self.runner(n_vis=-1)


if __name__ == '__main__':
    unittest.main()
