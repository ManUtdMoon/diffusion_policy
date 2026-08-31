import json
import pathlib
import tempfile
import unittest

import gym
import h5py
import numpy as np
from click.testing import CliRunner

from eval_smoothness import (
    TRACE_ACTION_POS_KEY,
    TRACE_CHECKSUM_KEY,
    TRACE_STATE_POS_KEY,
    PrimitiveTraceWrapper,
    TrajectoryRecord,
    TraceMultiStepWrapper,
    TranslationEMAFilterWrapper,
    _append_chunk_traces,
    aggregate_statistics,
    compute_trajectory_derivatives,
    load_eval_policy,
    main,
    write_outputs,
)


class PositionEnv(gym.Env):
    metadata = {}

    def __init__(self, terminate_at=None, action_dim=4):
        self.terminate_at = terminate_at
        self.step_count = 0
        self.position = np.zeros(3, dtype=np.float32)
        self.received_actions = []
        self.observation_space = gym.spaces.Dict({
            'robot0_eef_pos': gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
        })
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)

    def get_state(self):
        return {'states': np.concatenate([
            self.position,
            np.array([self.step_count], dtype=np.float32),
        ])}

    def reset(self):
        self.step_count = 0
        self.position = np.zeros(3, dtype=np.float32)
        self.received_actions = []
        return {'robot0_eef_pos': self.position.copy()}

    def step(self, action):
        self.received_actions.append(np.asarray(action).copy())
        self.step_count += 1
        self.position += np.asarray(action[:3], dtype=np.float32)
        done = self.step_count == self.terminate_at
        obs = {'robot0_eef_pos': self.position.copy()}
        return obs, float(done), done, {}


class EvalSmoothnessTest(unittest.TestCase):
    @staticmethod
    def _trace_info(start, length=4):
        action_pos = np.stack([
            np.arange(start, start + length, dtype=np.float32),
            np.zeros(length, dtype=np.float32),
            np.zeros(length, dtype=np.float32),
        ], axis=-1)
        state_pos = action_pos.copy()
        return {
            TRACE_ACTION_POS_KEY: action_pos,
            TRACE_STATE_POS_KEY: state_pos,
            TRACE_CHECKSUM_KEY: 'checksum',
        }

    @staticmethod
    def _record(seed, action_x, state_x, success=True):
        action_pos = np.zeros((len(action_x), 3), dtype=np.float64)
        state_pos = np.zeros((len(state_x), 3), dtype=np.float64)
        action_pos[:, 0] = action_x
        state_pos[:, 0] = state_x
        return TrajectoryRecord(
            seed=seed,
            success=success,
            kept_chunks=1,
            primitive_step_count=len(action_x),
            initial_state_checksum='checksum',
            action_pos=action_pos if success else np.empty((0, 3)),
            state_pos=state_pos if success else np.empty((0, 3)),
        )

    def test_cli_exposes_explicit_policy_type(self):
        result = CliRunner().invoke(main, ['--help'])

        self.assertEqual(result.exit_code, 0)
        self.assertIn('--policy-type [base|zprl|resrl]', result.output)
        self.assertIn('--checkpoint', result.output)
        self.assertIn('--ema-weight', result.output)

    def test_loader_rejects_unknown_policy_type_before_loading(self):
        with self.assertRaisesRegex(ValueError, 'Unknown policy type'):
            load_eval_policy('unknown', '/missing.ckpt', 'cpu')

    def test_primitive_trace_records_action_and_state_positions(self):
        env = PrimitiveTraceWrapper(PositionEnv())
        env.reset()
        env.start_chunk()
        env.step(np.array([1.0, 0.0, 0.0, 0.5], dtype=np.float32))
        env.step(np.array([0.0, 2.0, 0.0, -0.5], dtype=np.float32))

        trace = env.finish_chunk()

        np.testing.assert_array_equal(
            trace[TRACE_ACTION_POS_KEY],
            [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
        )
        np.testing.assert_array_equal(
            trace[TRACE_STATE_POS_KEY],
            [[1.0, 0.0, 0.0], [1.0, 2.0, 0.0]],
        )
        self.assertEqual(len(trace[TRACE_CHECKSUM_KEY]), 64)

    def test_multistep_trace_keeps_full_chunk_when_obs_history_is_short(self):
        env = TraceMultiStepWrapper(
            PrimitiveTraceWrapper(PositionEnv(terminate_at=2)),
            n_obs_steps=1,
            n_action_steps=3,
            max_episode_steps=10,
        )
        env.reset()
        action = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ], dtype=np.float32)

        _, _, done, info = env.step(action)

        self.assertTrue(done)
        self.assertEqual(info[TRACE_ACTION_POS_KEY].shape, (2, 3))
        self.assertEqual(info[TRACE_STATE_POS_KEY].shape, (2, 3))
        np.testing.assert_array_equal(
            info[TRACE_STATE_POS_KEY][-1], [1.0, 1.0, 0.0])

    def test_reset_state_and_chunk_traces_have_no_duplicate_boundaries(self):
        env = TraceMultiStepWrapper(
            PrimitiveTraceWrapper(PositionEnv()),
            n_obs_steps=1,
            n_action_steps=2,
            max_episode_steps=10,
        )
        reset_obs = env.reset()
        action = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ], dtype=np.float32)

        _, _, _, first_info = env.step(action)
        _, _, _, second_info = env.step(action)
        action_pos = np.concatenate([
            first_info[TRACE_ACTION_POS_KEY],
            second_info[TRACE_ACTION_POS_KEY],
        ])
        state_pos = np.concatenate([
            reset_obs['robot0_eef_pos'][-1:],
            first_info[TRACE_STATE_POS_KEY],
            second_info[TRACE_STATE_POS_KEY],
        ])

        self.assertEqual(action_pos.shape, (4, 3))
        self.assertEqual(state_pos.shape, (5, 3))
        np.testing.assert_array_equal(
            state_pos[:, 0], [0.0, 1.0, 2.0, 3.0, 4.0])

    def test_translation_ema_uses_previous_chunk_and_is_traced(self):
        base_env = PositionEnv(action_dim=10)
        env = TraceMultiStepWrapper(
            TranslationEMAFilterWrapper(
                PrimitiveTraceWrapper(base_env),
                weight=0.5,
                window_size=4,
            ),
            n_obs_steps=1,
            n_action_steps=4,
            max_episode_steps=20,
        )
        env.reset()
        first_chunk = np.zeros((4, 10), dtype=np.float32)
        first_chunk[:, 0] = [0.0, 1.0, 2.0, 3.0]
        first_chunk[:, 3:] = np.arange(28, dtype=np.float32).reshape(4, 7)
        second_chunk = np.zeros((4, 10), dtype=np.float32)
        second_chunk[:, 0] = [4.0, 5.0, 6.0, 7.0]
        second_chunk[:, 3:] = np.arange(
            28, 56, dtype=np.float32).reshape(4, 7)

        env.step(first_chunk)
        _, _, _, info = env.step(second_chunk)

        expected = np.average(
            [1.0, 2.0, 3.0, 4.0], weights=[0.125, 0.25, 0.5, 1.0])
        self.assertAlmostEqual(info[TRACE_ACTION_POS_KEY][0, 0], expected)
        self.assertEqual(info[TRACE_ACTION_POS_KEY].shape, (4, 3))
        np.testing.assert_array_equal(
            np.asarray(base_env.received_actions)[4:, 3:],
            second_chunk[:, 3:],
        )

    def test_zero_ema_weight_is_disabled(self):
        env = TraceMultiStepWrapper(
            TranslationEMAFilterWrapper(
                PrimitiveTraceWrapper(PositionEnv()),
                weight=0.0,
                window_size=2,
            ),
            n_obs_steps=1,
            n_action_steps=2,
            max_episode_steps=10,
        )
        env.reset()
        action = np.array([
            [1.0, 0.0, 0.0, -1.0],
            [3.0, 0.0, 0.0, 1.0],
        ], dtype=np.float32)

        _, _, _, info = env.step(action)

        np.testing.assert_array_equal(
            info[TRACE_ACTION_POS_KEY], action[:, :3])

    def test_success_chunk_is_kept_and_later_chunks_are_ignored(self):
        buffers = [{
            'seed': 100000,
            'recording': True,
            'success': False,
            'kept_chunks': 0,
            'action_pos': [],
            'state_pos': [np.zeros((1, 3), dtype=np.float32)],
            'initial_state_checksum': None,
        }]

        _append_chunk_traces(
            buffers, [0.0], [False], [self._trace_info(1)])
        _append_chunk_traces(
            buffers, [1.0], [False], [self._trace_info(5)])
        _append_chunk_traces(
            buffers, [0.0], [True], [self._trace_info(9)])
        buffer = buffers[0]
        action_pos = np.concatenate(buffer['action_pos'])
        state_pos = np.concatenate(buffer['state_pos'])

        self.assertTrue(buffer['success'])
        self.assertEqual(buffer['kept_chunks'], 2)
        self.assertEqual(action_pos.shape, (8, 3))
        self.assertEqual(state_pos.shape, (9, 3))
        self.assertEqual(action_pos[-1, 0], 8.0)

    def test_derivatives_and_aggregation_do_not_cross_trajectories(self):
        first = self._record(
            1, action_x=[0.0, 1.0, 2.0], state_x=[0.0, 1.0, 2.0, 3.0])
        second = self._record(
            2, action_x=[100.0, 102.0, 104.0],
            state_x=[100.0, 102.0, 104.0, 106.0])
        failed = self._record(
            3, action_x=[], state_x=[], success=False)

        derivatives = compute_trajectory_derivatives(first, dt=1.0)
        np.testing.assert_array_equal(
            derivatives['action_velocity'][:, 0], [1.0, 1.0])
        np.testing.assert_array_equal(
            derivatives['action_acceleration'][:, 0], [0.0])

        jerky = self._record(
            4,
            action_x=[0.0, 1.0, 4.0, 10.0, 20.0],
            state_x=[0.0, 1.0, 4.0, 10.0, 20.0, 35.0],
        )
        jerk_summary = aggregate_statistics([jerky], dt=1.0)
        action_jerk = jerk_summary['metrics']['action_jerk']
        np.testing.assert_array_equal(
            compute_trajectory_derivatives(jerky, 1.0)[
                'action_jerk'][:, 0],
            [1.0, 1.0],
        )
        self.assertEqual(action_jerk['pooled_sample']['norm']['mean'], 1.0)
        self.assertEqual(action_jerk['pooled_sample']['norm']['std'], 0.0)
        state_jerk = jerk_summary['metrics']['state_jerk']
        self.assertEqual(state_jerk['pooled_sample']['norm']['mean'], 1.0)
        self.assertEqual(state_jerk['pooled_sample']['norm']['std'], 0.0)

        summary = aggregate_statistics([first, second, failed], dt=1.0)
        action_velocity = summary['metrics']['action_velocity']
        self.assertEqual(summary['n_requested'], 3)
        self.assertEqual(summary['n_success'], 2)
        self.assertAlmostEqual(summary['success_rate'], 2 / 3)
        self.assertEqual(
            action_velocity['pooled_sample']['sample_count'], 4)
        self.assertAlmostEqual(
            action_velocity['pooled_sample']['norm']['mean'], 1.5)
        self.assertEqual(
            action_velocity['trajectory_level']['trajectory_count'], 2)
        self.assertAlmostEqual(
            action_velocity['trajectory_level'][
                'mean_of_trajectory_means'], 1.5)
        self.assertAlmostEqual(
            action_velocity['trajectory_level'][
                'std_of_trajectory_means'], np.sqrt(0.5))
        self.assertEqual(
            summary['metrics']['state_velocity'][
                'pooled_sample']['sample_count'], 6)

    def test_output_files_round_trip(self):
        successful = self._record(
            11, action_x=[0.0, 1.0, 2.0], state_x=[0.0, 1.0, 2.0, 3.0])
        failed = self._record(
            12, action_x=[], state_x=[], success=False)
        records = [successful, failed]
        summary = aggregate_statistics(records, dt=1.0)
        metadata = {'policy_type': 'resrl', 'seeds': [11, 12]}

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = pathlib.Path(tmpdir)
            write_outputs(output_dir, metadata, summary, records, dt=1.0)

            with output_dir.joinpath('summary.json').open() as f:
                saved_summary = json.load(f)
            self.assertEqual(saved_summary['n_success'], 1)
            with h5py.File(
                    output_dir.joinpath('trajectories.hdf5'), 'r') as f:
                self.assertEqual(list(f.keys()), ['seed_11'])
                self.assertEqual(f['seed_11/action_pos'].shape, (3, 3))
                self.assertEqual(f['seed_11/state_pos'].shape, (4, 3))


if __name__ == '__main__':
    unittest.main()
