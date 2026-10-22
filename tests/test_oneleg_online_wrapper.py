import unittest

import torch

from zprl.env.furniturebench.oneleg_online_env import OneLegOnlineEnv


class ScriptedOneLegEnv:
    """Mutable observation buffers and repeated native rewards after success."""

    device = torch.device('cpu')

    def __init__(self, success_steps=(3, None), reward_only=False):
        self.num_envs = len(success_steps)
        self.success_steps = success_steps
        self.reward_only = reward_only
        self.reset_calls = 0
        self.close_calls = 0
        self.actions = []
        self.steps = 0
        self.last_grasp = torch.full((self.num_envs,), -1.)
        self.obs = dict(color_image1=torch.zeros(self.num_envs, 3, 2, 3),
                        color_image2=torch.zeros(self.num_envs, 3, 2, 3),
                        robot_state=torch.zeros(self.num_envs, 16))
        self.fail_at = None
        self.error = RuntimeError('native simulation failed')
        self.nonfinite = None
        self.unexpected_done = False

    def reset(self, seed=None):
        self.reset_calls += 1
        self.steps = 0
        for value in self.obs.values():
            value.zero_()
        return self.obs

    def is_success(self):
        return torch.tensor([not self.reward_only and end is not None and self.steps >= end
                             for end in self.success_steps])

    def step(self, action):
        self.actions.append(action.clone())
        self.steps += 1
        if self.steps == self.fail_at:
            raise self.error
        changed = (action[:, -1].sign() != self.last_grasp.sign()) & (action[:, -1].abs() > 0.019)
        self.last_grasp[changed] = action[changed, -1]
        for index in range(self.num_envs):
            for value in self.obs.values():
                value[index].fill_(100 * index + self.steps)
        reward = torch.tensor([[float(end is not None and self.steps >= end)]
                               for end in self.success_steps])
        done = torch.tensor([[end is not None and self.steps > end] for end in self.success_steps])
        if self.unexpected_done:
            done[0] = True
        if self.nonfinite == 'obs':
            self.obs['color_image1'][0, 0, 0, 0] = float('nan')
        elif self.nonfinite == 'reward':
            reward[0] = float('inf')
        elif self.nonfinite == 'done':
            done = done.float()
            done[0] = float('nan')
        return self.obs, reward, done, {}

    def close(self):
        self.close_calls += 1


def actions_for(env):
    indices = torch.arange(env.num_envs, device=env.device)
    action = torch.zeros(env.num_envs, env.n_action_steps, 10, device=env.device)
    action[:, :, 3] = action[:, :, 7] = 1.
    action[:, :, 0] = (indices[:, None] + 1) * 0.01
    action[:, :, -1] = torch.where(indices[:, None] % 2 == 0, 1., -1.)
    return action


class OneLegOnlineWrapperTest(unittest.TestCase):
    def make_env(self, success_steps=(3, None), **kwargs):
        native = ScriptedOneLegEnv(success_steps)
        return OneLegOnlineEnv(env=native, **kwargs), native

    def test_solid_idle_accounting_and_reset_preserves_totals(self):
        env, native = self.make_env(max_steps=16)
        self.assertTrue(env.batch_done)
        env.reset()
        self.assertFalse(env.batch_done)
        env.step(actions_for(env))
        self.assertEqual((env.solid_steps, env.idle_steps), (11, 5))
        env.step(actions_for(env))
        self.assertTrue(env.batch_done)
        self.assertEqual((env.solid_steps, env.idle_steps), (19, 13))
        self.assertEqual(env.solid_steps + env.idle_steps, native.steps * native.num_envs)
        self.assertFalse(any(hasattr(env, key) for key in ('batch_id', 'native_steps', 'native_step_attempts')))
        env.reset()
        self.assertEqual((env.solid_steps, env.idle_steps), (19, 13))
        env.close()

    def test_full_batch_keeps_rows_and_overwrites_inactive_actions(self):
        env, native = self.make_env(max_steps=16, single_gamma=0.5)
        initial = env.reset(seed=100)
        self.assertEqual(initial['robot_state'].shape, (2, 1, 16))
        torch.testing.assert_close(env._control_pose, initial['robot_state'][:, 0, :9])
        self.assertNotEqual(env._control_pose.data_ptr(), native.obs['robot_state'].data_ptr())
        with self.assertRaisesRegex(RuntimeError, 'active episodes'):
            env.reset()
        action = actions_for(env)
        original = action.clone()
        obs, reward, done, infos = env.step(action)
        self.assertEqual([info['executed_action_steps'] for info in infos], [3, 8])
        self.assertEqual([info['env_index'] for info in infos], [0, 1])
        self.assertEqual([info['seed'] for info in infos], [100, 101])
        torch.testing.assert_close(reward, torch.tensor([0.25, 0.]))
        torch.testing.assert_close(done, torch.tensor([True, False]))
        torch.testing.assert_close(env.active, torch.tensor([False, True]))
        torch.testing.assert_close(action, original, rtol=0, atol=0)
        for step, full in enumerate(native.actions[3:], start=3):
            torch.testing.assert_close(full[0, :9], torch.full((9,), 3.))
            self.assertEqual(full[0, 9], 0.)
            torch.testing.assert_close(full[1], original[1, step])
        torch.testing.assert_close(env._control_pose, obs['robot_state'][:, 0, :9])
        self.assertFalse(torch.equal(env._control_pose[0], native.obs['robot_state'][0, :9]))
        torch.testing.assert_close(native.last_grasp, torch.tensor([1., -1.]))
        self.assertEqual(obs['robot_state'][0, 0, 0], 3)
        terminal = infos[0]['terminal_observation']['robot_state'].clone()
        self.assertEqual(native.reset_calls, 1)
        self.assertEqual((env.solid_steps, env.idle_steps), (11, 5))
        second_action = actions_for(env)
        original_second = second_action.clone()
        with self.assertRaises(IndexError):
            env.step(second_action[1:])
        next_obs, reward, done, later = env.step(second_action)
        self.assertEqual(next_obs['robot_state'].shape, (2, 1, 16))
        self.assertEqual([info['env_index'] for info in later], [0, 1])
        self.assertEqual([info['executed_action_steps'] for info in later], [0, 8])
        self.assertEqual([info['transition_valid'] for info in later], [False, True])
        torch.testing.assert_close(done, torch.tensor([False, True]))
        self.assertFalse(later[0]['TimeLimit.truncated'])
        self.assertEqual(later[0]['task_reward'], 0)
        self.assertNotIn('terminal_observation', later[0])
        self.assertTrue(later[1]['TimeLimit.truncated'])
        torch.testing.assert_close(second_action, original_second, rtol=0, atol=0)
        for step, full in enumerate(native.actions[8:], start=8):
            torch.testing.assert_close(full[0, :9], torch.full((9,), 3.))
            self.assertEqual(full[0, 9], 0.)
            torch.testing.assert_close(full[1], original_second[1, step - 8])
        torch.testing.assert_close(env._control_pose, next_obs['robot_state'][:, 0, :9])
        torch.testing.assert_close(next_obs['robot_state'][0], terminal)
        self.assertTrue(env.batch_done)
        self.assertEqual(float(reward.sum()), 0)
        self.assertEqual((env.solid_steps, env.idle_steps), (19, 13))
        self.assertEqual(native.reset_calls, 1)
        with self.assertRaises(RuntimeError):
            env.step(torch.empty(0, 8, 10))
        reset_obs = env.reset(seed=200)
        torch.testing.assert_close(env._control_pose, reset_obs['robot_state'][:, 0, :9])
        self.assertEqual(native.reset_calls, 2)
        self.assertEqual((env.solid_steps, env.idle_steps), (19, 13))
        self.assertEqual(env.results, [None, None])
        torch.testing.assert_close(infos[0]['terminal_observation']['robot_state'], terminal)
        self.assertNotEqual(obs['robot_state'][0].data_ptr(),
                            infos[0]['terminal_observation']['robot_state'].data_ptr())
        env.close()
        env.close()
        self.assertEqual(native.close_calls, 1)

    def test_success_first_middle_last_and_success_beats_timeout(self):
        for end in (1, 3, 8):
            with self.subTest(end=end):
                env, native = self.make_env((end, end), max_steps=end, single_gamma=0.9)
                env.reset()
                obs, reward, done, infos = env.step(actions_for(env))
                self.assertTrue(done.all())
                self.assertEqual(native.steps, end)
                torch.testing.assert_close(reward, torch.full((2,), 0.9 ** (end - 1)))
                for info in infos:
                    self.assertEqual(info['termination_reason'], 'success')
                    self.assertFalse(info['TimeLimit.truncated'])
                    self.assertEqual(info['episode_return'], 1)
                    self.assertEqual(info['executed_action_steps'], end)
                    self.assertTrue(info['transition_valid'])

    def test_current_reward_handles_lagging_native_success_and_done(self):
        env, native = self.make_env((1, 1))
        native.reward_only = True
        env.reset()
        _, reward, done, infos = env.step(actions_for(env))
        self.assertFalse(native.is_success().any())
        self.assertTrue(done.all())
        self.assertEqual(native.steps, 1)
        self.assertTrue((reward == 1).all())
        self.assertTrue(all(info['success'] for info in infos))

    def test_700_step_boundary_last_prefix_four_without_step_701(self):
        env, native = self.make_env((None, None))
        env.reset()
        for chunk in range(88):
            _, reward, done, infos = env.step(actions_for(env))
            self.assertEqual([info['executed_action_steps'] for info in infos],
                             [8, 8] if chunk < 87 else [4, 4])
            self.assertEqual(bool(done.all()), chunk == 87)
        self.assertTrue(env.batch_done)
        self.assertEqual(native.steps, 700)
        self.assertEqual((env.solid_steps, env.idle_steps), (1400, 0))
        self.assertTrue(all(info['TimeLimit.truncated'] and not info['success'] for info in infos))
        for info in infos:
            self.assertEqual(info['episode_steps'], 700)
            self.assertIn('terminal_observation', info)

    def test_input_contract_errors_do_not_step_the_native_environment(self):
        env, native = self.make_env()
        with self.assertRaises(RuntimeError):
            env.step(torch.empty(2, 8, 10))
        env.reset()
        for action in (torch.zeros(2, 10), torch.zeros(1, 8, 10), torch.full((2, 8, 10), float('nan'))):
            with self.assertRaises(ValueError):
                env.step(action)
        self.assertEqual(native.steps, 0)

    def test_unexpected_native_done_is_a_protocol_error(self):
        env, native = self.make_env((None, None), mode='eval')
        native.unexpected_done = True
        env.reset()
        with self.assertRaisesRegex(RuntimeError, 'without success or timeout'):
            env.step(actions_for(env))
        self.assertFalse(env.last_chunk_valid)
        self.assertEqual(env.results, [None, None])

    def test_nonfinite_results_invalidate_whole_chunk(self):
        for field in ('obs', 'reward', 'done'):
            with self.subTest(field=field):
                env, native = self.make_env((None, None))
                env.reset()
                native.nonfinite = field
                with self.assertRaises(FloatingPointError):
                    env.step(actions_for(env))
                self.assertFalse(env.last_chunk_valid)
                with self.assertRaises(RuntimeError):
                    env.reset()

    def test_failed_chunk_has_no_return_and_reports_physical_steps(self):
        env, native = self.make_env((None, None), n_action_steps=2)
        env.reset()
        env.step(actions_for(env))
        native.fail_at = 4
        with self.assertRaisesRegex(RuntimeError, 'native simulation failed'):
            env.step(actions_for(env))
        self.assertFalse(env.last_chunk_valid)
        self.assertIsNotNone(env.batch_error)
        self.assertEqual((env.solid_steps, env.idle_steps), (6, 0))
        self.assertEqual(native.reset_calls, 1)
        env.close()

    def test_failed_chunk_after_success_keeps_final_hold_pose(self):
        for mode in ('train', 'eval'):
            with self.subTest(mode=mode):
                env, native = self.make_env((1, None), n_action_steps=4, mode=mode)
                env.reset()
                native.fail_at = 3
                with self.assertRaisesRegex(RuntimeError, 'native simulation failed'):
                    env.step(actions_for(env))
                torch.testing.assert_close(native.actions[1][0, :9], torch.ones(9))
                self.assertEqual(native.actions[1][0, 9], 0.)
                torch.testing.assert_close(env._control_pose[0], torch.ones(9))
                torch.testing.assert_close(env._control_pose[1], native.obs['robot_state'][1, :9])
                self.assertFalse(env.last_chunk_valid)
                self.assertEqual((env.solid_steps, env.idle_steps), (3, 1))
                self.assertEqual(native.reset_calls, 1)
                with self.assertRaises(RuntimeError):
                    env.reset()
                env.close()
                self.assertIsNone(env._control_pose)

    def test_fixed_twenty_row_sessions_and_effective_decision_fixture(self):
        for history_length in (1, 4):
            with self.subTest(history_length=history_length):
                env, native = self.make_env(tuple(range(1, 21)), n_action_steps=1)
                obs = env.reset(seed=1000)
                sessions = {index: [] for index in range(20)}
                closed = []
                total = 0
                for expected in range(20, 0, -1):
                    entry_active = env.active.clone()
                    self.assertEqual(int(entry_active.sum()), expected)
                    action = actions_for(env)
                    self.assertEqual(action.shape, (20, 1, 10))
                    obs, reward, done, infos = env.step(action)
                    self.assertEqual(obs['robot_state'].shape, (20, 1, 16))
                    self.assertEqual(len(infos), 20)
                    # Fixed policy rows stay in place; only valid transitions update sessions.
                    for row, info in enumerate(infos):
                        index = info['env_index']
                        self.assertEqual(index, row)
                        self.assertEqual(info['transition_valid'], bool(entry_active[row]))
                        if not info['transition_valid']:
                            self.assertEqual(info['executed_action_steps'], 0)
                            self.assertFalse(done[row])
                            self.assertEqual(reward[row], 0)
                            continue
                        sessions[index].append((total + expected, action[row].clone(), obs['robot_state'][row].clone()))
                        sessions[index] = sessions[index][-history_length:]
                        self.assertEqual(info['executed_action_steps'], 1)
                        if done[row]:
                            closed.append(index)
                    self.assertEqual(sum(info['transition_valid'] for info in infos), expected)
                    total += expected
                    self.assertEqual(int(env.active.sum()), expected - 1)
                self.assertEqual(closed, list(range(20)))
                self.assertEqual((total, env.solid_steps, env.idle_steps), (210, 210, 190))
                records = [dict(result) for result in env.results]
                self.assertEqual(len(records), 20)
                self.assertEqual(native.reset_calls, 1)
                env.reset(seed=2000)
                self.assertEqual(native.reset_calls, 2)


if __name__ == '__main__':
    unittest.main()
