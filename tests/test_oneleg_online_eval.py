import unittest

import torch

from tests.test_oneleg_online_wrapper import ScriptedOneLegEnv, actions_for
from zprl.env.furniturebench.oneleg_online_env import OneLegOnlineEnv


class OneLegOnlineEvalTest(unittest.TestCase):
    def test_train_eval_share_normal_success_timeout_and_batch_lifecycle(self):
        outputs = []
        for mode in ('train', 'eval'):
            native = ScriptedOneLegEnv((3, None))
            env = OneLegOnlineEnv(env=native, max_steps=12, mode=mode)
            env.reset(seed=100)
            chunks = []
            while not env.batch_done:
                obs, reward, done, infos = env.step(actions_for(env))
                chunks.append((obs, reward, done, infos))
            self.assertEqual([row['termination_reason'] for row in env.results], ['success', 'timeout'])
            self.assertEqual([row['episode_steps'] for row in env.results], [3, 12])
            self.assertEqual(native.reset_calls, 1)
            self.assertEqual(sum(info['transition_valid'] for chunk in chunks for info in chunk[3]), 3)
            outputs.append((chunks, env.results))
        for left, right in zip(outputs[0][0], outputs[1][0]):
            for key in left[0]:
                torch.testing.assert_close(left[0][key], right[0][key])
            torch.testing.assert_close(left[1], right[1])
            torch.testing.assert_close(left[2], right[2])
        self.assertEqual(outputs[0][1], outputs[1][1])

    def test_batch_failure_preserves_success_and_emits_no_fake_transitions(self):
        native = ScriptedOneLegEnv((1, None, None))
        native.fail_at = 3
        env = OneLegOnlineEnv(env=native, mode='eval')
        env.reset()
        with self.assertRaisesRegex(RuntimeError, 'native simulation failed'):
            env.step(actions_for(env))
        self.assertEqual([row['termination_reason'] for row in env.results], ['success', 'failure', 'failure'])
        self.assertEqual([row['episode_steps'] for row in env.results], [1, 2, 2])
        self.assertEqual(env.results[0]['episode_return'], 1)
        self.assertFalse(env.last_chunk_valid)
        self.assertTrue(env.batch_done)
        self.assertEqual(native.reset_calls, 1)
        with self.assertRaises(RuntimeError):
            env.reset()
        env.close()
        self.assertEqual(native.close_calls, 1)

    def test_eval_retains_previously_returned_success_across_later_batch_error(self):
        native = ScriptedOneLegEnv((1, None))
        env = OneLegOnlineEnv(env=native, n_action_steps=2, mode='eval')
        env.reset()
        env.step(actions_for(env))
        confirmed = dict(env.results[0])
        native.fail_at = 3
        with self.assertRaises(RuntimeError):
            env.step(actions_for(env))
        self.assertEqual(env.results[0], confirmed)
        self.assertEqual(env.results[1]['termination_reason'], 'failure')

    def test_program_errors_are_not_converted_to_batch_failures(self):
        for error in (ValueError('program error'), TypeError('program error')):
            with self.subTest(error=type(error)):
                native = ScriptedOneLegEnv((None, None))
                native.fail_at = 1
                native.error = error
                env = OneLegOnlineEnv(env=native, mode='eval')
                env.reset()
                with self.assertRaises(type(error)) as raised:
                    env.step(actions_for(env))
                self.assertIs(raised.exception, error)
                self.assertEqual(env.results, [None, None])
                self.assertIsNone(env.batch_error)
                self.assertFalse(env.last_chunk_valid)


if __name__ == '__main__':
    unittest.main()
