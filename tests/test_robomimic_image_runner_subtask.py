import unittest
from unittest.mock import patch

import numpy as np
import torch

from zprl.env_runner.robomimic_image_runner import RobomimicImageRunner
from tests.test_reactive_history_rollout import make_obs, make_policy


class DummyProgress:
    def update(self, n):
        pass

    def close(self):
        pass


class FakeVectorEnv:
    def __init__(self):
        self.chunk = -1

    def call_each(self, name, args_list):
        self.chunk += 1

    def reset(self):
        return {'state': np.zeros((2, 1, 1), dtype=np.float32)}

    def step(self, action):
        masks = [
            ([[1.0, 0.0]], [[1.0, 1.0]]),
            ([[0.0, 0.0]], [[1.0, 1.0]]),
        ][self.chunk]
        info = [
            {
                'task_reward': float(i == 1),
                'completed_stage_mask': np.asarray(mask, dtype=np.float32),
            }
            for i, mask in enumerate(masks)
        ]
        done = np.ones(2, dtype=np.bool_)
        return self.reset(), np.zeros(2), done, info

    def render(self):
        return [None, None]


class FakePolicy:
    device = torch.device('cpu')
    dtype = torch.float32

    def reset(self):
        pass

    def predict_action(self, obs):
        batch_size = obs['state'].shape[0]
        return {'action': torch.zeros(batch_size, 1, 1)}


class HistoryVectorEnv(FakeVectorEnv):
    def call_each(self, name, args_list):
        super().call_each(name, args_list)
        self.index = 0

    def reset(self):
        return make_obs([1 + 10 * self.chunk, 2 + 10 * self.chunk])

    def step(self, action):
        dones = [[True, False], [False, True], [True, True]][self.index]
        self.index += 1
        infos = [{'task_reward': 0., 'completed_stage_mask': np.array([[0.]]),
                  'executed_action_steps': 1 if done else 2} for done in dones]
        if self.index == 1:
            infos[0].update({'TimeLimit.truncated': True,
                             'terminal_observation': {key: value[0] for key, value in
                                                      make_obs([99]).items()}})
        return make_obs([3 + self.index, 4 + self.index]), np.zeros(2), np.array(dones), infos


class RobomimicImageRunnerSubtaskTest(unittest.TestCase):
    @patch('zprl.env_runner.robomimic_image_runner.tqdm.tqdm',
           return_value=DummyProgress())
    def test_completion_rates_across_chunks_and_prefixes(self, _):
        runner = RobomimicImageRunner.__new__(RobomimicImageRunner)
        runner.env = FakeVectorEnv()
        runner.env_fns = [None, None]
        runner.env_init_fn_dills = [b'0', b'1', b'2']
        runner.env_seeds = [0, 1, 2]
        runner.env_prefixs = ['train/', 'train/', 'test/']
        runner.env_meta = {'env_name': 'Square'}
        runner.n_obs_steps = 1
        runner.n_action_steps = 1
        runner.past_action = False
        runner.max_steps = 1
        runner.action_pose_repr = 'delta'
        runner.subtask_stages = ('grasp', 'hover')
        runner.tqdm_interval_sec = 0

        result = runner.run(FakePolicy())

        self.assertEqual(result['train/grasp_completion_rate'], 1.0)
        self.assertEqual(result['train/hover_completion_rate'], 0.5)
        self.assertEqual(result['test/grasp_completion_rate'], 0.0)
        self.assertEqual(result['test/hover_completion_rate'], 0.0)

    @patch('zprl.env_runner.robomimic_image_runner.tqdm.tqdm',
           return_value=DummyProgress())
    def test_history_reset_and_feedback_across_chunks_and_partial_done(self, _):
        runner = RobomimicImageRunner.__new__(RobomimicImageRunner)
        runner.env = HistoryVectorEnv()
        runner.env_fns = [None, None]
        runner.env_init_fn_dills = [b'0', b'1', b'2']
        runner.env_seeds = [0, 1, 2]
        runner.env_prefixs = ['train/', 'train/', 'test/']
        runner.env_meta = {'env_name': 'Square'}
        runner.n_obs_steps = 1
        runner.n_action_steps = 2
        runner.past_action = False
        runner.max_steps = 6
        runner.action_pose_repr = 'delta'
        runner.subtask_stages = ('grasp',)
        runner.tqdm_interval_sec = 0
        policy, base = make_policy()
        windows, transitions = [], []
        feedback = policy.on_env_step

        def on_env_step(*args):
            transition = feedback(*args)
            windows.append(policy.history.get())
            transitions.append(transition)

        with patch.object(policy, 'reset', wraps=policy.reset) as reset, \
             patch.object(policy, 'on_env_step', side_effect=on_env_step) as step, \
             patch.object(policy, 'predict_action', wraps=policy.predict_action) as predict, \
             patch('zprl.env_runner.robomimic_image_runner.torch.from_numpy',
                   wraps=torch.from_numpy) as transfer:
            runner.run(policy)
        self.assertEqual(reset.call_count, 2)
        self.assertEqual(step.call_count, 6)
        self.assertEqual(transfer.call_count, 16)
        for chunk in range(2):
            self.assertIs(predict.call_args_list[chunk * 3].args[0],
                          reset.call_args_list[chunk].args[0])
            for index in range(1, 3):
                self.assertIs(predict.call_args_list[chunk * 3 + index].args[0],
                              step.call_args_list[chunk * 3 + index - 1].args[0])
        torch.testing.assert_close(reset.call_args_list[0].args[0]['state'][:, -1, 0],
                                   torch.tensor([1., 2.]))
        torch.testing.assert_close(reset.call_args_list[1].args[0]['state'][:, -1, 0],
                                   torch.tensor([11., 12.]))
        torch.testing.assert_close(windows[0][2].sum(dim=1), torch.tensor([1, 2]))
        torch.testing.assert_close(transitions[0]['next_obs'][0], torch.tensor([99., 0.]))
        torch.testing.assert_close(windows[0][0][0, -1], torch.tensor([4., 0.]))
        self.assertTrue(step.call_args_list[-1].args[1].all())
        self.assertIsNone(policy.pending)


if __name__ == '__main__':
    unittest.main()
