import unittest
from unittest.mock import patch

import numpy as np
import torch

from zprl.env_runner.robomimic_image_runner import RobomimicImageRunner


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


if __name__ == '__main__':
    unittest.main()
