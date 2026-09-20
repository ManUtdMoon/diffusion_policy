import json
import pathlib
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import gym
import numpy as np
import torch

from soe.collect_soe_rollouts import CollectionEnv, collect_rollouts
from soe.soe_util import SoeExplorationPolicy
from zprl.model.vib import VIBEncoder


class FakeEnv:
    metadata = {}
    action_space = gym.spaces.Box(-1., 1., shape=(1,), dtype=np.float32)
    observation_space = gym.spaces.Dict({
        'image': gym.spaces.Box(0., 1., shape=(3, 2, 2), dtype=np.float32),
        'agent_pos': gym.spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
    })

    def __init__(self, terminate_at=None, success_steps=(), goals=(), offset=0):
        self.terminate_at = terminate_at
        self.success_steps = success_steps
        self.goals = goals
        self.offset = offset
        self.closed = False
        self.seeds = []
        self.total_steps = 0

    def seed(self, seed):
        self.seeds.append(seed)

    def _obs(self):
        return {'image': np.full((3, 2, 2), self.t / 255, dtype=np.float32),
                'agent_pos': np.array([self.t + self.offset], dtype=np.float32)}

    def reset(self):
        self.t = 0
        self.actions = []
        return self._obs()

    def step(self, action):
        self.actions.append(action.copy())
        self.t += 1
        self.total_steps += 1
        info = {'success': self.t in self.success_steps}
        if self.goals:
            info.update(n_goal_achieved=self.goals[self.t-1],
                accumulated_goal_achieved=sum(self.goals[:self.t]))
        return self._obs(), 0., self.t == self.terminate_at, info

    def close(self):
        self.closed = True


class FakeVectorEnv:
    """Match AsyncVectorEnv's auto-reset behavior without subprocesses."""
    def __init__(self, env_fns, dummy_env_fn=None):
        self.envs = [fn() for fn in env_fns]

    def call_each(self, name, args_list):
        return [getattr(env, name)(*args) for env, args in zip(self.envs, args_list)]

    @staticmethod
    def stack(observations):
        return {key: np.stack([obs[key] for obs in observations]) for key in observations[0]}

    def reset(self):
        return self.stack([env.reset() for env in self.envs])

    def step(self, actions):
        observations, rewards, dones, results = [], [], [], []
        for env, action in zip(self.envs, actions):
            obs, reward, done, result = env.step(action)
            if done:
                obs = env.reset()
            observations.append(obs)
            rewards.append(reward)
            dones.append(done)
            results.append(result)
        return self.stack(observations), np.array(rewards), np.array(dones), results

    def close(self):
        for env in self.envs:
            env.close()


class FakePolicy:
    device = torch.device('cpu')
    dtype = torch.float32
    n_obs_steps = 2
    n_action_steps = 2
    num_inference_steps = 2

    def __init__(self):
        self.observations = []
        self.vib_encoder = VIBEncoder(1, 1, 4, alpha=3.)
        self.vib_decoder = torch.nn.Identity()

    def reset(self):
        pass

    def encode_obs(self, obs):
        self.observations.append(obs)
        return torch.zeros(obs['agent_pos'].shape[0], 1)

    def conditional_predict(self, condition):
        return {'action': torch.tensor([[[3.], [-3.]]]).repeat(condition.shape[0], 1, 1)}


def record_episode(raw_env, policy, task, max_steps):
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / 'episode.npz'
        env = CollectionEnv(raw_env, task, max_steps, policy.n_obs_steps, policy.n_action_steps)
        env.start_episode(0, str(path))
        obs = env.reset()
        exploration = SoeExplorationPolicy(policy)
        done = False
        while not done:
            inputs = {key: torch.from_numpy(value[None]) for key, value in obs.items()}
            actions = exploration.predict_action(inputs)['action'][0].numpy()
            obs, _, done, result = env.step(actions)
        with np.load(path) as saved:
            data = {key: saved[key] for key in saved.files}
        return data, result


class SoeCollectionTest(unittest.TestCase):
    def test_exploration_scales_posterior_std_without_changing_encoder(self):
        policy = FakePolicy()
        for p in policy.vib_encoder.parameters():
            p.data.zero_()
        policy.vib_encoder.mean_head.bias.data.fill_(4.)
        obs = {'agent_pos': torch.zeros(1, 2, 1)}
        with patch('torch.randn_like', return_value=torch.tensor([[0.5]])), \
                patch.object(policy, 'conditional_predict', side_effect=lambda z: {'action': z}):
            result = SoeExplorationPolicy(policy, 2.).predict_action(obs)
        torch.testing.assert_close(result['action'], torch.tensor([[7.]]))
        self.assertEqual(policy.vib_encoder.alpha, 3.)
        self.assertFalse(result['action'].requires_grad)

    def test_partial_chunk_history_and_pre_action_observations(self):
        env = FakeEnv(terminate_at=3, success_steps=(1,))
        policy = FakePolicy()
        data, result = record_episode(env, policy, 'metaworld_box-close', 10)
        np.testing.assert_array_equal(data['state'][:, 0], [0, 1, 2])
        np.testing.assert_array_equal(data['img'][:, 0, 0, 0], [0, 1, 2])
        np.testing.assert_array_equal(data['action'][:, 0], [1, -1, 1])
        np.testing.assert_array_equal(data['action'], env.actions)
        np.testing.assert_array_equal(data['terminated'], [False, False, True])
        self.assertEqual(data['img'].dtype, np.uint8)
        self.assertEqual(data['action'].dtype, np.float32)
        self.assertEqual(data['state'].dtype, np.float32)
        self.assertEqual(result['length'], 3)
        self.assertTrue(result['success'])
        self.assertEqual(result['termination_reason'], 'terminated')
        np.testing.assert_array_equal(policy.observations[0]['agent_pos'], [[[0], [0]]])
        np.testing.assert_array_equal(policy.observations[1]['agent_pos'], [[[1], [2]]])

    def test_hammer_counts_repeats_and_horizon_timeout(self):
        for final_count, success in [(49, False), (50, True)]:
            env = FakeEnv(goals=(20, 20, final_count-40))
            data, result = record_episode(env, FakePolicy(), 'adroit_hammer', 3)
            self.assertEqual(len(data['action']), 3)
            self.assertEqual(result['accumulated_goal_achieved'], final_count)
            self.assertEqual(result['success'], success)
            self.assertEqual(result['termination_reason'], 'timeout')
            np.testing.assert_array_equal(data['truncated'], [False, False, True])

    def test_parallel_batches_save_exact_budget_before_auto_reset(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = pathlib.Path(tmp) / 'rollouts'
            cfg = SimpleNamespace(task=SimpleNamespace(env_runner=SimpleNamespace(max_steps=3)))
            raw_envs = [FakeEnv(offset=0), FakeEnv(offset=10)]
            policy = FakePolicy()
            with patch('soe.collect_soe_rollouts.load_soe_policy', return_value=(policy, cfg, 'ema_model')), \
                    patch('soe.collect_soe_rollouts.make_env', side_effect=[(env, 3) for env in raw_envs]), \
                    patch('soe.collect_soe_rollouts.AsyncVectorEnv', side_effect=FakeVectorEnv) as vector:
                summary = collect_rollouts('test.ckpt', 'metaworld_box-close', 0, 1,
                    num_episodes=4, n_envs=2, device='cpu', output=output)
            vector.assert_called_once()
            self.assertTrue(all(env.closed for env in raw_envs))
            self.assertEqual([env.total_steps for env in raw_envs], [6, 6])
            self.assertEqual(summary['num_episodes'], 4)
            self.assertEqual(summary['failed_episodes'], 4)
            self.assertEqual(summary['env_steps'], 12)
            self.assertEqual(len(policy.observations), 4)
            self.assertTrue(all(obs['agent_pos'].shape[0] == 2 for obs in policy.observations))
            manifest = [json.loads(line) for line in (output / 'manifest.jsonl').read_text().splitlines()]
            self.assertEqual(len(list((output / 'episodes').glob('*.npz'))), 4)
            self.assertEqual(len({e['episode_uid'] for e in manifest}), 4)
            self.assertEqual([e['batch_start'] for e in manifest], [0, 0, 2, 2])
            self.assertEqual([e['batch_index'] for e in manifest], [0, 1, 0, 1])
            self.assertEqual(manifest[0]['policy_seed'], manifest[1]['policy_seed'])
            for episode in manifest:
                self.assertFalse(episode['success'])
                with np.load(output / episode['trajectory_path']) as data:
                    self.assertEqual(data['img'].shape, (3, 2, 2, 3))
                    np.testing.assert_array_equal(data['state'][:, 0],
                        np.arange(3) + 10 * episode['batch_index'])

    def test_worker_error_propagates_and_closes_environments(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = SimpleNamespace(task=SimpleNamespace(env_runner=SimpleNamespace(max_steps=3)))
            env = FakeEnv()
            with patch('soe.collect_soe_rollouts.load_soe_policy', return_value=(FakePolicy(), cfg, 'ema_model')), \
                    patch('soe.collect_soe_rollouts.make_env', return_value=(env, 3)), \
                    patch('soe.collect_soe_rollouts.AsyncVectorEnv', side_effect=FakeVectorEnv), \
                    patch.object(env, 'step', side_effect=RuntimeError('simulation error')), \
                    self.assertRaisesRegex(RuntimeError, 'simulation error'):
                collect_rollouts('test.ckpt', 'metaworld_box-close', 0, 1,
                    num_episodes=1, n_envs=1, device='cpu', output=pathlib.Path(tmp)/'error')
            self.assertTrue(env.closed)
            self.assertFalse((pathlib.Path(tmp)/'error/collection_summary.json').exists())

    def test_nondivisible_budget_fails_before_loading_policy(self):
        with self.assertRaisesRegex(AssertionError, 'divisible'):
            collect_rollouts('missing.ckpt', 'metaworld_box-close', 0, 1,
                num_episodes=3, n_envs=2)


if __name__ == '__main__':
    unittest.main()
