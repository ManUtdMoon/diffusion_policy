import unittest
from unittest.mock import patch

import gymnasium
import numpy as np
import torch

from zprl.common.oneleg_episode_replay_buffer import OneLegEpisodeReplayBuffer, OneLegEpisodeCollector
from zprl.env.furniturebench.oneleg_online_env import OneLegOnlineEnv
from zprl.policy.oneleg_history_policy import OneLegHistorySumPolicy, OneLegHistoryNoiseSumPolicy
from zprl.policy.reactive_history_policy import HistoryResiduePolicy
from zprl.policy.reactive_history_noise_policy import HistoryNoisePolicy
from tests.test_oneleg_online_wrapper import ScriptedOneLegEnv


class IdentityNormalizer:
    def unnormalize(self, value):
        return value


class FakeBase(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.normalizer = {'action': IdentityNormalizer()}
        self.horizon = 16
        self.n_obs_steps = 1
        self.n_action_steps = 8
        self.obs_feature_dim = 1
        self.action_dim = 10
        self.rows = []
        self.noises = []

    def encode_obs(self, obs):
        self.rows.append(obs['robot_state'].shape[0])
        return obs['robot_state'][:, -1, :1].clone()

    def predict_action(self, obs):
        cond = self.encode_obs(obs)
        plan = cond[:, None] + torch.arange(16, device=cond.device)[None, :, None]
        return {'obs_emb': cond, 'naction': plan.expand(-1, -1, 10)[:, :self.n_action_steps]}

    def conditional_predict_from_noise(self, cond, noise):
        self.rows.append(cond.shape[0])
        self.noises.append(noise.clone())
        plan = cond[:, None] + noise
        return {'naction_pred': plan, 'action_pred': plan}


def make_session(method, history_length=5, sub_ratio=2, device='cpu'):
    base = FakeBase().to(device)
    common = dict(obs_dim=1, history_length=history_length, sub_ratio=sub_ratio,
                  hidden_dim=8, gru_input_dim=8, gru_hidden_dim=8, gru_num_layers=1,
                  num_qs=2, num_subset=2, gamma=0.9)
    if method == 'resrl':
        policy = HistoryResiduePolicy(action_dim=40, **common).to(device)
        session = OneLegHistorySumPolicy(.05, 1, 0, 10, 8, base, policy, 4)
    else:
        policy = HistoryNoisePolicy(action_dim=20, exec_dim=40, **common).to(device)
        session = OneLegHistoryNoiseSumPolicy(1.5, 1, 0, 10, 4, 16, 1, base, policy, 2)
    return session, base, policy


def make_replay(method, capacity=100, history_length=5, n_step=3):
    return OneLegEpisodeReplayBuffer(
        capacity, gymnasium.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32),
        gymnasium.spaces.Box(-np.inf, np.inf, (40 if method == 'resrl' else 20,), dtype=np.float32),
        method=method, device='cpu', history_length=history_length,
        action_step_dim=10, n_steps=n_step, gamma=.9,
        **({'base_action_dim': 40} if method == 'resrl' else {'exec_dim': 40}))


class OneLegHistoryTest(unittest.TestCase):
    def test_full_batch_sessions_match_committed_replay_and_n_step_endpoints(self):
        for method in ('resrl', 'dsrl'):
            for h, r, n_step, max_steps in ((1, 1, 1, 16), (1, 1, 3, 18), (5, 2, 3, 18)):
                with self.subTest(method=method, h=h, max_steps=max_steps):
                    session, base, policy = make_session(method, h, r)
                    replay = make_replay(method, history_length=h, n_step=n_step)
                    collector = OneLegEpisodeCollector(replay, 3)
                    native = ScriptedOneLegEnv((3, None, 9))
                    env = OneLegOnlineEnv(4, max_steps=max_steps, single_gamma=.5, env=native)
                    obs = env.reset(seed=10)
                    collector.begin_batch()
                    session.reset(obs)
                    traces = [[] for _ in range(3)]
                    completed_rows = []
                    frozen = None
                    global_step = 0
                    while not env.batch_done:
                        before = session.history.get()
                        result = session.predict_action(obs, deterministic=True)
                        self.assertEqual(result['action'].shape, (3, 4, 10))
                        obs, reward, done, infos = env.step(result['action'])
                        transition = session.on_env_step(obs, done, infos)
                        after = session.history.get()
                        for row, info in enumerate(infos):
                            if info['transition_valid']:
                                traces[row].append((tuple(x[row].clone() for x in before),
                                    tuple(x[row].clone() for x in after),
                                    {key: value[row].clone() for key, value in transition.items()},
                                    float(reward[row]), bool(done[row]), info))
                        global_step += sum(info['transition_valid'] for info in infos)
                        completed_rows.extend(done.nonzero().flatten().tolist())
                        collector.collect(transition, reward, done, infos)
                        if frozen is not None:
                            for actual, expected in zip(after, frozen):
                                torch.testing.assert_close(actual[0], expected)
                        frozen = tuple(x[0].clone() for x in after)
                        self.assertEqual(collector.pending_transitions + replay.committed_transitions,
                                         global_step)
                        if len(traces[1]) == 1:
                            self.assertEqual(replay.size(), 1)  # Only the short success is visible.
                        self.assertEqual(session.current_obs_emb.shape[0], 3)
                    self.assertTrue(collector.closed.all())
                    self.assertEqual(replay.committed_episodes, 3)
                    self.assertEqual(replay.committed_transitions, global_step)
                    self.assertTrue(all(row == 3 for row in base.rows))
                    self.assertFalse(base.training)
                    self.assertTrue(all(not p.requires_grad for p in base.parameters()))
                    storage = replay.storage
                    for row, trace in enumerate(traces):
                        indices = np.flatnonzero(storage.episode_ids[:storage.pos, 0] == completed_rows.index(row))
                        self.assertEqual(len(indices), len(trace))
                        np.testing.assert_array_equal(np.diff(indices), np.ones(len(indices) - 1))
                        for start, index in enumerate(indices):
                            batch = storage._get_samples(np.array([index]))
                            end = min(start + n_step, len(trace)) - 1
                            for actual, wanted in zip((batch.obs, batch.a_exec, batch.obs_valid, batch.exec_steps), trace[start][0]):
                                torch.testing.assert_close(actual[0], wanted)
                            for actual, wanted in zip((batch.next_obs, batch.next_a_exec, batch.next_obs_valid, batch.next_exec_steps), trace[end][1]):
                                torch.testing.assert_close(actual[0], wanted)
                            expected_reward = sum(.9 ** j * trace[k][3] for j, k in enumerate(range(start, end + 1)))
                            self.assertAlmostEqual(batch.rewards.item(), expected_reward, places=6)
                            self.assertAlmostEqual(batch.discounts.item(), .9 ** (end - start + 1), places=6)
                            expected_done = trace[end][4] and not trace[end][5].get('TimeLimit.truncated', False)
                            self.assertEqual(batch.dones.item(), float(expected_done))
                            torch.testing.assert_close(batch.action[0], trace[start][2]['action'])
                            if method == 'resrl':
                                torch.testing.assert_close(batch.next_a_base[0], trace[end][2]['next_a_base'])
                    # The success tail survives in terminal history (H>1).
                    if h > 1:
                        torch.testing.assert_close(traces[0][0][1][1][-2], traces[0][0][2]['a_exec'])
                        self.assertTrue(traces[0][0][2]['a_exec'].reshape(4, 10)[3:].abs().sum() > 0)
                    old_count = replay.committed_transitions
                    obs = env.reset(seed=20)
                    collector.begin_batch()
                    session.reset(obs)
                    self.assertEqual(replay.committed_transitions, old_count)
                    self.assertEqual(session.history.get()[2].sum().item(), 3)
                    env.close()

    def test_resrl_timeout_keeps_old_plan_mid_segment_and_decodes_at_boundary(self):
        for max_steps in (6, 8):
            session, base, _ = make_session('resrl')
            env = OneLegOnlineEnv(4, max_steps=max_steps, env=ScriptedOneLegEnv((None, None)))
            obs = env.reset()
            session.reset(obs)
            for _ in range(2):
                action = session.predict_action(obs)['action']
                obs, _, done, infos = env.step(action)
                transition = session.on_env_step(obs, done, infos)
            # Second chunk ends a plan boundary regardless of actual L=2 or L=4.
            expected = base.predict_action(obs)['naction'][:, :4]
            torch.testing.assert_close(transition['next_a_base'], expected.flatten(1))
            env.close()
        # A timeout during the first segment must not decode a replacement plan.
        session, base, _ = make_session('resrl')
        env = OneLegOnlineEnv(4, max_steps=2, env=ScriptedOneLegEnv((None, None)))
        obs = env.reset()
        session.reset(obs)
        cache = session.base_cache.clone()
        action = session.predict_action(obs)['action']
        obs, _, done, infos = env.step(action)
        transition = session.on_env_step(obs, done, infos)
        torch.testing.assert_close(transition['next_a_base'], cache[:, 4:8].flatten(1))
        env.close()

    def test_noise_repeat_scale_and_warmup_remain_full_batch(self):
        session, base, policy = make_session('dsrl')
        env = OneLegOnlineEnv(4, max_steps=8, env=ScriptedOneLegEnv((1, None)))
        obs = env.reset()
        session.reset(obs)
        with patch.object(policy, 'predict_noise', side_effect=AssertionError('warmup actor')):
            result = session.predict_action(obs, global_step=0, learning_start=10)
        torch.testing.assert_close(base.noises[-1], result['nnoise'].reshape(2, 2, 10).repeat(1, 8, 1) * 1.5)
        obs, _, done, infos = env.step(result['action'])
        session.on_env_step(obs, done, infos)
        with patch.object(policy, 'predict_noise', wraps=policy.predict_noise) as actor:
            session.predict_action(obs, global_step=10, learning_start=10)
        self.assertEqual(actor.call_args.args[0].shape[0], 2)
        self.assertFalse(actor.call_args.kwargs['argmax'])
        env.close()


if __name__ == '__main__':
    unittest.main()
