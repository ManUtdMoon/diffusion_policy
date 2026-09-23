import copy
import io
import pathlib
import tempfile
import unittest
from unittest.mock import patch

import dill
import hydra
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from zprl.common.reactive_replay_buffer import ReactiveNStepReplayBuffer

from zprl.policy.reactive_policy import ResiduePolicy, SumPolicy
from zprl.policy.residue_policy import SumPolicy as ActionSumPolicy
from zprl.workspace import train_online_reactive_robomimic_workspace as workspace_module


class IdentityNormalizer:
    def unnormalize(self, x):
        return x


class DummyBasePolicy(torch.nn.Module):
    obs_feature_dim = 1

    def __init__(self, n_action_steps=4):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.normalizer = {'action': IdentityNormalizer()}
        self.n_action_steps = n_action_steps
        self.plan_inputs = []

    def encode_obs(self, obs):
        assert 'completed_stage_mask' not in obs
        return obs['state'][:, -1]

    def predict_action(self, obs):
        emb = self.encode_obs(obs)
        self.plan_inputs.append(emb.clone())
        plan = emb[:, None] * 10 + torch.arange(self.n_action_steps)[None, :, None]
        return {'obs_emb': emb, 'naction': plan}


def make_obs(values, masks=None):
    values = torch.tensor(values, dtype=torch.float32)
    obs = {'state': values[:, None, None]}
    if masks is not None:
        obs['completed_stage_mask'] = torch.tensor(masks, dtype=torch.float32)[:, None]
    return obs


def make_config(overrides=()):
    OmegaConf.register_new_resolver('eval', eval, replace=True)
    config_dir = pathlib.Path(__file__).resolve().parents[1] / 'zprl' / 'config'
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        return compose(config_name='train_online_reactive_robomimic_workspace',
                       overrides=list(overrides))


class ReactivePolicyTest(unittest.TestCase):
    def make_policy(self, tr=2, subtask_dim=2):
        base = DummyBasePolicy()
        res = ResiduePolicy(obs_dim=1 + subtask_dim, action_dim=tr,
                            hidden_dim=8, num_qs=2)
        policy = SumPolicy(1.0, 1, subtask_dim, 1, 4, base, res, tr)
        return policy, base, res

    def test_cached_plan_uses_fresh_observation_and_subtask_mask(self):
        policy, base, res = self.make_policy()
        inputs = []

        def residual(actor_input, argmax=False):
            inputs.append(actor_input.clone())
            return actor_input[:, 2:3].repeat(1, 2)

        with patch.object(res, 'predict_res_naction', side_effect=residual):
            a = policy.predict_action(make_obs([1], [[0, 0]]))['action']
            b = policy.predict_action(make_obs([9], [[1, 1]]))['action']
            c = policy.predict_action(make_obs([5], [[1, 0]]))['action']
        torch.testing.assert_close(a.flatten(), torch.tensor([10., 11.]))
        torch.testing.assert_close(b.flatten(), torch.tensor([13., 14.]))
        torch.testing.assert_close(c.flatten(), torch.tensor([50., 51.]))
        torch.testing.assert_close(inputs[1], torch.tensor([[9., 1., 1., 12., 13.]]))
        self.assertEqual(len(base.plan_inputs), 2)
        policy.train()
        self.assertFalse(base.training)
        self.assertFalse(base.weight.requires_grad)

    def test_reset_discards_plan_and_supports_new_batch(self):
        policy, base, _ = self.make_policy(subtask_dim=0)
        policy.res_scale = 0
        policy.predict_action(make_obs([1]))
        policy.reset()
        action = policy.predict_action(make_obs([2, 3]))['action']
        torch.testing.assert_close(action[:, 0, 0], torch.tensor([20., 30.]))
        self.assertEqual(len(base.plan_inputs), 2)

    def test_equal_frequencies_match_existing_policy(self):
        policy, base, res = self.make_policy(tr=4, subtask_dim=0)
        old = ActionSumPolicy(1.0, 1, 0, 1, 4, copy.deepcopy(base), res)
        for value in [1, 4, 7]:
            obs = make_obs([value])
            torch.testing.assert_close(policy.predict_action(obs)['action'],
                                       old.predict_action(obs)['action'])

    def test_segment_shapes_and_exploration_mask(self):
        policy, _, _ = self.make_policy()
        base_action = torch.ones(2, 2, 1)
        result = policy.predict_train_action(
            base_action, torch.zeros(2, 3), torch.ones(2, dtype=torch.bool))
        torch.testing.assert_close(result['action'], base_action)
        self.assertEqual(result['res_naction_flat'].shape, (2, 2))
        with self.assertRaises(AssertionError):
            policy.predict_train_action(torch.zeros(2, 4, 1), torch.zeros(2, 3))
        for tr in [0, 3, 8]:
            with self.subTest(tr=tr), self.assertRaises(AssertionError):
                SumPolicy(1.0, 1, 0, 1, 4, DummyBasePolicy(),
                          ResiduePolicy(1, max(1, tr), hidden_dim=8), tr)

    def test_checkpoint_roundtrip(self):
        cfg = make_config(['res_policy.hidden_dim=8', 'res_policy.num_qs=2'])
        res = hydra.utils.instantiate(cfg.res_policy, obs_dim=3, action_dim=2)
        stream = io.BytesIO()
        torch.save({'cfg': cfg, 'res_policy': res.state_dict()}, stream, pickle_module=dill)
        stream.seek(0)
        payload = torch.load(stream, pickle_module=dill)
        loaded = hydra.utils.instantiate(payload['cfg'].res_policy, obs_dim=3, action_dim=2)
        loaded.load_state_dict(payload['res_policy'])
        obs = make_obs([1], [[1, 0]])
        a = SumPolicy(0.05, 1, 2, 1, 4, DummyBasePolicy(), res, 2)
        b = SumPolicy(0.05, 1, 2, 1, 4, DummyBasePolicy(), loaded, 2)
        for _ in range(3):
            torch.testing.assert_close(a.predict_action(obs)['action'], b.predict_action(obs)['action'])

    def test_config_tracks_segment_discount_and_task_overrides(self):
        for task, ta, tr, stages in [('square', 4, 2, 2), ('transport', 5, 1, 2),
                                     ('tool_hang', 8, 4, 1)]:
            with self.subTest(task=task):
                cfg = make_config([f'online_task={task}_image_abs', f'n_action_steps={ta}',
                                   f'n_rl_steps={tr}', 'single_gamma=0.913',
                                   'online_task.subtask.reward_mode=pbrs'])
                self.assertEqual(cfg.online_task.env_runner.n_action_steps, tr)
                self.assertEqual(cfg.res_policy.gamma, round(0.913 ** tr, 4))
                self.assertEqual(cfg.online_task.env_runner.gamma, 0.913)
                self.assertEqual(cfg.online_task.env_runner.subtask.reward_mode, 'pbrs')
                self.assertEqual(len(cfg.online_task.subtask.stages), stages)
                self.assertEqual(cfg.training.n_step, 2)


class FakeVectorEnv:
    def __init__(self):
        self.index = 0
        self.actions = []

    def obs(self, values):
        return {k: v.numpy() for k, v in make_obs(values, [[1, 0]] * 3).items()}

    def reset(self):
        return self.obs([1, 2, 3])

    def step(self, action):
        self.actions.append(action.copy())
        values, dones, timeout, terminal = [
            ([10, 20, 4], [True, True, False], 0, 7),
            ([11, 21, 30], [False, False, True], 2, 8),
        ][self.index]
        self.index += 1
        infos = [dict(task_reward=float(i == 1),
                      completed_stage_mask=np.array([[1., 0.]])) for i in range(3)]
        terminal_obs = make_obs([terminal], [[1, 1]])
        infos[timeout].update({
            'TimeLimit.truncated': True,
            'terminal_observation': {k: v[0].numpy() for k, v in terminal_obs.items()},
        })
        return self.obs(values), np.ones(3), np.array(dones), infos


class ReactiveWorkspaceTest(unittest.TestCase):
    def test_async_reset_and_timeout_replay(self):
        for bootstrap in ['truncated', 'never']:
            with self.subTest(bootstrap=bootstrap), tempfile.TemporaryDirectory() as tmp:
                base_path = pathlib.Path(tmp) / 'base.ckpt'
                base_path.touch()
                cfg = make_config([
                    f'online_task.base_ckpt={base_path}', 'online_task.abs_action=false',
                    'online_task.shape_meta.action.shape=[1]', 'training.device=cpu',
                    'online_task.n_envs=3', 'training.training_freq=6',
                    'training.num_steps=6', 'training.learning_start=6',
                    'training.log_every=6', 'training.eval_every=6',
                    'training.checkpoint_every=6', 'training.utd=1',
                    'training.batch_size=2', 'training.buffer_size=12',
                    'res_policy.hidden_dim=8', 'res_policy.num_qs=2',
                    f'training.bootstrap_at_done={bootstrap}',
                ])
                base = DummyBasePolicy()
                payload = {'cfg': OmegaConf.create({'task_name': cfg.task_name, 'policy': {}}),
                           'state_dicts': {'ema_model': base.state_dict()}}
                env = FakeVectorEnv()
                buffers = []
                instantiate = hydra.utils.instantiate

                def instantiate_policy(config, **kwargs):
                    if '_target_' not in config:
                        return base
                    return instantiate(config, **kwargs)

                def make_buffer(*args, **kwargs):
                    buffer = ReactiveNStepReplayBuffer(*args, **kwargs)
                    buffers.append(buffer)
                    return buffer

                with patch.object(workspace_module.torch, 'load', return_value=payload), \
                     patch.object(workspace_module.hydra.utils, 'instantiate', side_effect=instantiate_policy), \
                     patch.object(workspace_module.FileUtils, 'get_env_metadata_from_dataset',
                                  return_value={'env_kwargs': {}}), \
                     patch.object(workspace_module, 'AsyncVectorEnv', return_value=env), \
                     patch.object(workspace_module, 'ReactiveNStepReplayBuffer', side_effect=make_buffer), \
                     patch.object(workspace_module, 'wandb'):
                    workspace = workspace_module.TrainOnlineReactiveRobomimicWorkspace(cfg, tmp)
                    workspace.run()

                rb = buffers[0]
                self.assertEqual(workspace.global_step, 6)
                self.assertEqual(workspace.global_update, 6)
                self.assertEqual(rb.pos, 2)
                self.assertEqual(rb.n_steps, 2)
                self.assertEqual(rb.actions.shape[-1], 2)
                np.testing.assert_allclose(rb.actions[0], 0)
                np.testing.assert_allclose(rb.observations[0, :, 3:],
                                           [[10, 11], [20, 21], [30, 31]])
                np.testing.assert_allclose(rb.observations[1, :, 3:],
                                           [[100, 101], [200, 201], [32, 33]])
                if bootstrap == 'truncated':
                    np.testing.assert_allclose(rb.next_observations[0, :, 3:],
                                               [[12, 13], [200, 201], [32, 33]])
                    np.testing.assert_allclose(rb.next_observations[1, :, 3:],
                                               [[102, 103], [202, 203], [80, 81]])
                    np.testing.assert_allclose(rb.next_observations[0, 0, :3], [7, 1, 1])
                    np.testing.assert_allclose(rb.next_observations[1, 2, :3], [8, 1, 1])
                    np.testing.assert_equal(rb.timeouts[:2], [[1, 0, 0], [0, 0, 1]])
                else:
                    np.testing.assert_allclose(rb.next_observations[0, :, 3:],
                                               [[100, 101], [200, 201], [32, 33]])
                    np.testing.assert_allclose(rb.next_observations[1, :, 3:],
                                               [[102, 103], [202, 203], [300, 301]])
                    np.testing.assert_equal(rb.timeouts[:2], 0)
                np.testing.assert_equal(rb.dones[:2], [[1, 1, 0], [0, 0, 1]])

                with patch('stable_baselines3.common.buffers.np.random.randint',
                           return_value=np.array([0, 1, 2])):
                    batch = rb._get_samples(np.zeros(3, dtype=np.int64))
                np.testing.assert_allclose(batch.actions[:, 2:4],
                                           [[10, 11], [20, 21], [30, 31]])
                expected_next_base = (
                    [[12, 13], [200, 201], [80, 81]]
                    if bootstrap == 'truncated' else
                    [[100, 101], [200, 201], [300, 301]]
                )
                np.testing.assert_allclose(batch.actions[:, 4:], expected_next_base)
                expected_next_obs = (
                    [[7, 1, 1], [20, 1, 0], [8, 1, 1]]
                    if bootstrap == 'truncated' else
                    [[10, 1, 0], [20, 1, 0], [30, 1, 0]]
                )
                torch.testing.assert_close(
                    batch.next_observations,
                    torch.tensor(expected_next_obs, dtype=torch.float32))
                torch.testing.assert_close(
                    batch.rewards.flatten(), torch.tensor([1., 1., 1.9801]))
                torch.testing.assert_close(
                    batch.discounts.flatten(),
                    torch.tensor([0.9801, 0.9801, 0.9801 ** 2]))
                expected_dones = [0, 1, 0] if bootstrap == 'truncated' else [1, 1, 1]
                torch.testing.assert_close(
                    batch.dones.flatten(), torch.tensor(expected_dones, dtype=torch.float32))
                self.assertFalse(base.training)
                self.assertFalse(base.weight.requires_grad)
                torch.testing.assert_close(base.weight, torch.ones(1))


if __name__ == '__main__':
    unittest.main()
