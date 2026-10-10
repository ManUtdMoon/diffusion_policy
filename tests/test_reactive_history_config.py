import copy
import json
import pathlib
import re
import tempfile
import types
import unittest
from unittest.mock import patch

import dill
import hydra
import numpy as np
import torch
from click.testing import CliRunner
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from tests.test_reactive_policy import DummyBasePolicy as BasePolicy, FakeVectorEnv
from zprl.common.reactive_history_replay_buffer import ReactiveHistoryReplayBuffer
from zprl.policy.reactive_history_policy import HistoryResiduePolicy, HistorySumPolicy
from zprl.policy.reactive_policy import SumPolicy as ReactiveSumPolicy
from zprl.policy.residue_policy import SumPolicy as ActionSumPolicy
from zprl.workspace import train_online_reactive_history_robomimic_workspace as workspace_module


ROOT = pathlib.Path(__file__).resolve().parents[1]


class AffineNormalizer:
    def unnormalize(self, action):
        return action * 3 + 5


class DummyBasePolicy(BasePolicy):
    def __init__(self, n_action_steps=4, num_inference_steps=2):
        super().__init__(n_action_steps)
        self.normalizer['action'] = AffineNormalizer()


class HistoryVectorEnv(FakeVectorEnv):
    def reset(self):
        obs = super().reset()
        obs.pop('completed_stage_mask')
        return obs

    def step(self, action):
        obs, reward, done, infos = super().step(action)
        obs.pop('completed_stage_mask')
        for info, ended in zip(infos, done):
            info['executed_action_steps'] = 1 if ended else 2
            if 'terminal_observation' in info:
                info['terminal_observation'].pop('completed_stage_mask')
        return obs, reward, done, infos


def make_config(overrides=(), history=True):
    OmegaConf.register_new_resolver('eval', eval, replace=True)
    name = 'train_online_reactive_history_robomimic_workspace' if history \
        else 'train_online_reactive_robomimic_workspace'
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / 'zprl/config')):
        return compose(config_name=name, overrides=[
            'online_task.subtask.enabled=false', 'online_task.subtask.reward_mode=sparse',
        ] + list(overrides))


def small_config(base_path, bootstrap='truncated'):
    return make_config([
        f'online_task.base_ckpt={base_path}', 'online_task.abs_action=false',
        'online_task.shape_meta.action.shape=[1]', 'training.device=cpu',
        'online_task.n_envs=3', 'n_rl_steps=2', 'training.training_freq=3',
        'training.num_steps=6', 'training.learning_start=6',
        'training.compile=false',
        'training.log_every=3', 'training.eval_every=6',
        'training.checkpoint_every=6', 'training.utd=1',
        'training.batch_size=2', 'training.buffer_size=12',
        'res_policy.hidden_dim=8', 'res_policy.num_qs=2',
        'res_policy.history_length=4',
        f'training.bootstrap_at_done={bootstrap}',
    ])


def save_base(path, cfg):
    base_cfg = OmegaConf.create({
        'task_name': cfg.task_name,
        'policy': {'_target_': 'tests.test_reactive_history_config.DummyBasePolicy',
                   'n_action_steps': 4},
        'task': {'env_runner': {'n_action_steps': 4}, 'dataset': {'pad_after': 3}},
    })
    torch.save({'cfg': base_cfg, 'state_dicts': {'ema_model': DummyBasePolicy().state_dict()}},
               path, pickle_module=dill)


def workspace_with_eval_enabled():
    # Execute the commented sections in memory; the production file stays commented.
    source = pathlib.Path(workspace_module.__file__).read_text()
    blocks = [
        ('        ## eval, only average score needed\n', '        ## train\n'),
        ('        self.res_policy.to(device)\n', '        optimizers = '),
        ('        with JsonLogger(log_path) as logger:\n', '            while self.global_step'),
        ('                # evaluation\n', '                sum_policy.train()'),
        ('                # checkpointing\n', '\n\n@hydra.main'),
    ]
    for start, end in blocks:
        first = source.index(start) + len(start)
        last = source.index(end, first)
        source = source[:first] + re.sub(r'^(\s*)# (.*)$', r'\1\2',
                                         source[first:last], flags=re.MULTILINE) + source[last:]
    module = types.ModuleType('history_workspace_eval_test')
    module.__file__ = workspace_module.__file__
    exec(compile(source, module.__file__, 'exec'), module.__dict__)
    return module


class FakeEvalRunner:
    def __init__(self):
        self.policies = []
        self.actions = []

    def run(self, policy):
        self.policies.append(policy)
        obs = {'state': np.ones((2, 1, 1), dtype=np.float32)}
        if hasattr(policy, 'on_env_step'):
            policy.reset(obs)
        else:
            policy.reset()
        result = policy.predict_action({key: torch.from_numpy(value) for key, value in obs.items()})
        self.actions.append(result['action'].clone())
        if hasattr(policy, 'on_env_step'):
            policy.on_env_step(obs, [True, True], [{'executed_action_steps': 2}] * 2)
        return {'test/mean_score': 1.0, 'test/grasp_completed': np.float32(0.5)}


def load_eval_module():
    source = (ROOT / 'eval_sum.py').read_text()
    # Keep CLI capture streams intact when executing the script in the test process.
    source = source.replace("sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)\n", '')
    source = source.replace("sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)\n", '')
    module = types.ModuleType('eval_sum_test')
    module.__file__ = str(ROOT / 'eval_sum.py')
    with patch('os.path.exists', return_value=True):
        exec(compile(source, module.__file__, 'exec'), module.__dict__)
    return module


class ReactiveHistoryConfigTest(unittest.TestCase):
    def test_defaults_and_old_config_are_independent(self):
        cfg = make_config()
        self.assertEqual(hydra.utils.get_class(cfg._target_),
                         workspace_module.TrainOnlineReactiveHistoryRobomimicWorkspace)
        self.assertEqual(cfg.res_policy.history_length, 2)
        self.assertEqual(cfg.res_policy.critic_action_input, 'base_res')
        self.assertNotIn('network_type', cfg.res_policy)
        self.assertNotIn('d_model', cfg.res_policy)
        self.assertEqual(cfg.training.n_step, 2)
        self.assertTrue(cfg.training.compile)
        self.assertEqual(cfg.n_rl_steps, 2)
        self.assertFalse(cfg.online_task.subtask.enabled)
        self.assertEqual(cfg.online_task.subtask.reward_mode, 'sparse')
        self.assertFalse(cfg.logging.resume)
        self.assertIsNone(cfg.logging.id)
        old = make_config(history=False)
        self.assertEqual(old.training.n_step, 2)
        self.assertTrue(old.logging.resume)
        self.assertNotIn('history', old)
        self.assertNotIn('compile', old.training)

    def test_history_and_segment_overrides_reach_policy_and_runner(self):
        for task, ta, tr in [('square', 4, 2), ('transport', 5, 1), ('tool_hang', 8, 4)]:
            with self.subTest(task=task):
                cfg = make_config([f'online_task={task}_image_abs', f'n_action_steps={ta}',
                                   f'n_rl_steps={tr}', 'single_gamma=0.913',
                                   'res_policy.history_length=3', 'res_policy.hidden_dim=8',
                                   'res_policy.num_qs=2'])
                policy = hydra.utils.instantiate(cfg.res_policy, obs_dim=1, action_dim=tr)
                self.assertIsInstance(policy, HistoryResiduePolicy)
                self.assertEqual(policy.history_length, 3)
                self.assertEqual(policy.gamma, round(0.913 ** tr, 4))
                self.assertEqual(policy.actor.encoder.output_dim, 3 + 2 * tr)
                self.assertEqual(policy.qs.net[0].weight.shape[1], 3 + 4 * tr)
                self.assertEqual(cfg.online_task.env_runner.n_action_steps, tr)


class ReactiveHistoryWorkspaceTest(unittest.TestCase):
    def run_workspace(self, tmp, bootstrap='truncated', enable_eval=False, compile=False):
        module = workspace_with_eval_enabled() if enable_eval else workspace_module
        base_path = pathlib.Path(tmp) / 'base.ckpt'
        cfg = small_config(base_path, bootstrap)
        cfg.training.compile = compile
        save_base(base_path, cfg)
        env = HistoryVectorEnv()
        buffers, policies, initial_weights = [], [], []
        eval_runner = FakeEvalRunner()
        instantiate = hydra.utils.instantiate

        def instantiate_policy(config, **kwargs):
            if str(config._target_).endswith('RobomimicImageRunner'):
                return eval_runner
            return instantiate(config, **kwargs)

        def make_buffer(*args, **kwargs):
            buffer = ReactiveHistoryReplayBuffer(*args, **kwargs)
            buffers.append(buffer)
            return buffer

        def make_policy(*args, **kwargs):
            policy = HistorySumPolicy(*args, **kwargs)
            policies.append(policy)
            initial_weights.append(policy.res_policy.actor.mean.weight.detach().clone())
            return policy

        with patch.object(module.hydra.utils, 'instantiate', side_effect=instantiate_policy), \
             patch.object(module.FileUtils, 'get_env_metadata_from_dataset',
                          return_value={'env_kwargs': {}}), \
             patch.object(module, 'AsyncVectorEnv', return_value=env), \
             patch.object(module, 'ReactiveHistoryReplayBuffer', side_effect=make_buffer), \
             patch.object(module, 'SumPolicy', side_effect=make_policy), \
             patch.object(module, 'wandb') as wandb_mock:
            workspace = module.TrainOnlineReactiveHistoryRobomimicWorkspace(cfg, tmp)
            self.assertEqual((workspace.global_step, workspace.global_update, workspace.num_episodes),
                             (0, 0, 0))
            workspace.run()
        return workspace, buffers[0], policies[0], env, eval_runner, initial_weights[0], wandb_mock

    def test_rollout_replay_update_and_episode_counts(self):
        for bootstrap in ['truncated', 'never']:
            with self.subTest(bootstrap=bootstrap), tempfile.TemporaryDirectory() as tmp:
                workspace, rb, policy, env, evaluation, before, wandb_mock = self.run_workspace(tmp, bootstrap)
                self.assertEqual((workspace.global_step, workspace.global_update, workspace.num_episodes),
                                 (6, 3, 3))
                self.assertEqual((rb.pos, rb.n_steps, rb.history_length), (2, 2, 4))
                self.assertEqual(rb.observations.shape[-1], 1)
                self.assertEqual(rb.actions.shape[-1], 2)
                np.testing.assert_equal(rb.episode_ids[:2], [[0, 0, 0], [1, 1, 0]])
                np.testing.assert_allclose(rb.actions[0], 0)
                np.testing.assert_allclose(rb.a_base[0], [[10, 11], [20, 21], [30, 31]])
                np.testing.assert_allclose(rb.a_exec[:2], rb.a_base[:2] + 0.05 * rb.actions[:2])
                np.testing.assert_allclose(env.actions[0], rb.a_exec[0].reshape(3, 2, 1) * 3 + 5)
                np.testing.assert_allclose(rb.next_observations[0, 0], [7])
                np.testing.assert_allclose(rb.next_a_base[0], [[12, 13], [200, 201], [32, 33]])
                np.testing.assert_allclose(rb.next_a_base[1], [[102, 103], [202, 203], [80, 81]])
                np.testing.assert_equal(rb.exec_steps[:2], [[1, 1, 2], [2, 2, 1]])
                with patch('stable_baselines3.common.buffers.np.random.randint',
                           return_value=np.array([0, 1, 2])):
                    batch = rb._get_samples(np.zeros(3, dtype=np.int64))
                expected_dones = [0, 1, 0] if bootstrap == 'truncated' else [1, 1, 1]
                torch.testing.assert_close(batch.dones.flatten(), torch.tensor(expected_dones, dtype=torch.float32))
                self.assertEqual(len(env.actions), 2)
                self.assertIsNone(policy.pending)
                self.assertFalse(torch.equal(before, workspace.res_policy.actor.mean.weight))
                self.assertFalse(workspace.base_policy.training)
                self.assertTrue(all(not p.requires_grad for p in workspace.base_policy.parameters()))
                logs = [call.args[0] for call in wandb_mock.init.return_value.log.call_args_list]
                self.assertEqual([log['info/num_episodes'] for log in logs], [2, 3])
                self.assertEqual(evaluation.policies, [])
                self.assertFalse((pathlib.Path(tmp) / 'checkpoints').exists())

    def test_commented_eval_and_checkpoint_are_executable_and_isolated(self):
        compile = torch.compile
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(torch, 'compile', side_effect=lambda fn, **kwargs:
                          compile(fn, backend='aot_eager', **kwargs)) as compile_mock:
            workspace, rb, policy, env, evaluation, _, _ = self.run_workspace(
                tmp, enable_eval=True, compile=True)
            self.assertEqual(compile_mock.call_count, 3)
            for call, module in zip(compile_mock.call_args_list,
                                    (workspace.res_policy.actor, workspace.res_policy.qs,
                                     workspace.res_policy.q_targets)):
                self.assertIs(call.args[0].__self__, module)
            self.assertFalse(any('_orig_mod' in key for key in workspace.res_policy.state_dict()))
            self.assertEqual(len(evaluation.policies), 2)
            eval_policy = evaluation.policies[-1]
            self.assertIs(eval_policy.res_policy.actor.forward.__self__, eval_policy.res_policy.actor)
            self.assertIsNot(eval_policy, policy)
            self.assertNotEqual(eval_policy.base_policy.weight.data_ptr(), policy.base_policy.weight.data_ptr())
            self.assertNotEqual(eval_policy.res_policy.actor.mean.weight.data_ptr(),
                                policy.res_policy.actor.mean.weight.data_ptr())
            self.assertTrue(policy.res_policy.actor.mean.weight.requires_grad)
            self.assertFalse(eval_policy.res_policy.actor.mean.weight.requires_grad)
            torch.testing.assert_close(eval_policy.res_policy.actor.mean.weight, policy.res_policy.actor.mean.weight)
            self.assertEqual(workspace.num_episodes, 3)
            torch.testing.assert_close(policy.current_obs_emb[:, 0], torch.tensor([11., 21., 30.]))
            checkpoint = pathlib.Path(tmp) / 'checkpoints/step_6.ckpt'
            payload = torch.load(checkpoint, pickle_module=dill)
            self.assertEqual(set(payload), {'cfg', 'res_policy'})
            loaded = hydra.utils.instantiate(payload['cfg'].res_policy, obs_dim=1, action_dim=2)
            loaded.load_state_dict(payload['res_policy'])
            restored = HistorySumPolicy(0.05, 1, 0, 1, 4, copy.deepcopy(workspace.base_policy), loaded, 2)
            test_runner = FakeEvalRunner()
            test_runner.run(restored)
            torch.testing.assert_close(test_runner.actions[0], evaluation.actions[-1])

    def test_eval_cli_loads_history_and_preserves_old_routes(self):
        module = load_eval_module()
        for target, expected_cls in [('reactive_history_policy.HistoryResiduePolicy', HistorySumPolicy),
                                     ('reactive_policy.ResiduePolicy', ReactiveSumPolicy),
                                     ('residue_policy.ResiduePolicy', ActionSumPolicy)]:
            with self.subTest(target=target), tempfile.TemporaryDirectory() as tmp:
                base_path = pathlib.Path(tmp) / 'base.ckpt'
                cfg = small_config(base_path) if expected_cls is HistorySumPolicy else make_config([
                    f'online_task.base_ckpt={base_path}', 'n_rl_steps=2',
                    'online_task.shape_meta.action.shape=[1]', 'res_policy.hidden_dim=8',
                    'res_policy.num_qs=2', f'res_policy._target_=zprl.policy.{target}',
                ], history=False)
                save_base(base_path, cfg)
                action_dim = 4 if expected_cls is ActionSumPolicy else 2
                res_policy = hydra.utils.instantiate(cfg.res_policy, obs_dim=1, action_dim=action_dim)
                checkpoint = pathlib.Path(tmp) / 'online.ckpt'
                torch.save({'cfg': cfg, 'res_policy': res_policy.state_dict()}, checkpoint, pickle_module=dill)
                runner = FakeEvalRunner()
                instantiate = hydra.utils.instantiate
                runner_steps = []

                def instantiate_policy(config, **kwargs):
                    if str(config._target_).endswith('RobomimicImageRunner'):
                        runner_steps.append(config.n_action_steps)
                        return runner
                    return instantiate(config, **kwargs)

                with patch.object(module.hydra.utils, 'instantiate', side_effect=instantiate_policy):
                    result = CliRunner().invoke(module.main, ['-c', str(checkpoint), '-o',
                                                             str(pathlib.Path(tmp) / 'eval'), '-d', 'cpu'])
                if result.exception:
                    raise result.exception
                self.assertEqual(result.exit_code, 0, result.output)
                self.assertIsInstance(runner.policies[0], expected_cls)
                self.assertEqual(runner_steps, [action_dim])
                self.assertFalse(runner.policies[0].res_policy.training)
                self.assertTrue(all(not p.requires_grad for p in runner.policies[0].res_policy.parameters()))
                log = json.loads((pathlib.Path(tmp) / 'eval/eval_log.json').read_text())
                self.assertEqual(log['test/grasp_completed'], 0.5)


if __name__ == '__main__':
    unittest.main()
