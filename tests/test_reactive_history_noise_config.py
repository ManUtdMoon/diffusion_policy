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
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from tests.test_reactive_history_noise_policy import FakeNoiseBase, make_obs
from tests.test_reactive_history_noise_replay import sample_at
from tests.test_robomimic_image_runner_subtask import DummyProgress
from zprl.common.reactive_history_noise_replay_buffer import ReactiveHistoryNoiseReplayBuffer
from zprl.env_runner.robomimic_image_runner import RobomimicImageRunner
from zprl.model.gru_history_encoder import GRUHistoryEncoder
from zprl.policy.reactive_history_noise_policy import HistoryNoisePolicy, HistoryNoiseSumPolicy
from zprl.workspace import train_online_reactive_history_noise_robomimic_workspace as workspace_module


ROOT = pathlib.Path(__file__).resolve().parents[1]
CONFIG_NAME = 'train_online_reactive_history_noise_robomimic_workspace'


def make_config(overrides=(), name=CONFIG_NAME):
    OmegaConf.register_new_resolver('eval', eval, replace=True)
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / 'zprl/config')):
        return compose(config_name=name, overrides=list(overrides))


def numpy_obs(values, masks=None):
    return {key: value.numpy() for key, value in make_obs(values, masks).items()}


class WorkspaceVectorEnv:
    def __init__(self):
        self.index = 0
        self.actions = []

    def reset(self):
        return numpy_obs([1, 2, 3])

    def step(self, action):
        assert action.shape == (3, 4, 10)
        self.actions.append(action.copy())
        rows = [([10, 20, 4], [True, True, False], [1, 2, 4]),
                ([11, 21, 30], [False, False, True], [4, 4, 1]),
                ([12, 22, 31], [False, False, False], [4, 4, 4])]
        values, dones, steps = rows[self.index]
        infos = [{'executed_action_steps': k, 'task_reward': float(i == 1),
                  'completed_stage_mask': np.array([[float(i == 1)]], dtype=np.float32)}
                 for i, k in enumerate(steps)]
        if self.index in (0, 1):
            env = 0 if self.index == 0 else 2
            terminal = numpy_obs([99 if env == 0 else 88], [1])
            infos[env].update({'TimeLimit.truncated': True,
                               'terminal_observation': {key: value[0] for key, value in terminal.items()}})
        self.index += 1
        return numpy_obs(values), np.ones(3, dtype=np.float32), np.array(dones), infos


class WorkspaceBase(FakeNoiseBase):
    def __init__(self, n_obs_steps=1, action_dim=10, n_action_steps=4, num_inference_steps=2):
        super().__init__(n_obs_steps, action_dim)
        self.n_action_steps = n_action_steps
        self.num_inference_steps = num_inference_steps


class FakeEvalRunner:
    def __init__(self):
        self.policies = []
        self.actions = []

    def run(self, policy):
        self.policies.append(policy)
        obs = numpy_obs([1, 2])
        policy.reset(obs)
        self.actions.append(policy.predict_action(obs)['action'].clone())
        policy.on_env_step(obs, [True, True], [{'executed_action_steps': 4}] * 2)
        return {'test/mean_score': 0.0}


def workspace_with_eval_enabled():
    # Test the disabled blocks in memory without changing production scheduling.
    source = pathlib.Path(workspace_module.__file__).read_text()
    blocks = [
        ('        ## eval, only average score needed\n', '        ## train\n'),
        ('        self.noise_policy.to(device)\n', '        optimizers = '),
        ('        with JsonLogger(log_path) as logger:\n', '            while self.global_step'),
        ('                # evaluation\n', '                sum_policy.train()'),
        ('                # checkpointing\n', '\n\n@hydra.main'),
    ]
    for start, end in blocks:
        first = source.index(start) + len(start)
        last = source.index(end, first)
        source = source[:first] + re.sub(r'^(\s*)# (.*)$', r'\1\2', source[first:last],
                                         flags=re.MULTILINE) + source[last:]
    module = types.ModuleType('noise_workspace_eval_test')
    module.__file__ = workspace_module.__file__
    exec(compile(source, module.__file__, 'exec'), module.__dict__)
    return module


def small_config(base_path, overrides=()):
    return make_config([
        'online_task=tool_hang_image_abs', f'online_task.base_ckpt={base_path}',
        'online_task.abs_action=false', 'online_task.n_envs=3', 'horizon=8',
        'n_action_steps=4', 'training.device=cpu', 'training.num_steps=9',
        'training.training_freq=3', 'training.learning_start=6',
        'training.n_step=2',
        'training.compile=false', 'training.log_every=3', 'training.eval_every=3',
        'training.checkpoint_every=3', 'training.utd=1', 'training.batch_size=2',
        'training.buffer_size=24', 'noise_policy.hidden_dim=8', 'noise_policy.num_qs=2',
        'noise_policy.history_length=4', 'noise_policy.gru_input_dim=8',
        'noise_policy.gru_hidden_dim=8',
    ] + list(overrides))


class NoiseHistoryConfigTest(unittest.TestCase):
    def test_history_baseline_can_be_selected_with_explicit_experiment_overrides(self):
        history = make_config(name='train_online_reactive_history_robomimic_workspace')
        # Experiment defaults may be edited; explicitly select the history baseline here.
        overrides = ['online_task=square_image_abs']
        excluded_policy = {'_target_', 'obs_dim', 'action_dim', 'actor_input', 'critic_action_input', 'res_scale', 'gamma'}
        for key, value in OmegaConf.to_container(history.res_policy, resolve=True).items():
            if key not in excluded_policy:
                overrides.append(f'noise_policy.{key}={json.dumps(value)}')
        for key, value in OmegaConf.to_container(history.training, resolve=True).items():
            if key not in ('res_scale', 'prog_explore'):
                overrides.append(f'training.{key}={json.dumps(value)}')
        for key in ('horizon', 'n_obs_steps', 'n_action_steps', 'n_latency_steps', 'single_gamma', 'num_inference_steps'):
            overrides.append(f'{key}={json.dumps(history[key])}')
        cfg = make_config(overrides)
        dsrl = make_config(name='train_online_noise_robomimic_workspace')
        self.assertEqual(hydra.utils.get_class(cfg._target_),
                         workspace_module.TrainOnlineReactiveHistoryNoiseRobomimicWorkspace)
        self.assertEqual(hydra.utils.get_class(cfg.noise_policy._target_), HistoryNoisePolicy)
        for key in history.res_policy:
            if key in excluded_policy:
                continue
            self.assertEqual(cfg.noise_policy[key], history.res_policy[key], key)
        for key in history.training:
            if key in ('res_scale', 'prog_explore'):
                continue
            self.assertEqual(cfg.training[key], history.training[key], key)
        for key in ('horizon', 'n_obs_steps', 'n_action_steps', 'n_latency_steps', 'single_gamma', 'num_inference_steps'):
            self.assertEqual(cfg[key], history[key], key)
        for key in history.logging:
            if key in ('name', 'tags'):
                continue
            self.assertEqual(cfg.logging[key], history.logging[key], key)
        self.assertEqual(cfg.training.noise_scale, dsrl.training.noise_scale)
        self.assertEqual(cfg.noise_policy.target_entropy, dsrl.noise_policy.target_entropy)
        self.assertEqual(cfg.n_noise_steps, dsrl.n_noise_steps)
        self.assertNotIn('res_policy', cfg)
        self.assertNotIn('n_rl_steps', cfg)
        for key in ('res_scale', 'prog_explore'):
            self.assertNotIn(key, cfg.training)
        for key in ('actor_input', 'critic_action_input', 'res_scale'):
            self.assertNotIn(key, cfg.noise_policy)
        source = pathlib.Path(workspace_module.__file__).read_text()
        for token in ('prog_explore', 'res_mask', 'res_ratio', 'nnoise_norm', 'base_n_rms', 'vib_forward'):
            self.assertNotIn(token, source)

    def test_execution_and_noise_overrides_reach_policy_runner_and_discount(self):
        for task, ta, tn, h in (('square', 4, 1, 1), ('transport', 5, 2, 3), ('tool_hang', 8, 1, 5)):
            cfg = make_config([f'online_task={task}_image_abs', f'n_action_steps={ta}',
                               f'n_noise_steps={tn}', f'noise_policy.history_length={h}',
                               'single_gamma=0.913', 'noise_policy.hidden_dim=8',
                               'noise_policy.gru_hidden_dim=8', 'noise_policy.gru_input_dim=8',
                               'noise_policy.target_entropy=-3', 'noise_policy.target_clip=null',
                               'noise_policy.q_ent=true', 'training.n_step=2'])
            policy = hydra.utils.instantiate(cfg.noise_policy, obs_dim=2,
                                             action_dim=tn * 10, exec_dim=ta * 10)
            self.assertEqual(cfg.online_task.env_runner.n_action_steps, ta)
            self.assertEqual(policy.gamma, round(0.913 ** ta, 4))
            self.assertEqual(policy.history_length, h)
            self.assertEqual(policy.action_dim, tn * 10)
            self.assertEqual(policy.exec_dim, ta * 10)
            self.assertEqual(policy.target_entropy, -3)
            self.assertIsNone(policy.target_clip)
            self.assertTrue(policy.q_ent)
            self.assertIsInstance(policy.actor.encoder, GRUHistoryEncoder)
            self.assertEqual(len(policy.actor.encoder.cells), 1)
            self.assertEqual(cfg.training.n_step, 2)


class NoiseHistoryWorkspaceTest(unittest.TestCase):
    def run_workspace(self, tmp, bootstrap='truncated', enable_eval=False, compile=False,
                      debug=False, overrides=(), base_task=None):
        module = workspace_with_eval_enabled() if enable_eval else workspace_module
        cfg = small_config(pathlib.Path(tmp) / 'base.ckpt', overrides)
        cfg.training.bootstrap_at_done = bootstrap
        cfg.training.compile = compile
        cfg.training.debug = debug
        base_cfg = OmegaConf.create({
            'task_name': cfg.task_name if base_task is None else base_task,
            'policy': {'_target_': 'tests.test_reactive_history_noise_config.WorkspaceBase',
                       'n_obs_steps': 1, 'action_dim': 10}})
        torch.save({'cfg': base_cfg, 'state_dicts': {'ema_model': FakeNoiseBase().state_dict()}},
                   cfg.online_task.base_ckpt, pickle_module=dill)
        env, evaluation = WorkspaceVectorEnv(), FakeEvalRunner()
        buffers, policies, before = [], [], []
        instantiate = hydra.utils.instantiate

        def instantiate_policy(config, **kwargs):
            if str(config._target_).endswith('RobomimicImageRunner'):
                return evaluation
            return instantiate(config, **kwargs)

        def make_buffer(*args, **kwargs):
            buffer = ReactiveHistoryNoiseReplayBuffer(*args, **kwargs)
            buffers.append(buffer)
            return buffer

        def make_sum(*args, **kwargs):
            policy = HistoryNoiseSumPolicy(*args, **kwargs)
            policies.append(policy)
            before.append(policy.noise_policy.actor.mean.weight.detach().clone())
            return policy

        with patch.object(module.hydra.utils, 'instantiate', side_effect=instantiate_policy), \
             patch.object(module.FileUtils, 'get_env_metadata_from_dataset',
                          return_value={'env_kwargs': {}}), \
             patch.object(module, 'AsyncVectorEnv', return_value=env), \
             patch.object(module, 'ReactiveHistoryNoiseReplayBuffer', side_effect=make_buffer), \
             patch.object(module, 'SumPolicy', side_effect=make_sum), \
             patch.object(module, 'wandb') as wandb:
            workspace = module.TrainOnlineReactiveHistoryNoiseRobomimicWorkspace(cfg, tmp)
            workspace.run()
        logs = [json.loads(line) for line in (pathlib.Path(tmp) / 'logs.json.txt').read_text().splitlines()]
        return workspace, buffers[0], policies[0], env, evaluation, before[0], wandb, logs

    def test_rollout_warmup_replay_updates_logs_and_episode_counts(self):
        original = HistoryNoisePolicy.predict_noise
        calls = []

        def predict(policy, *args, **kwargs):
            calls.append(kwargs['argmax'])
            return original(policy, *args, **kwargs)

        for bootstrap in ('truncated', 'never'):
            with tempfile.TemporaryDirectory() as tmp, \
                 patch.object(HistoryNoisePolicy, 'predict_noise', autospec=True, side_effect=predict):
                calls.clear()
                workspace, rb, policy, env, evaluation, before, wandb, logs = self.run_workspace(tmp, bootstrap)
                self.assertEqual((workspace.global_step, workspace.global_update, workspace.num_episodes), (9, 6, 3))
                self.assertEqual(calls, [False, False])  # No actor candidate at warmup.
                self.assertEqual((rb.pos, rb.n_steps, rb.action_dim, rb.exec_dim), (3, 2, 10, 40))
                np.testing.assert_equal(rb.episode_ids[:3], [[0, 0, 0], [1, 1, 0], [1, 1, 1]])
                np.testing.assert_equal(rb.exec_steps[:3], [[1, 2, 4], [4, 4, 1], [4, 4, 4]])
                np.testing.assert_allclose(rb.next_observations[0, 0], [99., 1.])
                np.testing.assert_allclose(rb.next_observations[1, 2], [88., 1.])
                self.assertTrue(np.all(rb.actions[:3] <= 1) and np.all(rb.actions[:3] >= -1))
                for index, action in enumerate(env.actions):
                    expected_noise = rb.actions[index, :, None].repeat(8, axis=1) * 1.5
                    np.testing.assert_allclose(policy.base_policy.decode_calls[index][1].numpy(), expected_noise)
                    plan = expected_noise + np.arange(8)[None, :, None]
                    np.testing.assert_allclose(action, plan[:, :4] * 3 + 5, rtol=1e-6)
                batch = sample_at(rb, [0, 0, 0], [0, 1, 2])
                wanted = [0., 1., 0.] if bootstrap == 'truncated' else [1., 1., 1.]
                torch.testing.assert_close(batch.dones[:, 0], torch.tensor(wanted))
                self.assertFalse(torch.equal(before, workspace.noise_policy.actor.mean.weight))
                self.assertFalse(workspace.base_policy.training)
                self.assertTrue(all(not p.requires_grad and p.grad is None for p in workspace.base_policy.parameters()))
                self.assertFalse(workspace.noise_policy.q_targets.training)
                self.assertIsNone(policy.pending)
                self.assertEqual(evaluation.policies, [])
                self.assertFalse((pathlib.Path(tmp) / 'checkpoints').exists())
                self.assertEqual(set(logs[0]), {'info/global_step', 'info/chunk_step', 'info/num_episodes'})
                self.assertEqual([log['info/global_step'] for log in logs], [12, 24, 36])
                self.assertEqual([log['info/chunk_step'] for log in logs], [3, 6, 9])
                self.assertEqual([log['info/global_update'] for log in logs[1:]], [3, 6])
                for log in logs[1:]:
                    amplitude = [key for key in log if 'rms' in key or 'ratio' in key or 'norm' in key and key.startswith('info/')]
                    self.assertEqual(amplitude, ['info/res_n_rms'])
                    self.assertTrue(all(np.isfinite(value) for value in log.values()))
                    self.assertIn('info/recent_frame_assembled_rate', log)
                self.assertEqual([call.kwargs['step'] for call in wandb.init.return_value.log.call_args_list], [12, 24, 36])

    def test_encoder_learning_rate_reaches_real_workspace_updates(self):
        original = HistoryNoisePolicy.get_optimizer
        optimizers = []

        def get_optimizer(policy, *args, **kwargs):
            result = original(policy, *args, **kwargs)
            optimizers.append(result)
            return result

        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(HistoryNoisePolicy, 'get_optimizer', autospec=True, side_effect=get_optimizer):
            workspace, *_ = self.run_workspace(tmp)
        self.assertEqual(len(optimizers), 1)
        for name, module in (('actor_optimizer', workspace.noise_policy.actor),
                              ('q_optimizer', workspace.noise_policy.qs)):
            optimizer = optimizers[0][name]
            self.assertEqual([group['lr'] for group in optimizer.param_groups], [1e-4, 1e-5])
            self.assertTrue(all(parameter in optimizer.state for parameter in module.encoder.parameters()))

    def test_debug_skips_compile_before_collection(self):
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(torch, 'compile') as compile_mock, \
             patch('tests.test_reactive_history_noise_config.ReactiveHistoryNoiseReplayBuffer',
                   side_effect=RuntimeError('stop after compile setup')):
            with self.assertRaisesRegex(RuntimeError, 'stop after compile setup'):
                self.run_workspace(tmp, compile=True, debug=True)
            compile_mock.assert_not_called()

    def test_task_and_checkpoint_dimensions_cannot_be_bypassed(self):
        for overrides, base_task in (((), 'square_image_abs'), (('horizon=16',), None),
                                      (('n_obs_steps=2',), None),
                                      (('online_task.shape_meta.action.shape=[7]',), None)):
            with tempfile.TemporaryDirectory() as tmp, self.assertRaises(AssertionError):
                self.run_workspace(tmp, overrides=overrides, base_task=base_task)

    def test_update_order_matches_history_template(self):
        sequence = []
        methods = ('compute_critic_loss', 'target_update', 'compute_actor_loss', 'compute_alpha_loss')
        originals = {name: getattr(HistoryNoisePolicy, name) for name in methods}

        def record(name):
            def invoke(policy, *args, **kwargs):
                sequence.append(name)
                return originals[name](policy, *args, **kwargs)
            return invoke

        from contextlib import ExitStack
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            for name in methods:
                stack.enter_context(patch.object(HistoryNoisePolicy, name, autospec=True, side_effect=record(name)))
            self.run_workspace(tmp)
        expected = []
        for update in range(1, 7):
            expected += list(methods[:2])
            if update % 2 == 0:
                expected += list(methods[2:])
        self.assertEqual(sequence, expected)

    def test_commented_eval_save_and_compiled_checkpoint_round_trip(self):
        compile_forward = torch.compile
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(torch, 'compile', side_effect=lambda fn, **kwargs:
                          compile_forward(fn, backend='aot_eager', **kwargs)) as compile_mock:
            workspace, _, policy, _, evaluation, _, _, _ = self.run_workspace(tmp, enable_eval=True, compile=True)
            self.assertEqual(compile_mock.call_count, 3)
            self.assertEqual(len(evaluation.policies), 3)
            eval_policy = evaluation.policies[-1]
            self.assertNotEqual(eval_policy.base_policy.weight.data_ptr(), policy.base_policy.weight.data_ptr())
            self.assertFalse(eval_policy.noise_policy.actor.mean.weight.requires_grad)
            self.assertTrue(policy.noise_policy.actor.mean.weight.requires_grad)
            self.assertIs(eval_policy.noise_policy.actor.forward.__self__, eval_policy.noise_policy.actor)
            payload = torch.load(pathlib.Path(tmp) / 'checkpoints/step_9.ckpt', pickle_module=dill)
            self.assertEqual(set(payload), {'cfg', 'noise_policy'})
            self.assertFalse(any('_orig_mod' in key for key in payload['noise_policy']))
            restored_noise = hydra.utils.instantiate(payload['cfg'].noise_policy, obs_dim=2, action_dim=10, exec_dim=40)
            restored_noise.load_state_dict(payload['noise_policy'])
            restored = HistoryNoiseSumPolicy(
                noise_scale=1.5, base_obs_emb_dim=1, subtask_dim=1, action_dim=10,
                n_action_steps=4, horizon=8, n_obs_steps=1,
                base_policy=copy.deepcopy(workspace.base_policy), noise_policy=restored_noise).eval()
            runner = FakeEvalRunner()
            runner.run(restored)
            torch.testing.assert_close(runner.actions[0], evaluation.actions[-1], atol=0, rtol=0)


class RunnerVectorEnv:
    def __init__(self):
        self.chunk = -1
        self.actions = []

    def call_each(self, name, args_list):
        self.chunk += 1
        self.index = 0

    def reset(self):
        return numpy_obs([1 + self.chunk * 10, 2 + self.chunk * 10])

    def step(self, action):
        assert action.shape == (2, 4, 10)
        self.actions.append(action.copy())
        dones = [[True, False], [True, True]][self.index]
        infos = [{'task_reward': 0., 'completed_stage_mask': np.array([[0.]]),
                  'executed_action_steps': 1 if done else 4} for done in dones]
        if self.index == 0:
            infos[0].update({'TimeLimit.truncated': True,
                             'terminal_observation': {key: value[0] for key, value in numpy_obs([99]).items()}})
        self.index += 1
        return numpy_obs([3 + self.index, 4 + self.index]), np.zeros(2), np.array(dones), infos

    def render(self):
        return [None, None]


class NoiseHistoryRunnerTest(unittest.TestCase):
    @patch('zprl.env_runner.robomimic_image_runner.tqdm.tqdm', return_value=DummyProgress())
    def test_existing_runner_calls_new_policy_lifecycle_across_chunks(self, _):
        from tests.test_reactive_history_noise_policy import make_sum
        policy, base = make_sum()
        policy.eval()
        runner = RobomimicImageRunner.__new__(RobomimicImageRunner)
        runner.env = RunnerVectorEnv()
        runner.env_fns = [None, None]
        runner.env_init_fn_dills = [b'0', b'1', b'2']
        runner.env_seeds = [0, 1, 2]
        runner.env_prefixs = ['train/', 'train/', 'test/']
        runner.env_meta = {'env_name': 'ToolHang'}
        runner.n_obs_steps = 1
        runner.n_action_steps = 4
        runner.past_action = False
        runner.max_steps = 8
        runner.action_pose_repr = 'delta'
        runner.subtask_stages = ('frame_assembled',)
        runner.tqdm_interval_sec = 0
        transitions = []
        feedback = policy.on_env_step

        def on_env_step(*args):
            transition = feedback(*args)
            transitions.append(transition)
            return transition

        with patch.object(policy, 'reset', wraps=policy.reset) as reset, \
             patch.object(policy, 'on_env_step', side_effect=on_env_step) as step:
            results = runner.run(policy)
        self.assertEqual(reset.call_count, 2)
        self.assertEqual(step.call_count, 4)
        self.assertEqual(len(base.decode_calls), 4)
        torch.testing.assert_close(transitions[0]['next_obs'][0], torch.tensor([99., 0.]))
        torch.testing.assert_close(transitions[0]['a_exec'][0, 10:], torch.zeros(30))
        self.assertIsNone(policy.pending)
        self.assertIn('test/frame_assembled_completion_rate', results)
        self.assertFalse(base.training)


if __name__ == '__main__':
    unittest.main()
