import io
import json
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
from stable_baselines3.common.type_aliases import ReplayBufferSamples

from zprl.common.reactive_replay_buffer import ReactiveNStepReplayBuffer
from zprl.policy.residue_policy import ResiduePolicy
from zprl.workspace import train_online_robomimic_workspace as workspace_module
from tests.test_reactive_n_step_replay import ConstantQ, sample_at
from tests.test_reactive_policy import DummyBasePolicy, FakeVectorEnv


class WholeChunkFeaturesTest(unittest.TestCase):
    def make_batch(self):
        return ReplayBufferSamples(
            observations=torch.zeros(2, 1),
            actions=torch.tensor([[0.2, 2., 3.], [0.4, 4., 5.]]),
            next_observations=torch.ones(2, 1),
            dones=torch.tensor([[0.], [1.]]),
            rewards=torch.tensor([[-2.], [2.]]),
            discounts=torch.tensor([[0.25], [0.5]]),
        )

    def test_clip_applies_to_complete_target(self):
        policy = ResiduePolicy(1, 1, hidden_dim=8, num_qs=2,
                               auto_alpha=False, q_ent=False, target_clip=[0, 1])
        policy.qs = ConstantQ(0., 2)
        policy.q_targets = ConstantQ(4., 2)
        loss, info = policy.compute_critic_loss(self.make_batch())
        torch.testing.assert_close(loss, torch.tensor(1.))
        torch.testing.assert_close(info['q_target'], torch.tensor(0.5))
        policy.target_clip = None
        loss, info = policy.compute_critic_loss(self.make_batch())
        torch.testing.assert_close(loss, torch.tensor(5.))
        torch.testing.assert_close(info['q_target'], torch.tensor(0.5))

    def test_critic_inputs_gradients_and_checkpoint_roundtrip(self):
        batch = self.make_batch()
        for mode in ('exec', 'base_res'):
            with self.subTest(mode=mode):
                policy = ResiduePolicy(1, 1, hidden_dim=8, num_qs=2,
                                       res_scale=0.5, critic_action_input=mode)
                inputs = []
                hook = policy.qs.register_forward_pre_hook(
                    lambda module, args: inputs.append(args[1].detach().clone()))
                loss, info = policy.compute_critic_loss(batch)
                expected = (torch.tensor([[2.1], [4.2]]) if mode == 'exec'
                            else torch.tensor([[2., 0.2], [4., 0.4]]))
                torch.testing.assert_close(inputs[0], expected)
                loss.backward()
                self.assertIsNone(policy.log_alpha.grad)
                self.assertTrue(all(not value.requires_grad for value in info.values()))
                policy.zero_grad(set_to_none=True)
                loss, info = policy.compute_actor_loss(batch)
                loss.backward()
                self.assertGreater(policy.actor.mean.weight.grad.abs().sum(), 0)
                self.assertIsNone(policy.log_alpha.grad)
                self.assertTrue(all(not value.requires_grad for value in info.values()))
                hook.remove()

                stream = io.BytesIO()
                torch.save(policy.state_dict(), stream, pickle_module=dill)
                stream.seek(0)
                loaded = ResiduePolicy(1, 1, hidden_dim=8, num_qs=2,
                                       res_scale=0.5, critic_action_input=mode)
                loaded.load_state_dict(torch.load(stream, pickle_module=dill))
                for method in ('compute_critic_loss', 'compute_actor_loss'):
                    torch.manual_seed(7)
                    expected_loss, _ = getattr(policy, method)(batch)
                    torch.manual_seed(7)
                    actual_loss, _ = getattr(loaded, method)(batch)
                    torch.testing.assert_close(actual_loss, expected_loss)

    def test_workspace_episode_count_step_and_n_step_endpoints(self):
        config_dir = pathlib.Path(__file__).resolve().parents[1] / 'zprl' / 'config'
        for n_step in (1, 2):
            for bootstrap in ('never', 'truncated'):
                with self.subTest(n_step=n_step, bootstrap=bootstrap), \
                     tempfile.TemporaryDirectory() as tmp:
                    base_path = pathlib.Path(tmp) / 'base.ckpt'
                    base_path.touch()
                    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
                        cfg = compose(config_name='train_online_robomimic_workspace', overrides=[
                            f'online_task.base_ckpt={base_path}', 'online_task.abs_action=false',
                            'online_task.shape_meta.action.shape=[1]', 'training.device=cpu',
                            'online_task.n_envs=3', 'training.training_freq=6',
                            'training.num_steps=6', 'training.learning_start=6',
                            'training.log_every=6', 'training.eval_every=6',
                            'training.checkpoint_every=6', 'training.utd=1',
                            'training.batch_size=2', 'training.buffer_size=12',
                            'res_policy.hidden_dim=8', 'res_policy.num_qs=2',
                            f'training.n_step={n_step}',
                            f'training.bootstrap_at_done={bootstrap}',
                        ])
                    self.assertFalse(cfg.logging.resume)
                    self.assertNotIn('preload_offline_data', cfg.training)
                    base = DummyBasePolicy()
                    payload = {'cfg': OmegaConf.create({'task_name': cfg.task_name, 'policy': {}}),
                               'state_dicts': {'ema_model': base.state_dict()}}
                    env = FakeVectorEnv()
                    buffers = []
                    instantiate = hydra.utils.instantiate

                    def instantiate_policy(config, **kwargs):
                        return instantiate(config, **kwargs) if '_target_' in config else base

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
                         patch.object(workspace_module, 'wandb') as wandb:
                        workspace = workspace_module.TrainOnlineRobomimicWorkspace(cfg, tmp)
                        workspace.run()

                    self.assertEqual(workspace.num_episodes, 3)
                    self.assertEqual(workspace.global_update, 6)
                    self.assertEqual(wandb.init.return_value.log.call_args.kwargs['step'], 24)
                    log = json.loads((pathlib.Path(tmp) / 'logs.json.txt').read_text())
                    self.assertEqual(log['info/num_episodes'], 3)
                    self.assertEqual(log['info/global_step'], 24)
                    self.assertIsInstance(log['info/q_target'], float)
                    rb = buffers[0]
                    batch = sample_at(rb, [0, 0, 0], [0, 1, 2])
                    gamma = cfg.res_policy.gamma
                    torch.testing.assert_close(batch.rewards.flatten(),
                        torch.tensor([1., 1., 1. + gamma if n_step == 2 else 1.]))
                    terminal_base = [70., 71., 72., 73.]
                    next_base = (terminal_base if bootstrap == 'truncated'
                                 else [100., 101., 102., 103.])
                    np.testing.assert_allclose(batch.actions[0, 8:], next_base)
                    last_base = ([80., 81., 82., 83.] if bootstrap == 'truncated'
                                 else [300., 301., 302., 303.])
                    np.testing.assert_allclose(batch.actions[2, 8:],
                        last_base if n_step == 2 else [40., 41., 42., 43.])
                    expected_dones = [0. if bootstrap == 'truncated' else 1., 1.,
                                      float(n_step == 2 and bootstrap == 'never')]
                    torch.testing.assert_close(batch.dones.flatten(), torch.tensor(expected_dones))
                    self.assertFalse(base.training)
                    self.assertFalse(base.weight.requires_grad)


if __name__ == '__main__':
    unittest.main()
