import ast
import importlib
import pathlib
import unittest
from unittest.mock import patch

import gym
import numpy as np
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra

from zprl.env.adroit.adroit import _AdroitGoalCountWrapper

WORKSPACE_MODULES = [
    "zprl.workspace.train_online_workspace",
    "zprl.workspace.train_online_vib_workspace",
    "zprl.workspace.train_online_noise_workspace",
]

class FakeImageEnv(gym.Env):
    metadata = {}
    def __init__(self, *args, **kwargs):
        self.reset_count = 0
        self.step_count = 0
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict({
            "image": gym.spaces.Box(
                low=0.0, high=10.0, shape=(1, 2, 2), dtype=np.float32),
            "agent_pos": gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
        })
    def _obs(self, value):
        return {
            "image": np.full((1, 2, 2), value, dtype=np.float32),
            "agent_pos": np.array([value], dtype=np.float32),
        }
    def reset(self):
        self.reset_count += 1
        self.step_count = 0
        return self._obs(self.reset_count)
    def step(self, action):
        self.step_count += 1
        return self._obs(self.step_count), 0.0, False, {
            "n_goal_achieved": 1, "success": False}
    def seed(self, seed=None):
        raise AssertionError("training factory must not seed the environment")
    def close(self):
        pass

class AdroitMetaworldOnlineWorkspaceTest(unittest.TestCase):
    def test_all_workspace_factories_support_both_envs_without_seed(self):
        for module_name in WORKSPACE_MODULES:
            module = importlib.import_module(module_name)
            for env_type in ("adroit", "metaworld"):
                env_class = "AdroitEnv" if env_type == "adroit" else "MetaWorldEnv"
                with patch.object(module, env_class, FakeImageEnv):
                    env = module._make_env_fn(
                        env_type, "task", 0, 1, 2, 5)()
                    first = env.reset()
                    second = env.reset()
                    self.assertFalse(np.array_equal(
                        first["image"], second["image"]))
                    env.close()

    def test_adroit_goal_count_survives_multistep_info_window(self):
        for module_name in WORKSPACE_MODULES:
            module = importlib.import_module(module_name)
            fake_adroit_env = lambda *args, **kwargs: \
                _AdroitGoalCountWrapper(FakeImageEnv())
            with patch.object(module, "AdroitEnv", fake_adroit_env):
                env = module._make_env_fn("adroit", "door", 0, 1, 2, 5)()
                env.reset()
                _, _, done, info = env.step(
                    np.zeros((2, 2), dtype=np.float32))
                self.assertFalse(done)
                value = np.asarray(
                    info["accumulated_goal_achieved"]).reshape(-1)[-1]
                self.assertEqual(int(value), 2)
                env.reset()
                _, _, _, info = env.step(
                    np.zeros((2, 2), dtype=np.float32))
                value = np.asarray(
                    info["accumulated_goal_achieved"]).reshape(-1)[-1]
                self.assertEqual(int(value), 2)
                env.close()

    def test_six_main_config_combinations_compose(self):
        names = [
            "train_online_workspace", "train_online_vib_workspace",
            "train_online_noise_workspace",
        ]
        for name in names:
            for task in ("adroit_door", "metaworld_box-close"):
                GlobalHydra.instance().clear()
                with initialize(version_base=None, config_path="../zprl/config"):
                    cfg = compose(
                        config_name=name, overrides=[f"online_task={task}"])
                self.assertTrue(str(cfg._target_).startswith("zprl.workspace."))
                self.assertEqual(cfg.training.bootstrap_at_done, "truncated")

    def test_shared_current_features_and_no_train_seed(self):
        root = pathlib.Path(__file__).parents[1]
        for module_name in WORKSPACE_MODULES:
            name = module_name.rsplit(".", 1)[-1] + ".py"
            source = (root / "zprl/workspace" / name).read_text()
            tree = ast.parse(source)
            for token in (
                    "prepare_base_policy_config", "get_crop_randomizers",
                    "cfg.num_inference_steps", "terminal_observation",
                    "eval_env_runner.run", "torch.save"):
                self.assertIn(token, source)
            seed_calls = [
                node for node in ast.walk(tree)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "seed"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in ("env", "envs")
            ]
            self.assertEqual(seed_calls, [], module_name)

    def test_eval_clis_have_no_import_time_dataset_check(self):
        root = pathlib.Path(__file__).parents[1]
        for name in ("eval_base.py", "eval_sum.py"):
            source = (root / name).read_text()
            self.assertNotIn("raise ValueError", source.split("def main", 1)[0])
            self.assertIn("env_runner.close()", source)

if __name__ == "__main__":
    unittest.main()
