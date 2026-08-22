import pathlib
import unittest

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


class RobomimicSubtaskConfigTest(unittest.TestCase):
    def test_runner_subtask_tracks_online_task_overrides(self):
        OmegaConf.register_new_resolver('eval', eval, replace=True)
        config_dir = pathlib.Path(__file__).parents[1] / 'zprl' / 'config'
        with initialize_config_dir(
                version_base=None, config_dir=str(config_dir)):
            cfg = compose(
                config_name='train_online_robomimic_workspace',
                overrides=[
                    'online_task.subtask.enabled=false',
                    'online_task.subtask.reward_mode=sparse',
                ]
            )

        self.assertFalse(cfg.online_task.env_runner.subtask.enabled)
        self.assertEqual(
            cfg.online_task.env_runner.subtask.reward_mode, 'sparse')


if __name__ == '__main__':
    unittest.main()
