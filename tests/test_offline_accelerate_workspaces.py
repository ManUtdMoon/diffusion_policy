import pathlib
import unittest
from unittest.mock import patch

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from zprl.policy.flow_match_vib_unet_image_policy import (
    FlowMatchVibUnetImagePolicy,
)

from zprl.workspace.train_diffusion_image_accelerate_workspace import (
    TrainDiffusionImageAccelerateWorkspace,
    _split_total_batch_size,
)
from zprl.workspace.train_flow_match_vib_unet_image_accelerate_workspace import (
    TrainFlowMatchVibUnetImageAccelerateWorkspace,
)


CONFIG_DIR = (pathlib.Path(__file__).parent.parent / "zprl" / "config").resolve()


class TestOfflineAccelerateWorkspaces(unittest.TestCase):
    def test_vib_forward_delegates_to_compute_loss(self):
        policy = FlowMatchVibUnetImagePolicy.__new__(
            FlowMatchVibUnetImagePolicy)
        batch = object()
        expected = (object(), {"metric": 1.0})
        with patch.object(
                FlowMatchVibUnetImagePolicy, "compute_loss",
                return_value=expected) as compute_loss:
            result = policy.forward(batch)

        self.assertIs(result, expected)
        compute_loss.assert_called_once_with(batch)

    def test_accelerate_configs_compose(self):
        cases = [
            (
                "train_diffusion_image_accelerate_workspace",
                "zprl.policy.flow_match_unet_image_policy.FlowMatchUnetImagePolicy",
                "zprl.workspace.train_diffusion_image_accelerate_workspace.TrainDiffusionImageAccelerateWorkspace",
            ),
            (
                "train_flow_match_vib_unet_image_accelerate_workspace",
                "zprl.policy.flow_match_vib_unet_image_policy.FlowMatchVibUnetImagePolicy",
                "zprl.workspace.train_flow_match_vib_unet_image_accelerate_workspace.TrainFlowMatchVibUnetImageAccelerateWorkspace",
            ),
        ]
        for config_name, policy_target, workspace_target in cases:
            with self.subTest(config_name=config_name):
                with initialize_config_dir(
                        version_base=None, config_dir=str(CONFIG_DIR)):
                    cfg = compose(config_name=config_name)

                self.assertEqual(cfg._target_, workspace_target)
                self.assertEqual(cfg.policy._target_, policy_target)
                self.assertEqual(
                    cfg.policy.obs_encoder.crop_randomizer_version, "v3")
                self.assertFalse(cfg.training.resume)
                self.assertEqual(cfg.training.gradient_accumulate_every, 1)
                self.assertFalse(cfg.training.freeze_encoder)
                self.assertEqual(cfg.training.mixed_precision, "no")

    def test_unsupported_accelerate_options_are_rejected(self):
        workspace_classes = [
            TrainDiffusionImageAccelerateWorkspace,
            TrainFlowMatchVibUnetImageAccelerateWorkspace,
        ]
        cases = [
            (
                {
                    "resume": True,
                    "gradient_accumulate_every": 1,
                    "freeze_encoder": False,
                },
                NotImplementedError,
                "Resume is not supported",
            ),
            (
                {
                    "resume": False,
                    "gradient_accumulate_every": 2,
                    "freeze_encoder": False,
                },
                ValueError,
                "does not support gradient accumulation",
            ),
            (
                {
                    "resume": False,
                    "gradient_accumulate_every": 1,
                    "freeze_encoder": True,
                },
                ValueError,
                "does not support freezing the encoder",
            ),
        ]
        for workspace_cls in workspace_classes:
            for training, error_type, message in cases:
                with self.subTest(
                        workspace=workspace_cls.__name__, training=training):
                    workspace = workspace_cls.__new__(workspace_cls)
                    workspace.cfg = OmegaConf.create({"training": training})
                    with self.assertRaisesRegex(error_type, message):
                        workspace.run()

    def test_total_batch_size_split(self):
        self.assertEqual(_split_total_batch_size(256, 2, "dataloader"), 128)
        with self.assertRaisesRegex(ValueError, "must be divisible"):
            _split_total_batch_size(255, 2, "dataloader")
        with self.assertRaisesRegex(ValueError, "smaller than world_size"):
            _split_total_batch_size(1, 2, "dataloader")


if __name__ == "__main__":
    unittest.main()
