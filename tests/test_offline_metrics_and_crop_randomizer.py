import unittest
from unittest.mock import patch

import torch
import torch.nn as nn

from zprl.common.action_mse_util import action_mse_per_sample
from zprl.model.vision.crop_randomizer import CropRandomizerV2, CropRandomizerV3
from zprl.model.vision.multi_image_obs_encoder import MultiImageObsEncoder


def _make_encoder(**kwargs):
    return MultiImageObsEncoder(
        shape_meta={
            "obs": {
                "camera": {
                    "shape": [3, 6, 6],
                    "type": "rgb",
                },
            },
        },
        rgb_model=nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 4 * 4, 5),
        ),
        crop_shape=(4, 4),
        random_crop=True,
        **kwargs,
    )


def _crop_randomizer(encoder):
    return encoder.key_transform_map["camera"][1]


class OfflineMetricsAndCropRandomizerTest(unittest.TestCase):
    def test_action_mse_per_sample_groups_and_open_end(self):
        pred_action = torch.zeros(2, 2, 4)
        gt_action = torch.tensor([
            [[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]],
            [[2.0, 4.0, 6.0, 8.0], [2.0, 4.0, 6.0, 8.0]],
        ])
        groups = {
            "pos_rot": [[0, 1], [2, 3]],
            "gripper": [[3, None]],
        }

        metrics = action_mse_per_sample(pred_action, gt_action, groups)

        torch.testing.assert_close(metrics[""], torch.tensor([7.5, 30.0]))
        torch.testing.assert_close(metrics["pos_rot"], torch.tensor([5.0, 20.0]))
        torch.testing.assert_close(metrics["gripper"], torch.tensor([16.0, 64.0]))

    def test_action_mse_per_sample_rejects_invalid_range(self):
        for invalid_range in [[-1, 1], [0, 5], [2, 2], [3, 2]]:
            with self.subTest(invalid_range=invalid_range):
                with self.assertRaisesRegex(ValueError, "Invalid action MSE range"):
                    action_mse_per_sample(
                        torch.zeros(1, 2, 4),
                        torch.zeros(1, 2, 4),
                        {"invalid": [invalid_range]},
                    )

    def test_multi_image_obs_encoder_defaults_to_v2_and_selects_v3(self):
        fallback_encoder = _make_encoder()
        v3_encoder = _make_encoder(crop_randomizer_version="v3")

        self.assertIsInstance(_crop_randomizer(fallback_encoder), CropRandomizerV2)
        self.assertIsInstance(_crop_randomizer(v3_encoder), CropRandomizerV3)

    def test_multi_image_obs_encoder_rejects_unknown_crop_randomizer(self):
        with self.assertRaisesRegex(ValueError, "crop_randomizer_version"):
            _make_encoder(crop_randomizer_version="v4")

    def test_crop_randomizer_v3_train_eval_and_force_random(self):
        randomizer = CropRandomizerV3(
            input_shape=(1, 4, 4),
            crop_height=2,
            crop_width=2,
        )
        inputs = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)

        def last_start(low, high, size, device=None, **kwargs):
            return torch.full(size, high - 1, dtype=torch.long, device=device)

        with patch("torch.randint", side_effect=last_start):
            randomizer.train()
            torch.testing.assert_close(
                randomizer(inputs), inputs[..., 2:4, 2:4], atol=1e-5, rtol=0)

            randomizer.eval()
            torch.testing.assert_close(
                randomizer(inputs), inputs[..., 1:3, 1:3], atol=1e-5, rtol=0)

            randomizer.force_random_crop = True
            torch.testing.assert_close(
                randomizer(inputs), inputs[..., 2:4, 2:4], atol=1e-5, rtol=0)

        self.assertEqual(list(randomizer.parameters()), [])
        self.assertEqual(list(randomizer.buffers()), [])

    def test_crop_randomizer_v3_shares_5d_shift_within_each_batch(self):
        randomizer = CropRandomizerV3(
            input_shape=(1, 4, 4),
            crop_height=2,
            crop_width=2,
        )
        inputs = torch.cat([
            torch.arange(16, dtype=torch.float32).reshape(1, 1, 1, 4, 4),
            torch.arange(16, dtype=torch.float32).reshape(1, 1, 1, 4, 4) + 100,
        ]).expand(-1, 3, -1, -1, -1).clone()
        shift_h = torch.tensor([0, 2]).reshape(2, 1, 1, 1)
        shift_w = torch.tensor([1, 0]).reshape(2, 1, 1, 1)

        with patch("torch.randint", side_effect=[shift_h, shift_w]):
            output = randomizer(inputs)

        torch.testing.assert_close(output[0], inputs[0, :, :, 0:2, 1:3])
        torch.testing.assert_close(output[1], inputs[1, :, :, 2:4, 0:2])
        torch.testing.assert_close(output[0, 0], output[0, 1])
        torch.testing.assert_close(output[1, 0], output[1, 2])

    def test_crop_randomizer_version_does_not_change_state_dict_keys(self):
        v2_encoder = _make_encoder(crop_randomizer_version="v2")
        v3_encoder = _make_encoder(crop_randomizer_version="v3")

        self.assertEqual(tuple(v2_encoder.state_dict()), tuple(v3_encoder.state_dict()))


if __name__ == "__main__":
    unittest.main()
