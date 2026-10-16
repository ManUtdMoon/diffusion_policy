import pathlib
import unittest

import hydra
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import torch
from torchvision.transforms import v2 as T
from torchvision.transforms import functional as TF
from zprl.model.vision.model_getter import ResNet18Pooling, SpatialLearnedEmbeddings


class OneLegInterfacesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        OmegaConf.register_new_resolver('eval', eval, replace=True)
        with initialize_config_dir(version_base=None,
                config_dir=str(pathlib.Path(__file__).parents[1] / 'zprl/config')):
            cls.cfg = compose(config_name='train_flow_match_oneleg_image_workspace')
        cls.encoder = hydra.utils.instantiate(cls.cfg.policy.obs_encoder)

    def test_job_contract(self):
        cfg = self.cfg
        self.assertEqual(cfg.horizon, 16)
        self.assertEqual(cfg.n_obs_steps, 1)
        self.assertEqual(cfg.task.dataset.pad_after, 15)
        self.assertEqual(cfg.policy.down_dims, [256, 384, 512])
        self.assertFalse(cfg.training.enable_rollout)
        self.assertEqual(cfg.task.env_runner.n_rollouts, 100)
        self.assertEqual(cfg.task.env_runner.n_envs, 20)
        self.assertEqual(cfg.task.env_runner.render_size, [640, 360])
        self.assertFalse(cfg.task.env_runner.record_video)
        self.assertEqual(cfg.task.env_runner.n_vis, 5)
        self.assertEqual(cfg.training.seed, 10)
        self.assertEqual(cfg.task.dataset.seed, cfg.training.seed)
        self.assertEqual(cfg.task.env_runner.seed, 10000)
        self.assertEqual(cfg.training.rollout_every, 10)
        self.assertEqual(cfg.training.ddp_timeout_seconds, 1800)
        self.assertEqual(cfg.n_action_steps, 8)
        self.assertEqual(cfg.policy.num_inference_steps, 5)
        self.assertEqual(cfg.checkpoint.topk.k, 3)
        self.assertEqual(cfg.checkpoint.topk.monitor_key, 'val_action_mse')
        self.assertTrue(cfg.val_dataloader.shuffle)
        self.assertTrue(cfg.val_dataloader.drop_last)
        self.assertNotIn('backend', cfg.training)
        self.assertNotIn('workspace_targets', cfg)
        self.assertEqual(cfg.training.mixed_precision, 'no')
        self.assertEqual(cfg.training.gradient_clip_val, 1.0)
        self.assertEqual(cfg._target_,
            'zprl.workspace.train_diffusion_image_accelerate_workspace.TrainDiffusionImageAccelerateWorkspace')

    def test_optional_precision_configuration(self):
        with initialize_config_dir(version_base=None,
                config_dir=str(pathlib.Path(__file__).parents[1] / 'zprl/config')):
            cfg = compose(config_name='train_flow_match_oneleg_image_workspace',
                overrides=['training.mixed_precision=bf16'])
        self.assertEqual(cfg._target_,
            'zprl.workspace.train_diffusion_image_accelerate_workspace.TrainDiffusionImageAccelerateWorkspace')
        self.assertEqual(cfg.training.mixed_precision, 'bf16')
        self.assertFalse(cfg.training.enable_rollout)

    def test_camera_models_are_independent_random_and_spatial(self):
        encoder = self.encoder
        first, second = [encoder.key_model_map[key] for key in encoder.rgb_keys]
        self.assertIsInstance(first, ResNet18Pooling)
        self.assertIsInstance(first.pooling, SpatialLearnedEmbeddings)
        self.assertEqual(first.pooling.num_spatial_blocks, 8)
        self.assertEqual(first.pooling.bottleneck_dim,
            self.cfg.obs_encoder.rgb_model.color_image1.bottleneck_dim)
        self.assertEqual(tuple(first.pooling.embeddings.shape), (512, 7, 7, 8))
        self.assertFalse(any(isinstance(m, torch.nn.BatchNorm2d)
                             for m in encoder.modules()))
        self.assertTrue(any(isinstance(m, torch.nn.GroupNorm)
                            for m in encoder.modules()))
        a, b = first.resnet_base[0].weight, second.resnet_base[0].weight
        self.assertNotEqual(a.data_ptr(), b.data_ptr())
        self.assertFalse(torch.equal(a, b))

    def test_author_camera_transforms_and_eval_determinism(self):
        encoder = self.encoder
        image = torch.linspace(0, 1, 240 * 320).reshape(1, 1, 240, 320).expand(1, 3, -1, -1)
        normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        encoder.eval()
        torch.testing.assert_close(encoder.key_transform_map['color_image1'](image),
            normalize(TF.resize(image, [224, 224], antialias=True)))
        torch.testing.assert_close(encoder.key_transform_map['color_image2'](image),
            normalize(TF.center_crop(image, [224, 224])))
        for key in encoder.rgb_keys:
            transform = encoder.key_transform_map[key]
            torch.testing.assert_close(transform(image), transform(image))
            author_transform = transform[0].train_transform.transforms
            self.assertIsInstance(author_transform[0], T.ColorJitter)
            self.assertIsInstance(author_transform[1], T.GaussianBlur)
            if key == 'color_image2':
                self.assertEqual(author_transform[2].size, (240, 280))
                self.assertIsInstance(author_transform[3], T.RandomCrop)
        encoder.train()
        first = encoder.key_transform_map['color_image2'](image)
        second = encoder.key_transform_map['color_image2'](image)
        self.assertEqual(tuple(first.shape), (1, 3, 224, 224))
        self.assertFalse(torch.equal(first, second))

    def test_encoder_feature_shape_and_legacy_configuration(self):
        encoder = self.encoder.eval()
        obs = {key: torch.zeros(2, *attr['shape'])
               for key, attr in self.cfg.shape_meta.obs.items()}
        with torch.no_grad():
            result = encoder(obs)
        expected_dim = 2 * self.cfg.obs_encoder.rgb_model.color_image1.bottleneck_dim + 16
        self.assertEqual(tuple(result.shape), (2, expected_dim))
        torch.testing.assert_close(result[:, -16:], obs['robot_state'])
        with initialize_config_dir(version_base=None,
                config_dir=str(pathlib.Path(__file__).parents[1] / 'zprl/config')):
            legacy = compose(config_name='train_flow_match_unet_image_workspace')
        self.assertNotIn('rgb_transforms', legacy.policy.obs_encoder)
        old_encoder = hydra.utils.instantiate(legacy.policy.obs_encoder)
        self.assertEqual(old_encoder.output_shape()[0], 521)


if __name__ == '__main__':
    unittest.main()
