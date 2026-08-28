if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

from itertools import chain
import pathlib

import dill
import hydra
import torch
from omegaconf import OmegaConf

from zprl.workspace.train_flow_match_vib_unet_image_workspace import (
    TrainFlowMatchVibUnetImageWorkspace,
)


_SOURCE_POLICY_TARGET = (
    'zprl.policy.flow_match_unet_image_policy.FlowMatchUnetImagePolicy')
_COMPATIBILITY_FIELDS = (
    'shape_meta',
    'horizon',
    'n_obs_steps',
    'n_action_steps',
    'n_latency_steps',
    'dataset_obs_steps',
    'policy.shape_meta',
    'policy.obs_encoder',
    'policy.noise_scheduler',
    'policy.horizon',
    'policy.n_action_steps',
    'policy.n_obs_steps',
    'policy.num_inference_steps',
    'policy.obs_as_global_cond',
    'policy.diffusion_step_embed_dim',
    'policy.down_dims',
    'policy.kernel_size',
    'policy.n_groups',
    'policy.cond_predict_scale',
    'task.task_name',
    'task.dataset._target_',
    'task.dataset.horizon',
    'task.dataset.pad_before',
    'task.dataset.pad_after',
    'task.dataset.n_obs_steps',
    'task.dataset.seed',
)
_OPTIONAL_COMPATIBILITY_FIELDS = (
    'task.dataset_type',
    'task.abs_action',
    'task.dataset.shape_meta',
    'task.dataset.abs_action',
    'task.dataset.rotation_rep',
    'task.dataset.use_legacy_normalizer',
    'task.dataset.num_demo',
    'task.dataset.val_ratio',
    'task.dataset.max_train_episodes',
)
_DATASET_LOCATION_FIELDS = (
    'task.dataset.dataset_path',
    'task.dataset.zarr_path',
)


class TrainFlowMatchPosthocVibUnetImageWorkspace(
        TrainFlowMatchVibUnetImageWorkspace):
    def get_optimizer_parameters(self):
        return chain(
            self.model.vib_encoder.parameters(),
            self.model.vib_decoder.parameters())

    def compute_batch_loss(self, batch, model=None):
        if model is None:
            model = self.model
        return model.compute_vib_loss(batch)

    @staticmethod
    def _resolved_config_value(cfg, path):
        value = OmegaConf.select(cfg, path, default=None)
        if value is None:
            raise ValueError(f"Missing required config field '{path}'")
        if OmegaConf.is_config(value):
            return OmegaConf.to_container(value, resolve=True)
        return value

    def _validate_source_config(self, source_cfg):
        source_target = self._resolved_config_value(
            source_cfg, 'policy._target_')
        if source_target != _SOURCE_POLICY_TARGET:
            raise ValueError(
                f"Unsupported source policy target '{source_target}'; "
                f"expected '{_SOURCE_POLICY_TARGET}'")

        source_task = self._resolved_config_value(source_cfg, 'task_name')
        target_task = self._resolved_config_value(self.cfg, 'task_name')
        if source_task != target_task:
            raise ValueError(
                f"Source config mismatch for 'task_name': "
                f"source={source_task!r}, target={target_task!r}")

        source_global_cond = self._resolved_config_value(
            source_cfg, 'policy.obs_as_global_cond')
        if source_global_cond is not True:
            raise ValueError(
                'Source policy must use obs_as_global_cond=True')

        for path in _COMPATIBILITY_FIELDS:
            source_value = self._resolved_config_value(source_cfg, path)
            target_value = self._resolved_config_value(self.cfg, path)
            if source_value != target_value:
                raise ValueError(
                    f"Source config mismatch for '{path}': "
                    f"source={source_value!r}, target={target_value!r}")

        for path in _OPTIONAL_COMPATIBILITY_FIELDS:
            source_value = OmegaConf.select(source_cfg, path, default=None)
            target_value = OmegaConf.select(self.cfg, path, default=None)
            if source_value is None and target_value is None:
                continue
            if OmegaConf.is_config(source_value):
                source_value = OmegaConf.to_container(
                    source_value, resolve=True)
            if OmegaConf.is_config(target_value):
                target_value = OmegaConf.to_container(
                    target_value, resolve=True)
            if source_value != target_value:
                raise ValueError(
                    f"Source config mismatch for '{path}': "
                    f"source={source_value!r}, target={target_value!r}")

        location_found = False
        for path in _DATASET_LOCATION_FIELDS:
            source_path = OmegaConf.select(source_cfg, path, default=None)
            target_path = OmegaConf.select(self.cfg, path, default=None)
            if source_path is None and target_path is None:
                continue
            location_found = True
            source_dataset = (
                pathlib.Path(str(source_path)).name
                if source_path is not None else None)
            target_dataset = (
                pathlib.Path(str(target_path)).name
                if target_path is not None else None)
            if source_dataset != target_dataset:
                raise ValueError(
                    f"Source config mismatch for '{path}': "
                    f"source filename={source_dataset!r}, "
                    f"target filename={target_dataset!r}")
        if not location_found:
            raise ValueError(
                'Missing required dataset location; expected one of '
                f'{_DATASET_LOCATION_FIELDS}')

    def _initialize_from_source_payload(self, payload, source_path):
        if 'cfg' not in payload or 'state_dicts' not in payload:
            raise ValueError(
                f"Source checkpoint '{source_path}' is missing cfg/state_dicts")
        source_cfg = payload['cfg']
        self._validate_source_config(source_cfg)

        source_use_ema = self._resolved_config_value(
            source_cfg, 'training.use_ema')
        source_key = 'ema_model' if source_use_ema else 'model'
        if source_key not in payload['state_dicts']:
            raise ValueError(
                f"Source checkpoint '{source_path}' is missing "
                f"state_dicts['{source_key}']")

        expected_missing = {
            key for key in self.model.state_dict()
            if key.startswith('vib_encoder.')
            or key.startswith('vib_decoder.')
        }
        missing_keys, unexpected_keys = self.model.load_state_dict(
            payload['state_dicts'][source_key], strict=False)
        if set(missing_keys) != expected_missing or unexpected_keys:
            raise RuntimeError(
                'Unexpected source state dict keys: '
                f"missing={sorted(missing_keys)}, "
                f"expected_missing={sorted(expected_missing)}, "
                f"unexpected={sorted(unexpected_keys)}")

        self.model.requires_grad_(False)
        self.model.vib_encoder.requires_grad_(True)
        self.model.vib_decoder.requires_grad_(True)

        trainable_names = [
            name for name, param in self.model.named_parameters()
            if param.requires_grad
        ]
        if not trainable_names or not all(
                name.startswith('vib_encoder.')
                or name.startswith('vib_decoder.')
                for name in trainable_names):
            raise RuntimeError(
                f'Invalid post-hoc trainable parameters: {trainable_names}')
        optimizer_param_ids = {
            id(param)
            for group in self.optimizer.param_groups
            for param in group['params']
        }
        trainable_param_ids = {
            id(param) for param in self.model.parameters()
            if param.requires_grad
        }
        if optimizer_param_ids != trainable_param_ids:
            raise RuntimeError(
                'Post-hoc optimizer parameters do not match trainable VIB parameters')

        if self.ema_model is None:
            raise ValueError('Post-hoc VIB training requires training.use_ema=True')
        self.ema_model.load_state_dict(self.model.state_dict())

        self.source_checkpoint = str(source_path)
        self.source_state_key = source_key
        self.trainable_param_names = trainable_names
        self.trainable_param_count = sum(
            param.numel() for param in self.model.parameters()
            if param.requires_grad)
        self.frozen_param_count = sum(
            param.numel() for param in self.model.parameters()
            if not param.requires_grad)
        global_cond_dim = self.model.obs_feature_dim * self.model.n_obs_steps
        print(
            f"Initialized post-hoc VIB from {source_path} "
            f"using state_dicts['{source_key}']")
        print(
            f"Trainable parameters ({self.trainable_param_count / 1e6:.2f}M): "
            f"{self.trainable_param_names}")
        print(
            f"Frozen parameter count: {self.frozen_param_count / 1e6:.2f}M; "
            f"VIB dimensions: input={global_cond_dim}, "
            f"latent={self.model.vib_latent_dim}, output={global_cond_dim}")

    def _initialize_from_base_checkpoint(self):
        source_path = pathlib.Path(self.cfg.base_ckpt)
        with source_path.open('rb') as file:
            payload = torch.load(
                file, pickle_module=dill, map_location='cpu')
        self._initialize_from_source_payload(payload, str(source_path))

    def run(self):
        if self.cfg.training.resume:
            raise ValueError('Post-hoc VIB training does not support resume')
        if not self.cfg.training.use_ema:
            raise ValueError('Post-hoc VIB training requires training.use_ema=True')
        self._initialize_from_base_checkpoint()
        super().run()


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath('config')),
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainFlowMatchPosthocVibUnetImageWorkspace(cfg)
    workspace.run()


if __name__ == '__main__':
    main()
