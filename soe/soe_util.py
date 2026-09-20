import pathlib

import dill
import hydra
import torch
from omegaconf import OmegaConf

from zprl.common.online_util import get_crop_randomizers
from zprl.model.vib import VIBEncoder
from zprl.policy.flow_match_vib_unet_image_policy import FlowMatchVibUnetImagePolicy


SOE_TASKS = ('metaworld_box-close', 'adroit_hammer')


def load_soe_policy(checkpoint, task, device='cpu', n_action_steps=None,
        num_inference_steps=None):
    if task not in SOE_TASKS:
        raise ValueError(f"Unsupported SOE task: {task}")
    OmegaConf.register_new_resolver('eval', eval, replace=True)
    payload = torch.load(checkpoint, map_location='cpu', pickle_module=dill)
    cfg = OmegaConf.create(OmegaConf.to_yaml(payload['cfg']).replace(
        'diffusion_policy', 'zprl'))
    task_path = pathlib.Path(__file__).parents[1] / 'zprl' / 'config' / 'task' / f'{task}.yaml'
    expected = OmegaConf.create({'task': OmegaConf.load(task_path)}).task
    if cfg.task_name != expected.name or cfg.task.task_name != expected.task_name:
        raise ValueError(f"Checkpoint task {cfg.task_name!r} does not match {task!r}")
    if cfg.task.env_runner._target_ != expected.env_runner._target_:
        raise ValueError(f"Checkpoint runner does not match {task}")
    if cfg.policy.shape_meta != expected.shape_meta:
        raise ValueError(f"Checkpoint observation/action shapes do not match {task}")
    policy_cls = hydra.utils.get_class(cfg.policy._target_)
    if not issubclass(policy_cls, FlowMatchVibUnetImagePolicy):
        raise ValueError('SOE requires a FlowMatchVibUnetImagePolicy checkpoint')

    if n_action_steps is not None:
        cfg.policy.n_action_steps = n_action_steps
    if num_inference_steps is not None:
        cfg.policy.num_inference_steps = num_inference_steps
    weights_key = 'ema_model' if cfg.training.use_ema else 'model'
    if weights_key not in payload['state_dicts']:
        raise ValueError(f"Checkpoint is missing required {weights_key} weights")
    policy = hydra.utils.instantiate(cfg.policy)
    if not isinstance(policy.vib_encoder, VIBEncoder):
        raise ValueError('SOE requires a VIB encoder, not an AE encoder')
    if not 1 <= policy.n_action_steps <= policy.horizon - policy.n_obs_steps + 1:
        raise ValueError('n_action_steps must fit within the predicted action horizon')
    if policy.n_obs_steps < 1 or policy.num_inference_steps < 1:
        raise ValueError('Observation and inference step counts must be positive')
    policy.load_state_dict(payload['state_dicts'][weights_key])
    for key in ('image', 'agent_pos', 'action'):
        if key not in policy.normalizer.params_dict:
            raise ValueError(f"Checkpoint normalizer is missing {key}")
    policy.to(device)
    policy.eval()
    policy.requires_grad_(False)
    for crop in get_crop_randomizers(policy):
        crop.force_random_crop = False
    return policy, cfg, weights_key


class SoeBasePolicy:
    """Run the original observation condition without entering the VIB path."""
    def __init__(self, policy):
        self.policy = policy

    @property
    def device(self):
        return self.policy.device

    @property
    def dtype(self):
        return self.policy.dtype

    def reset(self):
        self.policy.reset()

    @torch.no_grad()
    def predict_action(self, obs_dict):
        return self.policy.conditional_predict(self.policy.encode_obs(obs_dict))


class SoeExplorationPolicy(SoeBasePolicy):
    def __init__(self, policy, exploration_alpha=2.0):
        super().__init__(policy)
        self.exploration_alpha = exploration_alpha

    @torch.no_grad()
    def predict_action(self, obs_dict):
        obs_emb = self.policy.encode_obs(obs_dict)
        mean, logvar = self.policy.vib_encoder(obs_emb)
        z = mean + self.exploration_alpha * torch.exp(0.5 * logvar) * torch.randn_like(mean)
        return self.policy.conditional_predict(self.policy.vib_decoder(z))
