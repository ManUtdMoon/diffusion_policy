"""Frozen OneLeg base loading; live callers must import isaacgym before torch."""

import dill
import hydra
from omegaconf import OmegaConf
import torch


def load_oneleg_base(checkpoint, device='cpu', num_inference_steps=5):
    """Restore only the EMA policy and its normalizer, without a workspace/dataset."""
    OmegaConf.register_new_resolver('eval', eval, replace=True)
    with open(checkpoint, 'rb') as file:
        payload = torch.load(file, map_location='cpu', pickle_module=dill)
    cfg = payload['cfg']
    
    # check task
    if cfg.task_name != 'oneleg_image' or cfg.task.name != cfg.task_name:
        raise ValueError('Expected a oneleg_image base checkpoint')
    if cfg.horizon != 16 or cfg.n_obs_steps != 1:
        raise ValueError('OneLeg base must have horizon=16 and n_obs_steps=1')
    
    # check shapes
    shape = OmegaConf.to_container(cfg.shape_meta, resolve=True)
    expected = dict(
        color_image1=[3, 240, 320], color_image2=[3, 240, 320], robot_state=[16]
    )
    if (
        set(shape['obs']) != set(expected)
        or any(shape['obs'][key]['shape'] != value for key, value in expected.items())
        or shape['action']['shape'] != [10]
    ):
        raise ValueError('Incompatible OneLeg observation/action shapes')

    base = hydra.utils.instantiate(cfg.policy)
    base.load_state_dict(payload['state_dicts']['ema_model'], strict=True)
    if (
        base.horizon != 16 or base.n_obs_steps != 1 or base.action_dim != 10
        or not base.obs_as_global_cond or not 1 <= base.n_action_steps <= 16
    ):
        raise ValueError('Base policy interface differs from the OneLeg contract')

    # check normalizer
    for key, width in dict(action=10, robot_state=16, color_image1=1, color_image2=1).items():
        params = base.normalizer.params_dict[key]
        scale, offset = params['scale'], params['offset']
        if (
            scale.shape != (width,) or offset.shape != (width,)
            or not torch.isfinite(scale).all() or not torch.isfinite(offset).all()
            or torch.any(scale == 0)
        ):
            raise ValueError(f'Invalid base normalizer for {key}')
        if (
            key.startswith('color_image')
            and (not torch.all(scale == 1) or not torch.all(offset == 0))
        ):
            raise ValueError('OneLeg images must retain identity normalization')
    base.num_inference_steps = num_inference_steps
    return base.to(device).requires_grad_(False).eval(), cfg


def _base_sha256(checkpoint):
    import hashlib

    digest = hashlib.sha256()
    with open(checkpoint, 'rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def oneleg_online_snapshot(cfg, method, policy, global_step, global_update, num_episodes):
    """Evaluation snapshot: steering weights and an external, identified base."""
    import copy

    key = {'resrl': 'res_policy', 'dsrl': 'noise_policy'}[method]
    checkpoint = cfg.online_task.base_ckpt
    return {
        'cfg': copy.deepcopy(cfg), 'method': method,
        key: {name: value.detach().cpu().clone() for name, value in policy.state_dict().items()},
        'base_ckpt': checkpoint, 'base_sha256': _base_sha256(checkpoint),
        'global_step': global_step, 'global_update': global_update,
        'num_episodes': num_episodes,
    }


def load_oneleg_online(checkpoint, device='cuda:0'):
    """Reconstruct an evaluation policy without a training workspace or replay."""
    from zprl.policy.oneleg_history_policy import OneLegHistorySumPolicy, OneLegHistoryNoiseSumPolicy

    with open(checkpoint, 'rb') as file:
        payload = torch.load(file, map_location='cpu', pickle_module=dill)
    cfg, method = payload['cfg'], payload['method']
    key = {'resrl': 'res_policy', 'dsrl': 'noise_policy'}[method]
    if _base_sha256(payload['base_ckpt']) != payload['base_sha256']:
        raise ValueError('External OneLeg base differs from the snapshot reference')
    base, base_cfg = load_oneleg_base(payload['base_ckpt'], device, cfg.num_inference_steps)
    if (cfg.task_name != base_cfg.task_name or cfg.horizon != base.horizon
            or cfg.n_obs_steps != base.n_obs_steps or cfg.shape_meta.action.shape[0] != base.action_dim):
        raise ValueError('Online snapshot and base interfaces differ')
    obs_dim = base.obs_feature_dim * base.n_obs_steps
    common = dict(base_obs_emb_dim=obs_dim, subtask_dim=0, action_dim=base.action_dim,
                  n_action_steps=cfg.n_action_steps, base_policy=base)
    if method == 'resrl':
        policy = hydra.utils.instantiate(cfg.res_policy, obs_dim=obs_dim,
                                        action_dim=cfg.n_rl_steps * base.action_dim).to(device)
        combined = OneLegHistorySumPolicy(**common, res_scale=cfg.training.res_scale,
            n_rl_steps=cfg.n_rl_steps, res_policy=policy)
    else:
        policy = hydra.utils.instantiate(cfg.noise_policy, obs_dim=obs_dim,
            action_dim=cfg.n_noise_steps * base.action_dim,
            exec_dim=cfg.n_action_steps * base.action_dim).to(device)
        combined = OneLegHistoryNoiseSumPolicy(**common, noise_scale=cfg.training.noise_scale,
            horizon=cfg.horizon, n_obs_steps=cfg.n_obs_steps,
            n_noise_steps=cfg.n_noise_steps, noise_policy=policy)
    policy.load_state_dict(payload[key], strict=True)
    policy.requires_grad_(False)
    combined.eval()
    metadata = {name: payload[name] for name in (
        'method', 'base_ckpt', 'base_sha256', 'global_step', 'global_update', 'num_episodes')}
    return combined, cfg, metadata
