import traceback
import pathlib

import dill
import hydra
import numpy as np
import random
import torch
import zmq
from omegaconf import OmegaConf

from zprl.common.pytorch_util import dict_apply
from zprl.policy.base_image_policy import BaseImagePolicy
from zprl.policy.latent_policy import SumPolicy as LatentSumPolicy
from zprl.policy.residue_policy import SumPolicy as ActionSumPolicy
from zprl.workspace.base_workspace import BaseWorkspace


def patch_legacy_cfg(cfg):
    cfg_yaml = OmegaConf.to_yaml(cfg)
    if 'diffusion_policy' in cfg_yaml:
        cfg = OmegaConf.create(cfg_yaml.replace('diffusion_policy', 'zprl'))
    return cfg


def set_deterministic(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_base_policy(checkpoint, device, n_action_steps, num_inference_steps):
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = patch_legacy_cfg(payload['cfg'])

    set_deterministic(cfg.training.seed)

    cfg.n_action_steps = n_action_steps
    cfg.policy.n_action_steps = n_action_steps
    cfg.task.env_runner.n_action_steps = n_action_steps
    cfg.task.dataset.pad_after = n_action_steps - 1

    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=None)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    policy.num_inference_steps = num_inference_steps

    device = torch.device(device)
    policy.to(device)
    policy.eval()
    policy.requires_grad_(False)
    return policy


def load_sum_policy(checkpoint, device, base_ckpt=None):
    device = torch.device(device)
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = patch_legacy_cfg(payload['cfg'])
    res_policy_state_dict = payload['res_policy']

    Ta = int(cfg.n_action_steps)
    To = int(cfg.n_obs_steps)

    set_deterministic(cfg.training.seed)

    if base_ckpt is not None:
        cfg.online_task.base_ckpt = base_ckpt
        print(f"Using base policy from cli argument: {base_ckpt}")
    if (cfg.online_task.base_ckpt is None) or not pathlib.Path(cfg.online_task.base_ckpt).exists():
        raise ValueError(
            f"Base policy not specified and not found at {cfg.online_task.base_ckpt}. "
            "Please download the base ckpt and specify the correct path using --base_ckpt.")

    base_payload = torch.load(open(cfg.online_task.base_ckpt, 'rb'), pickle_module=dill)
    base_cfg = patch_legacy_cfg(base_payload['cfg'])
    assert base_cfg.task_name == cfg.task_name, \
        f"Base policy task {base_cfg.task_name} does not match current task {cfg.task_name}"

    base_cfg.n_action_steps = Ta
    base_cfg.policy.n_action_steps = Ta
    base_cfg.task.env_runner.n_action_steps = Ta
    base_cfg.task.dataset.pad_after = Ta - 1
    num_inference_steps = OmegaConf.select(cfg, 'num_inference_steps', default=2)
    base_cfg.policy.num_inference_steps = num_inference_steps
    base_policy: BaseImagePolicy = hydra.utils.instantiate(base_cfg.policy)
    base_policy.load_state_dict(base_payload['state_dicts']['ema_model'])
    base_policy.eval()
    base_policy.requires_grad_(False)
    base_policy.to(device)
    print(f"Loaded base policy from {cfg.online_task.base_ckpt}")

    cfg.n_action_steps = Ta
    do = int(base_policy.obs_feature_dim)
    Do = int(To * do)
    da = int(cfg.shape_meta.action.shape[0])
    Da = int(Ta * da)
    res_target = str(cfg.res_policy._target_)

    if "latent_policy" in res_target:
        z_dim = int(base_policy.vib_latent_dim)
        res_policy = hydra.utils.instantiate(
            cfg.res_policy, obs_dim=Do, z_dim=z_dim, action_dim=Da)
        sum_policy = LatentSumPolicy(
            res_scale=cfg.training.res_scale,
            base_policy=base_policy,
            res_policy=res_policy,
        )
        print(f"ZPRL (latent) with To={To}, do={do}, Do={Do}, "
              f"Ta={Ta}, da={da}, Da={Da}, z_dim={z_dim}")
    elif "residue_policy" in res_target:
        res_policy = hydra.utils.instantiate(
            cfg.res_policy, obs_dim=do, action_dim=Da)
        sum_policy = ActionSumPolicy(
            res_scale=cfg.training.res_scale,
            obs_emb_dim=do,
            action_dim=da,
            n_action_steps=Ta,
            base_policy=base_policy,
            res_policy=res_policy,
        )
        print(f"ResRL (action) with To={To}, do={do}, Ta={Ta}, da={da}, Da={Da}")
    else:
        raise ValueError(f"Unknown res_policy target: {res_target}")

    res_policy.load_state_dict(res_policy_state_dict)
    res_policy.eval()
    res_policy.requires_grad_(False)
    res_policy.to(device)
    sum_policy.eval()
    return sum_policy


def run_policy_server(policy, bind_addr):
    context = zmq.Context.instance()
    socket = context.socket(zmq.REP)
    socket.bind(bind_addr)
    print(f"Policy server listening on {bind_addr}")

    while True:
        raw_request = socket.recv()
        request = dill.loads(raw_request)
        request_id = request.get('request_id')
        request_type = request.get('type')
        try:
            payload = request.get('payload', {})
            if request_type == 'ping':
                response_payload = {'message': 'pong'}
            elif request_type == 'reset':
                policy.reset()
                response_payload = {}
            elif request_type == 'shutdown':
                response_payload = {}
            elif request_type == 'predict_action':
                obs_dict = dict_apply(
                    payload['obs'], lambda x: x.to(device=policy.device) if isinstance(x, torch.Tensor) else x)
                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict)
                keep_keys = ('action', 'naction', 'action_pred', 'naction_pred')
                action_dict = {
                    k: v.detach().to('cpu') if isinstance(v, torch.Tensor) else v
                    for k, v in action_dict.items()
                    if k in keep_keys
                }
                response_payload = {'action_dict': action_dict}
            else:
                raise ValueError(f"Unknown request type: {request_type}")

            response = {
                'type': 'ok',
                'request_id': request_id,
                'payload': response_payload,
                'error': None,
            }
        except Exception:
            response = {
                'type': 'error',
                'request_id': request_id,
                'payload': {},
                'error': traceback.format_exc(),
            }

        socket.send(dill.dumps(response))
        if request_type == 'shutdown' and response['type'] == 'ok':
            break

    socket.close(linger=0)
