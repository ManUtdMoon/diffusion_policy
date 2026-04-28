import pathlib
import random
import traceback

import dill
import hydra
import numpy as np
import torch
import zmq

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.policy.flow_match_vib_unet_image_policy import FlowMatchVibUnetImagePolicy
from diffusion_policy.policy.latent_policy import (
    ResiduePolicy as LatentResiduePolicy,
    SumPolicy as LatentSumPolicy,
)
from diffusion_policy.policy.residue_policy import ResiduePolicy as ActionResiduePolicy
from diffusion_policy.policy.sum_policy import SumPolicy as ActionSumPolicy


def set_deterministic(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_sum_policy(
    checkpoint,
    device,
    n_action_steps,
    num_inference_steps,
    base_ckpt=None,
):
    device = torch.device(device)
    payload = torch.load(open(checkpoint, "rb"), pickle_module=dill)
    cfg = payload["cfg"]
    res_policy_state_dict = payload["res_policy"]
    Ta = int(n_action_steps)

    print(f"load RL ckpt @ step = {payload['global_step']} from {checkpoint}")

    set_deterministic(int(cfg.training.seed))

    if base_ckpt is not None:
        cfg.online_task.base_ckpt = base_ckpt
        print(f"Using base policy from cli argument: {base_ckpt}")

    base_ckpt = cfg.online_task.base_ckpt
    if (base_ckpt is None) or not pathlib.Path(base_ckpt).exists():
        raise ValueError(
            f"Base policy not specified and not found at {base_ckpt}. "
            "Please specify it with --base_ckpt."
        )

    base_payload = torch.load(open(base_ckpt, "rb"), pickle_module=dill)
    base_cfg = base_payload["cfg"]
    To = int(base_cfg.n_obs_steps)

    base_task_name = base_cfg.task_name
    online_task_name = cfg.task_name
    assert base_task_name == online_task_name, (
        f"Base policy task {base_task_name} does not match current task {online_task_name}"
    )

    base_cfg.n_action_steps = Ta
    base_cfg.policy.n_action_steps = Ta
    base_cfg.policy.num_inference_steps = num_inference_steps
    base_cfg.task.dataset.pad_after = Ta - 1

    base_policy: FlowMatchVibUnetImagePolicy = hydra.utils.instantiate(base_cfg.policy)
    base_model_state = base_payload["state_dicts"]["ema_model"]
    base_policy.load_state_dict(base_model_state)
    base_policy.eval()
    base_policy.requires_grad_(False)
    base_policy.to(device)

    if hasattr(base_policy, "num_inference_steps"):
        base_policy.num_inference_steps = num_inference_steps
    if hasattr(base_policy, "n_action_steps"):
        base_policy.n_action_steps = Ta
    print(f"Loaded base policy from {base_ckpt}")

    cfg.n_action_steps = Ta
    do = int(base_policy.obs_feature_dim)
    Do = int(To * do)
    da = int(cfg.shape_meta.action.shape[0])
    Da = int(Ta * da)
    res_target = str(cfg.res_policy._target_)

    if "latent_policy" in res_target:
        z_dim = int(base_policy.vib_latent_dim)
        res_policy: LatentResiduePolicy = hydra.utils.instantiate(
            cfg.res_policy, obs_dim=Do, z_dim=z_dim, action_dim=Da
        )
        sum_policy = LatentSumPolicy(
            res_scale=cfg.training.res_scale,
            base_policy=base_policy,
            res_policy=res_policy,
        )
        print(
            f"Loaded latent residual policy with To={To}, do={do}, Do={Do}, "
            f"Ta={Ta}, da={da}, Da={Da}, z_dim={z_dim}"
        )
    else:
        res_policy: ActionResiduePolicy = hydra.utils.instantiate(
            cfg.res_policy, obs_dim=do, action_dim=Da
        )
        sum_policy = ActionSumPolicy(
            res_scale=cfg.training.res_scale,
            obs_emb_dim=do,
            action_dim=da,
            n_action_steps=Ta,
            base_policy=base_policy,
            res_policy=res_policy,
        )
        print(f"Loaded action residual policy with To={To}, do={do}, Ta={Ta}, da={da}, Da={Da}")

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
        request_id = request.get("request_id")
        request_type = request.get("type")
        try:
            payload = request.get("payload", {})
            if request_type == "ping":
                response_payload = {"message": "pong"}
            elif request_type == "reset":
                policy.reset()
                response_payload = {}
            elif request_type == "shutdown":
                response_payload = {}
            elif request_type == "predict_action":
                obs_dict = dict_apply(
                    payload["obs"],
                    lambda x: x.to(device=policy.device) if isinstance(x, torch.Tensor) else x,
                )
                rtc_context = payload.get("rtc_context", None)
                if rtc_context is not None:
                    rtc_context = dict_apply(
                        rtc_context,
                        lambda x: x.to(device=policy.device) if isinstance(x, torch.Tensor) else x,
                    )
                with torch.no_grad():
                    if rtc_context is None:
                        action_dict = policy.predict_action(obs_dict)
                    else:
                        action_dict = policy.predict_action(obs_dict, rtc_context=rtc_context)
                keep_keys = (
                    "action",
                    "naction",
                    "action_pred",
                    "naction_pred",
                    "action_pred_all",
                    "naction_pred_all",
                )
                action_dict = {
                    k: v.detach().to("cpu") if isinstance(v, torch.Tensor) else v
                    for k, v in action_dict.items()
                    if k in keep_keys
                }
                response_payload = {"action_dict": action_dict}
            else:
                raise ValueError(f"Unknown request type: {request_type}")

            response = {
                "type": "ok",
                "request_id": request_id,
                "payload": response_payload,
                "error": None,
            }
        except Exception:
            response = {
                "type": "error",
                "request_id": request_id,
                "payload": {},
                "error": traceback.format_exc(),
            }

        socket.send(dill.dumps(response))
        if request_type == "shutdown" and response["type"] == "ok":
            break

    socket.close(linger=0)
