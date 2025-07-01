import sys
import os
import pathlib

ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import click
import dill
import pickle
from copy import deepcopy
import numpy as np
import torch
import hydra
from tqdm import tqdm
import shutil
import random
import json
import wandb
from sklearn.metrics import confusion_matrix

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.policy.flow_match_unet_image_policy import FlowMatchUnetImagePolicy
from diffusion_policy.env_runner.robomimic_image_runner_with_detector import RobomimicImageRunnerWithDetector

from rnd.model import RND

@click.command()
@click.option('--rnd_ckpt', '-c', required=True,
    help='Path to trained RND checkpoint')
@click.option('--n_action_steps', '-Ta', default=4, type=int, 
    help='Action horizon (number of action steps to execute)')
@click.option('--device', '-d', default='cuda:0')
@click.option('--mod_type', '-m', default='const', type=str)
@click.option('--confidence_interval', '-ci', default=0.95, type=float)
@click.option('--src', '-s', default='rollout', type=str)
def main(rnd_ckpt, n_action_steps, device, mod_type, confidence_interval, src):
    """
    This script initialize a env_runner, a rnd and a base policy.
    During running the policy, rnd_scores of obs_emb are collected.
    
    Then the calibrated band is tested for TPR and TNR.
    """
    # 1. load stuff
    device = torch.device(device)
    output_dir = str(pathlib.Path(rnd_ckpt).parent / 'test')
    calib_dir = str(pathlib.Path(rnd_ckpt).parent / 'calib')
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 1.1 load rnd
    print(f"Loading RND checkpoint from {rnd_ckpt}")
    rnd_payload = torch.load(open(rnd_ckpt, 'rb'), pickle_module=dill)
    rnd_cfg = deepcopy(rnd_payload['config'])
    rnd = RND(
        input_dim=rnd_cfg['input_dim'],
        hidden_dims=rnd_cfg['hidden_dims'],
        output_dim=rnd_cfg['output_dim'],
    )
    rnd.load_state_dict(rnd_payload['model'])
    rnd.to(device)
    rnd.eval()
    rnd.requires_grad_(False)

    # 1.2 load base_policy
    policy_ckpt = rnd_cfg['policy']
    print(f"Loading base policy from {policy_ckpt}")
    base_payload = torch.load(open(policy_ckpt, 'rb'), pickle_module=dill)
    base_cfg = base_payload['cfg']
    base_cfg.policy.n_action_steps = n_action_steps
    base: FlowMatchUnetImagePolicy = hydra.utils.instantiate(base_cfg.policy)
    base.load_state_dict(base_payload['state_dicts']['ema_model'])
    base.to(device)
    base.eval()
    base.requires_grad_(False)

    seed = base_cfg.training.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

    # 1.3 load band
    task = base_cfg.task_name
    CI = confidence_interval
    calib_data_path = pathlib.Path(calib_dir) / f'{src}' / f'{task}_calib_band_{CI}_{mod_type}.pkl'
    print(f"Loading calibrated band from {calib_data_path}")
    calib_data = pickle.load(open(calib_data_path, 'rb'))
    
    # 2. run env_runner
    env_cfg = deepcopy(base_cfg.task.env_runner)
    env_cfg._target_ = 'diffusion_policy.env_runner.robomimic_image_runner_with_detector.RobomimicImageRunnerWithDetector'
    env_cfg.n_train = 0
    env_cfg.n_train_vis = 0
    env_cfg.n_test = 200
    env_cfg.n_test_vis = 50
    env_cfg.test_start_seed = 100_000
    env_cfg.n_envs = 50
    env_cfg.n_action_steps = n_action_steps
    env_runner = hydra.utils.instantiate(env_cfg, output_dir=output_dir)

    test_log = env_runner.run_with_detector(
        policy=base,
        detector=rnd,
    )

    # 3. calculate TPR and TNR
    pred_ood_perstep = test_log['rnd_scores'] > calib_data['bound']
    pred_ood_pertraj = pred_ood_perstep.any(axis=-1) # (n_test,)

    tn, fp, fn, tp = confusion_matrix(
        test_log['failure'], pred_ood_pertraj
    ).ravel().tolist()

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    print(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(f"TPR: {tpr:.4f}, TNR: {tnr:.4f}")

    # 4. save results
    json_log = dict()
    for key, value in test_log.items():
        if isinstance(value, wandb.sdk.data_types.video.Video):
            json_log[key] = value._path
        else:
            json_log[key] = value
    del json_log['rnd_scores']

    scores_path = pathlib.Path(output_dir) / f'{task}_rnd_scores_{CI}_{mod_type}.pkl'
    with open(scores_path, 'wb') as f:
        pickle.dump({"rnd_scores": test_log['rnd_scores']}, f)

    json_log['failure'] = np.array(test_log['failure'], dtype=bool).tolist()
    json_log['failure_pred'] = pred_ood_pertraj.tolist()
    json_log['tpr'] = tpr
    json_log['tnr'] = tnr
    json_log['cm'] = {
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'tp': tp
    }

    out_path = os.path.join(output_dir, 'eval_log.json')
    json.dump(json_log, open(out_path, 'w'), indent=2, sort_keys=True)

    # 5 print FN and FP indices in the first n_test_vis
    n_test_vis = env_cfg.n_test_vis
    vis_label = np.array(test_log['failure'][:n_test_vis], dtype=bool)
    vis_pred = np.array(pred_ood_pertraj[:n_test_vis], dtype=bool)

    print("False Negatives (FN) indices in the first n_test_vis:")
    fn_indices = np.where(np.logical_and(vis_label, np.logical_not(vis_pred)))[0]
    print(fn_indices)

    print("False Positives (FP) indices in the first n_test_vis:")
    fp_indices = np.where(np.logical_and(np.logical_not(vis_label), vis_pred))[0]
    print(fp_indices)


if __name__ == "__main__":
    main()
