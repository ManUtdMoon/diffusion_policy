import sys
import os
import pathlib

ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import click
import torch
import dill
import hydra
import numpy as np
import random
from copy import deepcopy
from torch.utils.data import DataLoader
from tqdm import tqdm

from diffusion_policy.policy.flow_match_unet_image_policy import FlowMatchUnetImagePolicy
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.pytorch_util import dict_apply


@click.command()
@click.option('--checkpoint', '-c', required=True, help='Path to trained flow_match policy checkpoint')
@click.option('--output', '-o', required=True, help='Output path to store observation embeddings')
@click.option('--batch-size', '-b', default=64, help='Batch size for processing')
@click.option('--num-workers', '-w', default=8, help='Number of workers for data loading')
@click.option('--device', '-d', default='cuda:0', help='Device to use for inference')
def main(
        checkpoint,
        output,
        batch_size,
        num_workers,
        device):
    """
    Extract observation embeddings from a trained flow_match policy.
    
    This script loads a trained FlowMatchUnetImagePolicy and the corresponding dataset,
    then uses the policy to predict observation embeddings (obs_cond) for all data
    samples and stores them.
    """
    # 1. create output directory if it doesn't exist
    pathlib.Path(output).mkdir(parents=True, exist_ok=True)
    
    # 2. Load checkpoint, extract cfg, dataset, and policy
    device = torch.device(device)
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    state_dicts = payload['state_dicts']

    # 2.1 deterministic mode
    seed = cfg.training.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

    # 2.2 load dataset
    dataset: BaseImageDataset = hydra.utils.instantiate(cfg.task.dataset)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False
    )
    
    # 2.3 Instantiate policy
    policy: FlowMatchUnetImagePolicy = hydra.utils.instantiate(cfg.policy)
    policy.load_state_dict(payload['state_dicts']['ema_model'])
    policy.eval()
    policy.requires_grad_(False)
    policy.to(device)

    print(f"Dataset loaded. Total samples: {len(dataset)}")
    print(f"Processing {len(dataloader)} batches...")
    
    # 3. Collect observation embeddings
    obs_embs = []
    actions = []
    sample_indices = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Extracting obs embeddings")):
            batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))

            # only use obs_encoder
            nobs = policy.normalizer.normalize(batch['obs'])
            B, To = next(iter(nobs.values())).shape[:2]
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            obs_emb = policy.obs_encoder(this_nobs) # (B*To, do)
            obs_emb = obs_emb.reshape(B, -1) # (B, Do=To*do)

            # action process
            nactions = policy.normalizer['action'].normalize(batch['action'])

            # Move to CPU and store
            obs_emb_cpu = obs_emb.cpu()
            obs_embs.append(obs_emb_cpu)
            actions.append(nactions.cpu())

            # Keep track of sample indices
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(dataset))
            sample_indices.extend(range(start_idx, end_idx))
    
    print(f'obs_embeddings shape: {obs_embs[0].shape} (B, Do)')
    print(f'actions shape: {actions[0].shape} (B, H, Da)')
    
    # 4. Concatenate all embeddings
    obs_embs = torch.cat(obs_embs, dim=0)
    print(f'Total obs_embeddings shape: {obs_embs.shape} (N, Do)')
    actions = torch.cat(actions, dim=0)
    print(f'Total actions shape: {actions.shape} (N, H, Da)')

    # 5. Save results
    task = cfg.task.name
    num_demo = cfg.task.dataset.num_demo
    policy_type = 'flow' if 'flow' in cfg.name else 'diffusion'
    file_name = f'{output}/{task}_{num_demo}_{policy_type}_obs_emb_action.pt'
    torch.save({
        'obs_emb': obs_embs,
        'action': actions,
        'checkpoint': checkpoint,
    }, file_name)


if __name__ == "__main__":
    main()
