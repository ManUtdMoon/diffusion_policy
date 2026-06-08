from typing import Dict
import torch


def split_prev_action(obs_dict: Dict[str, torch.Tensor]):
    obs_dict = dict(obs_dict)
    prev_action = obs_dict.pop('prev_action', None)
    prev_action_valid_mask = obs_dict.pop('prev_action_valid_mask', None)
    return obs_dict, prev_action, prev_action_valid_mask


def get_prev_action_cond(prev_action, prev_action_valid_mask, n_prev_action_steps, action_normalizer):
    if n_prev_action_steps <= 0:
        return None
    nprev_action = action_normalizer.normalize(prev_action)
    mask = prev_action_valid_mask.to(device=nprev_action.device, dtype=nprev_action.dtype)
    nprev_action = nprev_action * mask.unsqueeze(-1)
    return nprev_action.flatten(start_dim=1)


def append_prev_action_cond(obs_emb, prev_action_cond):
    if prev_action_cond is None:
        return obs_emb
    return torch.cat([obs_emb, prev_action_cond], dim=-1)
