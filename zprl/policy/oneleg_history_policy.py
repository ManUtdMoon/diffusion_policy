"""Fixed-row OneLeg sessions; native batches never autoreset individual rows."""

import torch

from zprl.policy.reactive_history_policy import HistorySumPolicy
from zprl.policy.reactive_history_noise_policy import HistoryNoiseSumPolicy


def _step_masks(policy, dones, infos, max_steps):
    batch_size = policy.current_obs_emb.shape[0]
    assert len(infos) == batch_size
    dones = torch.as_tensor(dones, device=policy.device, dtype=torch.bool)
    valid = torch.tensor([info['transition_valid'] for info in infos],
                         device=policy.device, dtype=torch.bool)
    lengths = torch.tensor([info['executed_action_steps'] for info in infos],
                           device=policy.device, dtype=torch.long)
    assert dones.shape == valid.shape == lengths.shape == (batch_size,)
    # assert torch.all((lengths[valid] > 0) & (lengths[valid] <= max_steps))
    # assert torch.all(lengths[~valid] == 0) and not torch.any(dones & ~valid)
    # assert not any(info.get('TimeLimit.truncated', False) and not done
    #                for info, done in zip(infos, dones.tolist()))
    return valid, lengths


def _advance_history(policy, next_obs_emb, valid, lengths):
    # Keep ended rows frozen, including their terminal endpoint, until batch reset.
    previous = policy.history.get()
    policy.history.complete(policy.pending['a_exec'], lengths)
    policy.history.append(next_obs_emb)
    for current, old in zip((policy.history.obs, policy.history.a_exec,
                             policy.history.obs_valid, policy.history.exec_steps), previous):
        current[~valid] = old[~valid]
    policy.current_obs_emb[valid] = next_obs_emb[valid].detach()


class OneLegHistorySumPolicy(HistorySumPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_policy.n_action_steps = self.n_action_steps

    @torch.no_grad()
    def on_env_step(self, next_obs_dict, dones, infos):
        if self.pending is None:
            raise RuntimeError('Call predict_action before on_env_step')
        valid, lengths = _step_masks(self, dones, infos, self.n_rl_steps)
        next_idx = self.seg_idx + valid.long()
        boundary = valid & (next_idx == self.n_segments)
        # All encoding/decoding retains B rows. OneLeg next_obs already contains
        # first terminal snapshots, so timeout never needs a reset observation.
        base_pred, next_emb = self.encode_obs(
            self._tensor_obs(next_obs_dict), predict_action=bool(boundary.any()))
        if boundary.any():
            self.base_cache[boundary] = base_pred['naction'][boundary].detach()
            next_idx[boundary] = 0
        next_base = self._base_segment(next_idx).flatten(start_dim=1)
        _advance_history(self, next_emb, valid, lengths)
        transition = {**self.pending, 'next_obs': next_emb.detach().clone(),
                      'next_a_base': next_base.detach().clone(),
                      'exec_steps': lengths, 'transition_valid': valid}
        self.seg_idx = next_idx
        self.pending = None
        return transition


class OneLegHistoryNoiseSumPolicy(HistoryNoiseSumPolicy):
    @torch.no_grad()
    def on_env_step(self, next_obs_dict, dones, infos):
        if self.pending is None:
            raise RuntimeError('Call predict_action before on_env_step')
        valid, lengths = _step_masks(self, dones, infos, self.n_action_steps)
        base_cond, next_emb = self.encode_obs(self._tensor_obs(next_obs_dict))
        # Preserve the complete normalized policy chunk, including unexecuted tail.
        _advance_history(self, next_emb, valid, lengths)
        self.current_base_cond[valid] = base_cond[valid].detach()
        transition = {**self.pending, 'next_obs': next_emb.detach().clone(),
                      'exec_steps': lengths, 'transition_valid': valid}
        self.pending = None
        return transition
