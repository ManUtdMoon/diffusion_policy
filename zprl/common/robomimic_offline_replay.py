"""
Helper to load offline robomimic demo data into SB3 ReplayBuffer
for online residual RL pre-training.
"""
import os
import numpy as np
import torch
import zarr
from zprl.common.replay_buffer import ReplayBuffer as ZarrReplayBuffer
from zprl.common.sampler import SequenceSampler


def get_cached_robomimic_zarr_path(dataset_path: str, num_demo=None) -> str:
    """
    Derive the local zarr cache path from dataset_path and num_demo,
    consistent with RobomimicReplayImageDataset.
    """
    path = os.path.expanduser(dataset_path)
    return path + f'-n_{num_demo}' + '.zarr'


def _discounted_sum(rewards, gamma=0.99):
    return sum(gamma ** i * r for i, r in enumerate(rewards))


def load_robomimic_offline_data_into_replay_buffer(
    rb,                # SB3 ReplayBuffer
    dataset_path: str,
    base_policy,       # FlowMatchVibUnetImagePolicy (eval, frozen)
    base_cfg,          # OmegaConf config of the base (offline) policy
    cfg,               # OmegaConf config of the online training
    device,
    n_envs: int,
):
    """
    Load offline demo transitions into the SB3 replay buffer.

    The zarr must already exist locally and contain rewards/dones
    (via backfill_robomimic_reward_done_to_zarr.py).
    """
    from zprl.dataset.robomimic_image_util import get_shape_meta_obs_keys

    # ---- 0. resolve paths and config ----
    num_demo = base_cfg.task.dataset.num_demo
    zarr_path = get_cached_robomimic_zarr_path(dataset_path, num_demo)
    assert os.path.exists(zarr_path), \
        f"Offline zarr not found: {zarr_path}"

    n_obs_steps = cfg.n_obs_steps
    n_action_steps = cfg.n_action_steps
    res_scale = cfg.training.res_scale
    gamma = cfg.single_gamma
    seed = cfg.training.seed
    buffer_size = cfg.training.buffer_size
    batch_size_encode = cfg.training.get('preload_batch_size', 256)
    shape_meta = cfg.shape_meta

    rgb_keys, lowdim_keys = get_shape_meta_obs_keys(shape_meta)
    obs_keys = list(rgb_keys) + list(lowdim_keys)

    # ---- 1. load zarr into numpy-backed ZarrReplayBuffer ----
    needed_keys = obs_keys + ['action', 'rewards', 'dones']
    zarr_store = zarr.DirectoryStore(zarr_path)
    src_root = zarr.group(zarr_store)
    for k in ['action', 'rewards', 'dones']:
        assert k in src_root['data'], \
            f"Key '{k}' not found in zarr {zarr_path}"
    for k in obs_keys:
        assert k in src_root['data'], \
            f"Obs key '{k}' not found in zarr {zarr_path}"

    zarr_rb = ZarrReplayBuffer.copy_from_store(
        src_store=zarr_store, store=None, keys=needed_keys)
    print(f"[preload] Loaded zarr from {zarr_path}, "
          f"n_episodes={zarr_rb.n_episodes}, n_steps={zarr_rb.n_steps}")

    # ---- 2. build three samplers ----
    all_episode_mask = np.ones(zarr_rb.n_episodes, dtype=bool)

    obs_sampler = SequenceSampler(
        replay_buffer=zarr_rb,
        sequence_length=n_obs_steps,
        pad_before=n_obs_steps - 1,
        pad_after=0,
        keys=obs_keys,
        episode_mask=all_episode_mask,
    )
    action_sampler = SequenceSampler(
        replay_buffer=zarr_rb,
        sequence_length=n_action_steps,
        pad_before=0,
        pad_after=n_action_steps - 1,
        keys=['action'],
        episode_mask=all_episode_mask,
    )
    signal_sampler = SequenceSampler(
        replay_buffer=zarr_rb,
        sequence_length=n_action_steps,
        pad_before=0,
        pad_after=n_action_steps - 1,
        keys=['rewards', 'dones'],
        episode_mask=all_episode_mask,
    )

    n_total = len(obs_sampler)  # one per primitive step
    assert len(action_sampler) == n_total
    assert len(signal_sampler) == n_total

    # ---- 3. rebuild chunk-level transitions ----
    episode_ends = zarr_rb.episode_ends[:]
    episode_starts = np.concatenate([[0], episode_ends[:-1]])

    # pre-compute per-episode obs_sampler index offset so we can look up
    # next_obs by primitive step within episode
    # obs_sampler has exactly one index per primitive step (same order)
    # so obs_sampler.indices[t] corresponds to primitive step t

    # collect all obs_seq, next_obs_seq, demo_action_chunk, reward, done
    all_obs_seqs = []        # list of dicts
    all_next_obs_seqs = []   # list of dicts
    all_demo_actions = []    # (N, Ta, da)
    all_rewards = []         # (N,)
    all_dones = []           # (N,)

    for t in range(n_total):
        # obs
        obs_data = obs_sampler.sample_sequence(t)

        # action
        act_data = action_sampler.sample_sequence(t)
        demo_action_chunk = act_data['action']  # (Ta, da)

        # signal
        sig_data = signal_sampler.sample_sequence(t)
        reward_chunk = sig_data['rewards']  # (Ta,)
        done_chunk = sig_data['dones']      # (Ta,)

        # effective steps from action_sampler indices
        _, _, sample_start_idx, sample_end_idx = action_sampler.indices[t]

        # initial effective window
        window_dones = done_chunk[sample_start_idx:sample_end_idx]

        # truncate at first done=True to match MultiStepWrapper.step break semantics
        done_positions = np.where(window_dones > 0.5)[0]
        if len(done_positions) > 0:
            k = int(done_positions[0]) + 1  # include the done step itself
        else:
            k = sample_end_idx - sample_start_idx

        effective_rewards = reward_chunk[sample_start_idx:sample_start_idx + k]
        effective_dones = done_chunk[sample_start_idx:sample_start_idx + k]
        chunk_reward = _discounted_sum(effective_rewards, gamma)
        chunk_done = float(np.max(effective_dones))

        # next_obs: state after executing the last action in the chunk
        # obs[t] in zarr is the observation BEFORE action[t], so after action[t]
        # the resulting state is obs[t+1].
        buf_start = obs_sampler.indices[t][0]
        ep_idx = np.searchsorted(episode_ends, buf_start, side='right')
        ep_start = episode_starts[ep_idx]
        ep_end = episode_ends[ep_idx]
        ep_len = ep_end - ep_start
        prim_idx = buf_start - ep_start

        next_prim_idx = prim_idx + k
        if next_prim_idx < ep_len:
            # next obs exists in zarr; use sampler for correct multi-frame window
            next_sampler_idx = ep_start + next_prim_idx
            next_obs_data = obs_sampler.sample_sequence(next_sampler_idx)
        else:
            # truly at episode boundary, no obs[t+k] stored
            # build window like MultiStepWrapper: last n_obs_steps real frames, left-pad
            avail_start = max(ep_end - n_obs_steps, ep_start)
            n_avail = ep_end - avail_start
            next_obs_data = {}
            for key in obs_keys:
                arr = zarr_rb[key]
                frames = arr[avail_start:ep_end]  # (n_avail, ...)
                if n_avail < n_obs_steps:
                    pad = np.repeat(frames[0:1], n_obs_steps - n_avail, axis=0)
                    padded = np.concatenate([pad, frames], axis=0)
                else:
                    padded = frames
                next_obs_data[key] = padded

        all_obs_seqs.append(obs_data)
        all_next_obs_seqs.append(next_obs_data)
        all_demo_actions.append(demo_action_chunk)
        all_rewards.append(chunk_reward)
        all_dones.append(chunk_done)

    all_demo_actions = np.array(all_demo_actions, dtype=np.float32)  # (N, Ta, da)
    all_rewards = np.array(all_rewards, dtype=np.float32)           # (N,)
    all_dones = np.array(all_dones, dtype=np.float32)               # (N,)

    # stack obs dicts: each key -> (N, To, ...)
    def stack_obs_list(obs_list):
        result = {}
        for key in obs_keys:
            result[key] = np.stack([o[key] for o in obs_list], axis=0)
        return result

    all_obs_dict = stack_obs_list(all_obs_seqs)
    all_next_obs_dict = stack_obs_list(all_next_obs_seqs)

    print(f"[preload] Built {n_total} chunk-level transitions")

    # ---- 4. encode obs through base policy in batches ----
    # temporarily disable random crop
    from zprl.model.vision.crop_randomizer import CropRandomizerV2
    crop_randomizers = []
    for m in base_policy.modules():
        if isinstance(m, CropRandomizerV2):
            crop_randomizers.append(m)
    old_crop_modes = [m.force_random_crop for m in crop_randomizers]
    for m in crop_randomizers:
        m.force_random_crop = False

    def encode_obs_batched(obs_np_dict):
        """Encode all obs through base_policy, return obs_emb and base_naction."""
        N = next(iter(obs_np_dict.values())).shape[0]
        all_obs_emb = []
        all_base_naction = []
        for start in range(0, N, batch_size_encode):
            end = min(start + batch_size_encode, N)
            batch_obs = {}
            for key in obs_keys:
                arr = obs_np_dict[key][start:end]
                if key in rgb_keys:
                    # (B, To, H, W, C) -> (B, To, C, H, W), float32 /255
                    arr = np.moveaxis(arr, -1, 2).astype(np.float32) / 255.0
                else:
                    arr = arr.astype(np.float32)
                batch_obs[key] = torch.from_numpy(arr).to(device=device)
            with torch.no_grad():
                base_dict = base_policy.predict_action(batch_obs)
            all_obs_emb.append(base_dict['obs_emb'].cpu())
            all_base_naction.append(base_dict['naction'].cpu())
        return torch.cat(all_obs_emb, dim=0), torch.cat(all_base_naction, dim=0)

    print("[preload] Encoding obs through base policy...")
    obs_emb, base_naction = encode_obs_batched(all_obs_dict)
    print("[preload] Encoding next_obs through base policy...")
    next_obs_emb, base_next_naction = encode_obs_batched(all_next_obs_dict)

    # restore random crop
    for m, old_mode in zip(crop_randomizers, old_crop_modes):
        m.force_random_crop = old_mode

    # ---- 5. compute residual actions ----
    # normalize demo actions
    demo_naction = base_policy.normalizer['action'].normalize(
        torch.from_numpy(all_demo_actions).to(device=device)
    ).cpu()  # (N, Ta, da)

    demo_naction_flat = demo_naction.reshape(n_total, -1)              # (N, Da)
    base_naction_flat = base_naction.flatten(start_dim=1)              # (N, Da)
    base_next_naction_flat = base_next_naction.flatten(start_dim=1)    # (N, Da)
    res_naction_flat = (demo_naction_flat - base_naction_flat) / res_scale  # (N, Da)

    actions_to_save = torch.cat(
        [res_naction_flat, base_naction_flat, base_next_naction_flat],
        dim=-1
    ).numpy()  # (N, Da*3)

    obs_emb_np = obs_emb.numpy()          # (N, Do)
    next_obs_emb_np = next_obs_emb.numpy() # (N, Do)
    rewards_np = all_rewards               # (N,)
    dones_np = all_dones                   # (N,)

    # ---- 6. shuffle and write into SB3 replay buffer ----
    if cfg.training.preload_shuffle:
        rng = np.random.default_rng(seed=seed)
        perm = rng.permutation(n_total)
        obs_emb_np = obs_emb_np[perm]
        next_obs_emb_np = next_obs_emb_np[perm]
        actions_to_save = actions_to_save[perm]
        rewards_np = rewards_np[perm]
        dones_np = dones_np[perm]

    # truncate to buffer capacity (buffer_size is already the total capacity)
    max_transitions = buffer_size
    if n_total > max_transitions:
        print(f"[preload] Truncating {n_total} -> {max_transitions} (buffer_size)")
        obs_emb_np = obs_emb_np[:max_transitions]
        next_obs_emb_np = next_obs_emb_np[:max_transitions]
        actions_to_save = actions_to_save[:max_transitions]
        rewards_np = rewards_np[:max_transitions]
        dones_np = dones_np[:max_transitions]
        n_total = max_transitions

    # drop tail that doesn't fill a complete n_envs group
    n_tail = n_total % n_envs
    if n_tail > 0:
        print(f"[preload] Discarding {n_tail} tail transitions (not a full n_envs group)")
        n_total = n_total - n_tail
        obs_emb_np = obs_emb_np[:n_total]
        next_obs_emb_np = next_obs_emb_np[:n_total]
        actions_to_save = actions_to_save[:n_total]
        rewards_np = rewards_np[:n_total]
        dones_np = dones_np[:n_total]

    num_add_calls = n_total // n_envs

    # reshape to (num_add_calls, n_envs, ...)
    obs_emb_np = obs_emb_np.reshape(num_add_calls, n_envs, -1)
    next_obs_emb_np = next_obs_emb_np.reshape(num_add_calls, n_envs, -1)
    actions_to_save = actions_to_save.reshape(num_add_calls, n_envs, -1)
    rewards_np = rewards_np.reshape(num_add_calls, n_envs)
    dones_np = dones_np.reshape(num_add_calls, n_envs)

    for i in range(num_add_calls):
        rb.add(
            obs=obs_emb_np[i],
            next_obs=next_obs_emb_np[i],
            action=actions_to_save[i],
            reward=rewards_np[i],
            done=dones_np[i],
            infos=[{} for _ in range(n_envs)],
        )

    print(f"[preload] Wrote {n_total} transitions into replay buffer "
          f"({num_add_calls} add calls x {n_envs} n_envs)")
    print(f"[preload] Buffer size after preload: {rb.size()}")
    print(f"[preload] obs shape: {obs_emb_np.shape[1:]}, "
          f"action shape: {actions_to_save.shape[1:]}, "
          f"reward shape: {rewards_np.shape[1:]}")
