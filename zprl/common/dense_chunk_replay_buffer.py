import numpy as np
import torch
from typing import List, Optional
from stable_baselines3.common.type_aliases import ReplayBufferSamples

from zprl.common.online_episode_accumulator import EpisodeData


class DenseChunkReplayBuffer:
    """
    Replay buffer that receives complete episodes, expands them into all
    valid dense sliding-window chunks, and stores flat training samples.

    Each sample is compatible with ResiduePolicy's expected batch format:
        observations:      obs_emb          (Do,)
        next_observations: next_obs_emb     (Do,)
        actions:           concat(res_naction, base_naction, base_next_naction)  (Da*3,)
        rewards:           scalar
        dones:             scalar
    """

    def __init__(
        self,
        capacity: int,
        obs_dim: int,        # Do = To * do
        action_dim: int,     # Da = Ta * da
        device: torch.device = torch.device('cpu'),
    ):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device

        # pre-allocate storage on CPU (moved to device at sample time)
        self.obs_emb = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs_emb = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim * 3), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

        self.pos = 0       # next write position
        self.size = 0      # current number of valid samples

    def add_chunks(
        self,
        obs_embs: np.ndarray,         # (M, Do)
        next_obs_embs: np.ndarray,    # (M, Do)
        packed_actions: np.ndarray,   # (M, Da*3)
        rewards: np.ndarray,          # (M,) or (M,1)
        dones: np.ndarray,            # (M,) or (M,1)
    ):
        """Write M pre-computed chunks into the ring buffer."""
        M = obs_embs.shape[0]
        if M == 0:
            return

        rewards = rewards.reshape(-1, 1)
        dones = dones.reshape(-1, 1)

        # ring-buffer write
        if self.pos + M <= self.capacity:
            sl = slice(self.pos, self.pos + M)
            self.obs_emb[sl] = obs_embs
            self.next_obs_emb[sl] = next_obs_embs
            self.actions[sl] = packed_actions
            self.rewards[sl] = rewards
            self.dones[sl] = dones
            self.pos = self.pos + M
        else:
            # wrap around
            first = self.capacity - self.pos
            self.obs_emb[self.pos:] = obs_embs[:first]
            self.next_obs_emb[self.pos:] = next_obs_embs[:first]
            self.actions[self.pos:] = packed_actions[:first]
            self.rewards[self.pos:] = rewards[:first]
            self.dones[self.pos:] = dones[:first]

            rest = M - first
            self.obs_emb[:rest] = obs_embs[first:]
            self.next_obs_emb[:rest] = next_obs_embs[first:]
            self.actions[:rest] = packed_actions[first:]
            self.rewards[:rest] = rewards[first:]
            self.dones[:rest] = dones[first:]
            self.pos = rest

        if self.pos >= self.capacity:
            self.pos = self.pos % self.capacity
        self.size = min(self.size + M, self.capacity)

    def sample(self, batch_size: int) -> ReplayBufferSamples:
        indices = np.random.randint(0, self.size, size=batch_size)
        return ReplayBufferSamples(
            observations=torch.as_tensor(
                self.obs_emb[indices], device=self.device),
            next_observations=torch.as_tensor(
                self.next_obs_emb[indices], device=self.device),
            actions=torch.as_tensor(
                self.actions[indices], device=self.device),
            rewards=torch.as_tensor(
                self.rewards[indices], device=self.device),
            dones=torch.as_tensor(
                self.dones[indices], device=self.device),
        )

    def __len__(self):
        return self.size

    @staticmethod
    def finalize_episode(
        episode: EpisodeData,
        base_policy,
        n_obs_steps: int,    # To
        n_action_steps: int, # Ta
        action_dim: int,     # da (per-step)
        res_scale: float,
        gamma: float = 0.99,
        chunk_stride: int = 1,
        keep_terminal_chunk: bool = True,
        encode_batch_size: int = 256,
    ):
        """
        Convert a complete episode into training chunks.

        Chunk selection is controlled by chunk_stride and keep_terminal_chunk:
        - stride samples starting points {0, stride, 2*stride, ...}
        - if keep_terminal_chunk, the last valid chunk (T-Ta) is always included

        Returns numpy arrays ready for add_chunks():
            obs_embs, next_obs_embs, packed_actions, rewards, dones
        """
        To = n_obs_steps
        Ta = n_action_steps
        da = action_dim
        T = len(episode.rewards)  # number of primitive steps
        N = T + 1                 # number of observations

        # invariant checks
        assert chunk_stride >= 1, f"chunk_stride must be >= 1, got {chunk_stride}"
        assert res_scale != 0, "res_scale must be non-zero"
        assert len(episode.observations) == T + 1, \
            f"observations length {len(episode.observations)} != rewards length {T} + 1"
        assert len(episode.base_actions) == T, \
            f"base_actions length {len(episode.base_actions)} != rewards length {T}"
        assert len(episode.sum_actions) == T, \
            f"sum_actions length {len(episode.sum_actions)} != rewards length {T}"
        assert len(episode.dones) == T, \
            f"dones length {len(episode.dones)} != rewards length {T}"

        if T < Ta:
            # episode too short to form any valid chunk
            return None

        num_all = T - Ta + 1      # total valid starting points: 0 .. T-Ta
        terminal_t = num_all - 1  # last valid starting point

        # ----------------------------------------------------------
        # 0. Build selected_t: stride subset + terminal keep
        # ----------------------------------------------------------
        selected_t = list(range(0, num_all, chunk_stride))
        if keep_terminal_chunk and (len(selected_t) == 0 or selected_t[-1] != terminal_t):
            selected_t.append(terminal_t)
        selected_t = np.array(selected_t, dtype=np.int64)
        M = len(selected_t)  # number of chunks to produce

        device = base_policy.device
        dtype = base_policy.dtype

        # ----------------------------------------------------------
        # 1. Encode all observations into frame embeddings
        # ----------------------------------------------------------
        obs_list = episode.observations  # list of N obs dicts
        keys = obs_list[0].keys()
        batched_obs = {
            k: np.stack([o[k] for o in obs_list])
            for k in keys
        }

        frame_embs_list = []
        with torch.no_grad():
            for start in range(0, N, encode_batch_size):
                end = min(start + encode_batch_size, N)
                batch_dict = {
                    k: torch.from_numpy(v[start:end]).to(
                        device=device, dtype=dtype)
                    for k, v in batched_obs.items()
                }
                emb = base_policy.encode_frame(batch_dict)  # (B, do)
                frame_embs_list.append(emb.cpu().numpy())

        frame_embs = np.concatenate(frame_embs_list, axis=0)  # (N, do)
        do = frame_embs.shape[1]
        Do = To * do

        # ----------------------------------------------------------
        # 2. Build obs_chunk and next_obs_chunk for selected_t
        #    with left-padding (same as MultiStepWrapper)
        # ----------------------------------------------------------
        pad = np.repeat(frame_embs[0:1], To - 1, axis=0)  # (To-1, do)
        padded = np.concatenate([pad, frame_embs], axis=0)  # (To-1+N, do)

        obs_embs = np.zeros((M, Do), dtype=np.float32)
        next_obs_embs = np.zeros((M, Do), dtype=np.float32)
        for i, t in enumerate(selected_t):
            obs_embs[i] = padded[t:t + To].reshape(-1)
            next_obs_embs[i] = padded[t + Ta:t + Ta + To].reshape(-1)

        # ----------------------------------------------------------
        # 3. Build action chunks for selected_t
        # ----------------------------------------------------------
        base_actions = np.stack(episode.base_actions)  # (T, da)
        sum_actions = np.stack(episode.sum_actions)    # (T, da)
        Da = Ta * da

        base_nactions = np.zeros((M, Da), dtype=np.float32)
        sum_nactions = np.zeros((M, Da), dtype=np.float32)
        for i, t in enumerate(selected_t):
            base_nactions[i] = base_actions[t:t + Ta].reshape(-1)
            sum_nactions[i] = sum_actions[t:t + Ta].reshape(-1)

        res_nactions = (sum_nactions - base_nactions) / res_scale

        # ----------------------------------------------------------
        # 4. Compute base_next_naction from next_obs_chunk embeddings
        # ----------------------------------------------------------
        base_next_nactions_list = []
        with torch.no_grad():
            for start in range(0, M, encode_batch_size):
                end = min(start + encode_batch_size, M)
                next_emb = torch.from_numpy(
                    next_obs_embs[start:end]
                ).to(device=device, dtype=dtype)

                # VIB forward if available
                if hasattr(base_policy, 'vib_forward'):
                    next_emb, _, _, _ = base_policy.vib_forward(
                        next_emb, deterministic=True)

                result = base_policy.conditional_predict(next_emb)
                naction = result['naction'].reshape(-1, Da)
                base_next_nactions_list.append(naction.cpu().numpy())

        base_next_nactions = np.concatenate(
            base_next_nactions_list, axis=0)  # (M, Da)

        # ----------------------------------------------------------
        # 5. Pack actions: concat(res, base, base_next)
        # ----------------------------------------------------------
        packed_actions = np.concatenate(
            [res_nactions, base_nactions, base_next_nactions], axis=-1
        )  # (M, Da*3)

        # ----------------------------------------------------------
        # 6. Compute reward and done for selected_t
        # ----------------------------------------------------------
        ep_rewards = np.array(episode.rewards, dtype=np.float64)
        ep_dones = np.array(episode.dones, dtype=np.float32)

        discounts = np.array([gamma ** i for i in range(Ta)], dtype=np.float64)
        chunk_rewards = np.zeros(M, dtype=np.float32)
        chunk_dones = np.zeros(M, dtype=np.float32)
        for i, t in enumerate(selected_t):
            chunk_rewards[i] = np.dot(discounts, ep_rewards[t:t + Ta])
            chunk_dones[i] = ep_dones[t + Ta - 1]

        return obs_embs, next_obs_embs, packed_actions, chunk_rewards, chunk_dones
