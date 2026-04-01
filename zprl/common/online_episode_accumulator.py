from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


@dataclass
class EpisodeData:
    """Complete primitive-level trajectory for one episode."""
    observations: list = field(default_factory=list)   # [o_0, ..., o_T]  len = T+1
    base_actions: list = field(default_factory=list)    # [a^base_0, ..., a^base_{T-1}]
    sum_actions: list = field(default_factory=list)     # [a^sum_0, ..., a^sum_{T-1}]
    rewards: list = field(default_factory=list)         # [r_0, ..., r_{T-1}]
    dones: list = field(default_factory=list)           # [d_0, ..., d_{T-1}]


class OnlineEpisodeAccumulator:
    """
    Buffers ongoing episodes per env_idx from dense_step info and action
    chunks provided by the workspace.  When an episode finishes, it is
    returned as a complete EpisodeData.

    Usage (per outer step):
        finished = accumulator.add_step(
            infos          = infos,          # list[dict] from envs.step()
            base_naction_chunks = ...,       # (n_envs, Ta, da) np.ndarray, normalized
            sum_naction_chunks  = ...,       # (n_envs, Ta, da) np.ndarray, normalized
        )
        # finished: list of EpisodeData
    """

    def __init__(self, n_envs: int):
        self.n_envs = n_envs
        self.ongoing: List[Optional[EpisodeData]] = [None] * n_envs

    def add_step(
        self,
        infos: list,
        base_naction_chunks: np.ndarray,
        sum_naction_chunks: np.ndarray,
    ) -> List[EpisodeData]:
        """
        Args:
            infos: list of info dicts, one per env.  Each must contain
                   infos[i]['dense_step'].
            base_naction_chunks: (n_envs, Ta, da) normalized base action chunks.
            sum_naction_chunks:  (n_envs, Ta, da) normalized sum action chunks.

        Returns:
            List of completed EpisodeData (may be empty).
        """
        finished: List[EpisodeData] = []

        for env_idx in range(self.n_envs):
            dense = infos[env_idx]['dense_step']
            include_initial: bool = dense['include_initial']
            episode_done: bool = dense['episode_done']
            dense_obs: list = dense['observations']       # list of obs dicts
            dense_rewards: np.ndarray = dense['rewards']  # (k,)
            dense_dones: np.ndarray = dense['dones']      # (k,)
            k = len(dense_obs)  # actual primitive steps in this chunk

            # --- start new episode if needed ---
            if include_initial:
                assert self.ongoing[env_idx] is None, (
                    f"env {env_idx}: include_initial=True but previous episode "
                    f"was not finished (len={len(self.ongoing[env_idx].observations)}). "
                    f"This indicates a bug in wrapper or accumulator wiring."
                )
                initial_obs = dense['initial_obs']
                assert initial_obs is not None
                ep = EpisodeData()
                ep.observations.append(initial_obs)
                self.ongoing[env_idx] = ep

            ep = self.ongoing[env_idx]
            assert ep is not None, (
                f"env {env_idx}: received chunk without an ongoing episode "
                f"(include_initial={include_initial})"
            )

            # --- append dense obs / reward / done ---
            ep.observations.extend(dense_obs)
            ep.rewards.extend(dense_rewards.tolist())
            ep.dones.extend(dense_dones.tolist())

            # --- append primitive actions (only the first k) ---
            base_chunk = base_naction_chunks[env_idx]  # (Ta, da)
            sum_chunk = sum_naction_chunks[env_idx]     # (Ta, da)
            for j in range(k):
                ep.base_actions.append(base_chunk[j])
                ep.sum_actions.append(sum_chunk[j])

            # --- finish episode ---
            if episode_done:
                finished.append(ep)
                self.ongoing[env_idx] = None

        return finished
