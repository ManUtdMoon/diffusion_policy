from typing import NamedTuple

import numpy as np
import torch
from stable_baselines3.common.buffers import ReplayBuffer


class SVMReplayBufferSamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor
    rewards: torch.Tensor
    svm_actions: torch.Tensor


class SVMDiscriminatorSamples(NamedTuple):
    observations: torch.Tensor
    svm_actions: torch.Tensor
    labels: torch.Tensor


class SVMReplayBuffer(ReplayBuffer):
    def __init__(self,
            buffer_size,
            observation_space,
            action_space,
            svm_action_dim: int,
            device="auto",
            n_envs: int = 1,
            optimize_memory_usage: bool = False,
            handle_timeout_termination: bool = True):
        super().__init__(
            buffer_size=buffer_size,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            n_envs=n_envs,
            optimize_memory_usage=optimize_memory_usage,
            handle_timeout_termination=handle_timeout_termination,
        )

        self.svm_action_dim = svm_action_dim
        self.svm_actions = np.zeros(
            (self.buffer_size, self.n_envs, svm_action_dim), dtype=np.float32)
        self.episode_labels = np.full(
            (self.buffer_size, self.n_envs), -1, dtype=np.int8)
        self._slot_generations = np.zeros(
            (self.buffer_size, self.n_envs), dtype=np.int64)
        self._pending_slots = [[] for _ in range(self.n_envs)]
        self._episode_success = np.zeros(self.n_envs, dtype=np.bool_)

        self.num_success_episodes = 0
        self.num_failure_episodes = 0

    def add(
            self,
            obs: np.ndarray,
            next_obs: np.ndarray,
            action: np.ndarray,
            reward: np.ndarray,
            done: np.ndarray,
            infos: list[dict],
            svm_action: np.ndarray,
            outcome_reward: np.ndarray) -> None:
        pos = self.pos
        svm_action = np.asarray(svm_action).reshape(
            self.n_envs, self.svm_action_dim)
        outcome_reward = np.asarray(outcome_reward).reshape(self.n_envs)
        done = np.asarray(done).reshape(self.n_envs)

        for env_idx, pending_slots in enumerate(self._pending_slots):
            for pending_pos, generation in pending_slots:
                if (
                    pending_pos == pos and
                    self._slot_generations[pos, env_idx] == generation
                ):
                    raise RuntimeError(
                        "SVM replay buffer cannot overwrite an unfinished "
                        f"episode for environment {env_idx}."
                    )

        super().add(
            obs=obs,
            next_obs=next_obs,
            action=action,
            reward=reward,
            done=done,
            infos=infos,
        )

        self._slot_generations[pos] += 1
        self.episode_labels[pos] = -1
        self.svm_actions[pos] = svm_action

        for env_idx in range(self.n_envs):
            generation = self._slot_generations[pos, env_idx]
            self._pending_slots[env_idx].append((pos, generation))
            self._episode_success[env_idx] |= outcome_reward[env_idx] > 0.9

            if done[env_idx]:
                label = int(self._episode_success[env_idx])
                for pending_pos, pending_generation in self._pending_slots[env_idx]:
                    if (
                        self._slot_generations[pending_pos, env_idx] !=
                        pending_generation
                    ):
                        raise RuntimeError(
                            "SVM replay buffer episode data was overwritten "
                            f"for environment {env_idx}."
                        )
                    self.episode_labels[pending_pos, env_idx] = label

                if label == 1:
                    self.num_success_episodes += 1
                else:
                    self.num_failure_episodes += 1
                self._pending_slots[env_idx].clear()
                self._episode_success[env_idx] = False

    def _get_samples(
            self, batch_inds: np.ndarray, env=None) -> SVMReplayBufferSamples:
        env_indices = np.random.randint(
            0, high=self.n_envs, size=len(batch_inds))

        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(
                self.observations[
                    (batch_inds + 1) % self.buffer_size, env_indices, :], env)
        else:
            next_obs = self._normalize_obs(
                self.next_observations[batch_inds, env_indices, :], env)

        data = (
            self._normalize_obs(
                self.observations[batch_inds, env_indices, :], env),
            self.actions[batch_inds, env_indices, :],
            next_obs,
            (
                self.dones[batch_inds, env_indices] *
                (1 - self.timeouts[batch_inds, env_indices])
            ).reshape(-1, 1),
            self._normalize_reward(
                self.rewards[batch_inds, env_indices].reshape(-1, 1), env),
            self.svm_actions[batch_inds, env_indices, :],
        )
        return SVMReplayBufferSamples(
            *tuple(map(self.to_torch, data)))

    def sample_discriminator(
            self, batch_size: int, env=None) -> SVMDiscriminatorSamples:
        if batch_size % 2 != 0:
            raise ValueError("Discriminator batch size must be even.")

        flat_labels = self.episode_labels.reshape(-1)
        positive_indices = np.flatnonzero(flat_labels == 1)
        negative_indices = np.flatnonzero(flat_labels == 0)
        if len(positive_indices) == 0 or len(negative_indices) == 0:
            raise RuntimeError(
                "Both successful and failed transitions are required to "
                "sample a discriminator batch."
            )

        half_batch_size = batch_size // 2
        flat_indices = np.concatenate([
            np.random.choice(positive_indices, size=half_batch_size),
            np.random.choice(negative_indices, size=half_batch_size),
        ])
        np.random.shuffle(flat_indices)

        batch_inds = flat_indices // self.n_envs
        env_indices = flat_indices % self.n_envs
        data = (
            self._normalize_obs(
                self.observations[batch_inds, env_indices, :], env),
            self.svm_actions[batch_inds, env_indices, :],
            self.episode_labels[batch_inds, env_indices]
                .astype(np.float32).reshape(-1, 1),
        )
        return SVMDiscriminatorSamples(
            *tuple(map(self.to_torch, data)))

    def get_discriminator_sample_counts(self) -> tuple[int, int]:
        num_positive = int(np.count_nonzero(self.episode_labels == 1))
        num_negative = int(np.count_nonzero(self.episode_labels == 0))
        return num_positive, num_negative

    def reset(self) -> None:
        super().reset()
        self.svm_actions.fill(0)
        self.episode_labels.fill(-1)
        self._slot_generations.fill(0)
        self._pending_slots = [[] for _ in range(self.n_envs)]
        self._episode_success.fill(False)
        self.num_success_episodes = 0
        self.num_failure_episodes = 0
