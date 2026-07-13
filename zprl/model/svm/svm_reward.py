import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.type_aliases import ReplayBufferSamples


class SVMDiscriminator(nn.Module):
    def __init__(self,
            obs_dim: int,
            action_dim: int,
            hidden_dim: int = 256,
            num_layers: int = 3):
        super().__init__()

        layers = []
        input_dim = obs_dim + action_dim
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
            ])
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        action = action.flatten(start_dim=1)
        return self.network(torch.cat([obs, action], dim=-1))

    def predict_probability(
            self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(obs, action))


class SVMRewardModel(nn.Module):
    def __init__(self,
            # runtime dimensions supplied by workspace
            obs_dim: int,
            action_dim: int,
            device,
            # process reward
            enable: bool = False,
            reward_scale: float = 0.05,
            reward_clip: float = 20.0,
            # discriminator architecture
            hidden_dim: int = 256,
            num_layers: int = 3,
            # discriminator training
            batch_size: int = 64,
            learning_rate: float = 1e-4,
            gradient_steps: int = 32,
            warmup_gradient_steps: int = 256,
            update_every: int = 1000,
            min_samples_per_class: int = 64):
        super().__init__()

        self.enable = enable
        self.reward_scale = reward_scale
        self.reward_clip = reward_clip
        self.batch_size = batch_size
        self.gradient_steps = gradient_steps
        self.warmup_gradient_steps = warmup_gradient_steps
        self.update_every = update_every
        self.min_samples_per_class = min_samples_per_class

        self.discriminator = None
        self.optimizer = None
        self.ready = False
        self.last_reward_info = dict()

        if not self.enable:
            return

        self.discriminator = SVMDiscriminator(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        ).to(device)
        self.optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=learning_rate,
        )

    def warmup_update(self, replay_buffer) -> dict:
        if not self.enable:
            return dict()

        self._check_warmup_samples(replay_buffer)
        info = self._get_sample_info(replay_buffer)
        info.update(self._train(replay_buffer, self.warmup_gradient_steps))
        self.ready = True
        return info

    def update(self, replay_buffer, global_step: int) -> dict:
        if (
            not self.enable or
            not self.ready or
            global_step % self.update_every != 0
        ):
            return dict()

        info = self._get_sample_info(replay_buffer)
        info.update(self._train(replay_buffer, self.gradient_steps))
        return info

    def _get_sample_info(self, replay_buffer) -> dict:
        num_positive, num_negative = (
            replay_buffer.get_discriminator_sample_counts())
        return {
            'svm/samples_pos': num_positive,
            'svm/samples_neg': num_negative,
            'svm/epi_success':
                replay_buffer.num_success_episodes,
            'svm/epi_failure':
                replay_buffer.num_failure_episodes,
        }

    def _check_warmup_samples(self, replay_buffer) -> None:
        num_positive, num_negative = (
            replay_buffer.get_discriminator_sample_counts())
        if min(num_positive, num_negative) < self.min_samples_per_class:
            raise RuntimeError(
                "SVM warmup data is insufficient to train the reward model: "
                f"positive_samples={num_positive}, "
                f"negative_samples={num_negative}, "
                f"min_samples_per_class={self.min_samples_per_class}. "
                "Increase training.learning_start / warmup rollouts, or "
                "lower svm.min_samples_per_class."
            )

    def _train(self, replay_buffer, gradient_steps: int) -> dict:
        update_info = []
        self.discriminator.train()
        for _ in range(gradient_steps):
            batch = replay_buffer.sample_discriminator(self.batch_size)
            update_info.append(self._update_batch(batch))
        self.discriminator.eval()

        info = dict()
        for key in update_info[0]:
            info[key] = sum(item[key] for item in update_info) / len(update_info)
        return info

    def _update_batch(self, batch) -> dict:
        logits = self.discriminator(
            batch.observations, batch.svm_actions)
        loss = F.binary_cross_entropy_with_logits(logits, batch.labels)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            positive_mask = batch.labels > 0.5
            negative_mask = ~positive_mask
            accuracy = ((logits >= 0) == positive_mask).float().mean()
            probabilities = torch.sigmoid(logits)

        return {
            'svm/discriminator_loss': loss.item(),
            'svm/discriminator_accuracy': accuracy.item(),
            'svm/logit_pos': logits[positive_mask].mean().item(),
            'svm/logit_neg': logits[negative_mask].mean().item(),
            'svm/prob_pos':
                probabilities[positive_mask].mean().item(),
            'svm/prob_neg':
                probabilities[negative_mask].mean().item(),
        }

    def process_batch_reward(self, batch) -> ReplayBufferSamples:
        rewards = batch.rewards
        process_rewards = torch.zeros_like(rewards)

        if self.enable:
            if not self.ready:
                raise RuntimeError(
                    "SVM reward model is not ready. "
                    "Call warmup_update() after warmup and before "
                    "process_batch_reward()."
                )

            self.discriminator.eval()
            with torch.no_grad():
                logits = self.discriminator(
                    batch.observations, batch.svm_actions)
                process_rewards = torch.clamp(
                    logits, -self.reward_clip, self.reward_clip)
                rewards = rewards + self.reward_scale * process_rewards

        self.last_reward_info = {
            'svm/reward_process': process_rewards.mean().item(),
            'svm/reward_combined': rewards.mean().item(),
        }

        return ReplayBufferSamples(
            observations=batch.observations,
            actions=batch.actions,
            next_observations=batch.next_observations,
            dones=batch.dones,
            rewards=rewards,
            discounts=getattr(batch, 'discounts', None),
        )
