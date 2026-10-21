"""Fixed-batch chunks over the unchanged native OneLeg environment."""

import torch

from zprl.env.furniturebench.oneleg_env import OneLegEnv


class OneLegOnlineEnv:
    """Keep fixed batch rows; overwrite inactive actions before each native step.

    n_action_steps is the decision interval Tr, including for ResRL with Ta>Tr.
    Inputs and outputs always retain all num_envs rows. Previously inactive rows
    have zero reward/length, no new done event, and transition_valid=False.
    A failed chunk raises without returning transitions. Eval additionally marks
    unfinished results as batch failures; the caller closes the failed instance.
    """

    def __init__(self, n_action_steps=8, max_steps=700, single_gamma=0.997,
            mode='train', env=None, **env_kwargs):
        if n_action_steps < 1 or max_steps < 1 or mode not in ('train', 'eval'):
            raise ValueError('Invalid chunk length, episode limit or environment mode')
        self.env = env if env is not None else OneLegEnv(max_env_steps=max_steps, **env_kwargs)
        self.num_envs = self.env.num_envs
        self.device = self.env.device
        self.n_action_steps = n_action_steps
        self.max_steps = max_steps
        self.single_gamma = single_gamma
        self.mode = mode
        # Gripper zero stays inside the native hysteresis margin: retain last_grasp.
        self.hold_action = torch.tensor([0., 0., 0., 1., 0., 0., 0., 1., 0., 0.],
                                        device=self.device)
        self.active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.episode_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.episode_returns = torch.zeros(self.num_envs, device=self.device)
        self.results = [None] * self.num_envs
        self.batch_seed = None
        self.solid_steps = 0
        self.idle_steps = 0
        self.last_chunk_valid = False
        self.batch_error = None
        self._obs = None
        self._failed = False

    @property
    def batch_done(self):
        return not bool(self.active.any())

    @torch.no_grad()
    def reset(self, seed=None):
        if self.env is None or self._failed:
            raise RuntimeError('Close and recreate a closed or failed OneLeg batch')
        if self.active.any():
            raise RuntimeError('Cannot reset while the batch has active episodes')
        obs = self.env.reset(seed=seed)
        self._obs = obs
        self.batch_seed = seed
        self.active.fill_(True)
        self.episode_steps.zero_()
        self.episode_returns.zero_()
        self.results = [None] * self.num_envs
        self.last_chunk_valid = False
        self.batch_error = None
        return {key: value[:, None].clone() for key, value in obs.items()}

    def _result(self, index, reason):
        return dict(env_index=index,
            seed=None if self.batch_seed is None else self.batch_seed + index,
            success=reason == 'success', termination_reason=reason,
            episode_steps=int(self.episode_steps[index]),
            episode_return=float(self.episode_returns[index]))

    def _batch_failed(self, error):
        self._failed = True
        self.batch_error = str(error)
        if self.mode == 'eval':
            for index in self.active.nonzero().flatten().tolist():
                self.results[index] = self._result(index, 'failure')
            self.active.zero_()

    @torch.no_grad()
    def step(self, action):
        if self._obs is None or self.env is None or self._failed or self.batch_done:
            raise RuntimeError('Call reset on a fresh batch before stepping active episodes')
        entry_active = self.active.clone()
        action = torch.as_tensor(action, device=self.device, dtype=torch.float32)

        self.last_chunk_valid = False
        next_obs = {key: value[:, None].clone() for key, value in self._obs.items()}
        rewards = torch.zeros(self.num_envs, device=self.device)
        raw_rewards = torch.zeros_like(rewards)
        lengths = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        terminals = {}
        for step in range(self.n_action_steps):
            if not self.active.any():
                break
            active = self.active.clone()
            full_action = action[:, step].clone()
            full_action[~active] = self.hold_action
            try:
                obs, reward, native_done, _ = self.env.step(full_action)
                count = int(active.sum())
                self.solid_steps += count
                self.idle_steps += self.num_envs - count

                reward = reward.reshape(self.num_envs)
                native_done = native_done.reshape(self.num_envs).bool()
                success = self.env.is_success()
                success = success.bool() | (reward == 1)
            except (RuntimeError, FloatingPointError) as error:
                self._batch_failed(error)
                raise

            timeout = self.episode_steps + active.long() >= self.max_steps
            if (native_done & active & ~(success | timeout)).any():
                self._failed = True
                raise RuntimeError('Native done without success or timeout')
            self.episode_steps[active] += 1
            self.episode_returns[active] += reward[active]
            lengths[active] += 1
            rewards[active] += self.single_gamma ** step * reward[active]
            raw_rewards[active] += reward[active]
            for key, value in obs.items():
                next_obs[key][active, 0] = value[active]
            finished = active & (success | timeout)
            for index in finished.nonzero().flatten().tolist():
                reason = 'success' if success[index] else 'timeout'
                self.results[index] = self._result(index, reason)
                terminals[index] = {key: value[index].clone() for key, value in next_obs.items()}
            self.active[finished] = False

        done = entry_active & ~self.active
        infos = []
        for index in range(self.num_envs):
            result = self.results[index] or self._result(index, None)
            info = dict(
                result,
                executed_action_steps=int(lengths[index]),
                transition_valid=bool(entry_active[index]),
                raw_reward=float(raw_rewards[index]),
                task_reward=float(done[index] and result['success'])
            )
            info['TimeLimit.truncated'] = bool(
                done[index] and result['termination_reason'] == 'timeout')
            if index in terminals:
                info['terminal_observation'] = terminals[index]
            infos.append(info)
        self._obs = {key: value[:, 0].clone() for key, value in next_obs.items()}
        self.last_chunk_valid = True
        return next_obs, rewards, done, infos

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None
        self._obs = None
