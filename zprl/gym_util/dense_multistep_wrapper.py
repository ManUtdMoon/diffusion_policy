import gym
import numpy as np
import dill
from collections import defaultdict, deque

from zprl.gym_util.multistep_wrapper import (
    repeated_space,
    stack_last_n_obs,
    dict_take_last_n,
    aggregate,
)


class DenseMultiStepWrapper(gym.Wrapper):
    """
    Like MultiStepWrapper, but additionally provides dense primitive-level
    observation / reward / done data in info['dense_step'] for episode
    trajectory reconstruction.

    The outer interface (obs_seq, aggregated reward, done) stays identical
    to MultiStepWrapper so existing training code keeps working.
    """

    def __init__(
        self,
        env,
        n_obs_steps,
        n_action_steps,
        max_episode_steps=None,
        reward_agg_method='max',
        gamma=0.99,
    ):
        super().__init__(env)
        self._action_space = repeated_space(env.action_space, n_action_steps)
        self._observation_space = repeated_space(env.observation_space, n_obs_steps)
        self.max_episode_steps = max_episode_steps
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.reward_agg_method = reward_agg_method
        self.gamma = gamma

        # --- stacking state (same as MultiStepWrapper) ---
        self.obs = deque(maxlen=n_obs_steps + 1)
        self.reward = list()
        self.done = list()
        self.info = defaultdict(lambda: deque(maxlen=n_obs_steps + 1))

        # --- dense trajectory state ---
        self._initial_obs = None      # o_0 of current episode
        self._is_first_chunk = True   # whether the next step() is the first chunk

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------
    def reset(self):
        obs = super().reset()

        # stacking state
        self.obs = deque([obs], maxlen=self.n_obs_steps + 1)
        self.reward = list()
        self.done = list()
        self.info = defaultdict(lambda: deque(maxlen=self.n_obs_steps + 1))

        # dense state
        self._initial_obs = self._copy_obs(obs)
        self._is_first_chunk = True

        obs_seq = self._get_obs(self.n_obs_steps)
        return obs_seq

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------
    def step(self, action):
        """
        action: (n_action_steps,) + action_shape
        """
        chunk_rewards = []
        chunk_dones = []
        dense_observations = []
        dense_rewards = []
        dense_dones = []

        for act in action:
            if len(self.done) > 0 and self.done[-1]:
                break

            observation, reward, done, info = super().step(act)

            # stacking state
            self.obs.append(observation)
            self.reward.append(reward)
            chunk_rewards.append(reward)
            if (self.max_episode_steps is not None) \
                    and (len(self.reward) >= self.max_episode_steps):
                done = True
            self.done.append(done)
            chunk_dones.append(done)
            self._add_info(info)

            # dense collection
            dense_observations.append(self._copy_obs(observation))
            dense_rewards.append(reward)
            dense_dones.append(done)

        # build outer return (identical to MultiStepWrapper)
        obs_seq = self._get_obs(self.n_obs_steps)
        agg_reward = aggregate(chunk_rewards, self.reward_agg_method, self.gamma)
        agg_done = aggregate(chunk_dones, 'max')
        out_info = dict_take_last_n(self.info, self.n_obs_steps)

        # build dense_step
        episode_done = bool(agg_done)
        dense_step = {
            'initial_obs': self._initial_obs if self._is_first_chunk else None,
            'observations': dense_observations,
            'rewards': np.array(dense_rewards, dtype=np.float64),
            'dones': np.array(dense_dones, dtype=bool),
            'include_initial': self._is_first_chunk,
            'episode_done': episode_done,
        }
        out_info['dense_step'] = dense_step

        # state transitions
        if self._is_first_chunk:
            self._is_first_chunk = False

        if episode_done:
            # After auto-reset (called by AsyncVectorEnv worker),
            # reset() will set _initial_obs and _is_first_chunk for the
            # new episode.
            pass

        return obs_seq, agg_reward, agg_done, out_info

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _get_obs(self, n_steps=1):
        assert len(self.obs) > 0
        if isinstance(self.observation_space, gym.spaces.Box):
            return stack_last_n_obs(self.obs, n_steps)
        elif isinstance(self.observation_space, gym.spaces.Dict):
            result = dict()
            for key in self.observation_space.keys():
                result[key] = stack_last_n_obs(
                    [obs[key] for obs in self.obs], n_steps)
            return result
        else:
            raise RuntimeError('Unsupported space type')

    def _add_info(self, info):
        for key, value in info.items():
            self.info[key].append(value)

    @staticmethod
    def _copy_obs(obs):
        if isinstance(obs, dict):
            return {k: np.array(v) for k, v in obs.items()}
        return np.array(obs)

    def get_rewards(self):
        return self.reward

    def get_attr(self, name):
        return getattr(self, name)

    def run_dill_function(self, dill_fn):
        fn = dill.loads(dill_fn)
        return fn(self)

    def get_infos(self):
        result = dict()
        for k, v in self.info.items():
            result[k] = list(v)
        return result
