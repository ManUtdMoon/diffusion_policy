import numpy as np
import tqdm
from termcolor import cprint

from diffusion_policy.env.wallet.wallet_env import WalletEnv
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.gym_util.multistep_wrapper import SingleStepStackedObsWrapper
from diffusion_policy.gym_util.sparse_obs_history_wrapper import SparseObsHistoryWrapper
from diffusion_policy.real_world.real_time_chunk_runtime import RealTimeChunkRuntime


class WalletRealtimeRunner(BaseImageRunner):
    def __init__(
        self,
        output_dir,
        server_addr,
        eval_episodes=30,
        max_steps=1000,
        n_obs_steps=1,
        obs_step_indices=None,
        n_action_steps=8,
        min_exec_horizon=None,
        delay_buffer_size=6,
        timeout_ms=60000,
        tqdm_interval_sec=5.0,
    ):
        super().__init__(output_dir)

        env_n_obs_steps = n_obs_steps
        if obs_step_indices is not None:
            assert len(obs_step_indices) == n_obs_steps
            env_n_obs_steps = max(obs_step_indices) + 1

        self.env = SingleStepStackedObsWrapper(
            WalletEnv(),
            n_obs_steps=env_n_obs_steps,
            max_episode_steps=max_steps,
        )
        if obs_step_indices is not None:
            self.env = SparseObsHistoryWrapper(self.env, obs_step_indices=obs_step_indices)

        self.server_addr = server_addr
        self.eval_episodes = eval_episodes
        self.max_steps = max_steps
        self.n_obs_steps = n_obs_steps
        self.obs_step_indices = obs_step_indices
        self.n_action_steps = n_action_steps
        self.min_exec_horizon = (
            min_exec_horizon if min_exec_horizon is not None else n_action_steps
        )
        self.delay_buffer_size = delay_buffer_size
        self.timeout_ms = timeout_ms
        self.tqdm_interval_sec = tqdm_interval_sec

    def run(self):
        env = self.env

        completed_episodes = 0
        all_success = []
        all_returns = []
        all_epi_len = []
        all_policy_latencies = []

        pbar = tqdm.tqdm(
            total=self.eval_episodes,
            desc="Eval in Wallet Realtime Env",
            leave=False,
            mininterval=self.tqdm_interval_sec,
        )

        try:
            while completed_episodes < self.eval_episodes:
                obs = env.reset()
                runtime = RealTimeChunkRuntime(
                    server_addr=self.server_addr,
                    n_action_steps=self.n_action_steps,
                    min_exec_horizon=self.min_exec_horizon,
                    timeout_ms=self.timeout_ms,
                    delay_buffer_size=self.delay_buffer_size,
                )
                runtime.reset(self._make_obs_dict(obs))

                actual_step_count = 0
                episode_done = False
                episode_return = 0
                pre_reward = -1
                info = {}

                try:
                    while not episode_done:
                        obs_dict = self._make_obs_dict(obs)
                        action = runtime.step(obs_dict)
                        obs, reward, done, info = env.step(action.copy())

                        if reward is None:
                            reward = pre_reward

                        episode_return += reward
                        pre_reward = reward
                        actual_step_count += 1
                        episode_done = done
                finally:
                    runtime.close()
                    runtime_log = runtime.get_log()
                    all_policy_latencies.extend(runtime_log["policy_latency_sec"])

                completed_episodes += 1
                is_success = bool(info.get("is_success", False))
                if is_success:
                    assert reward > 0.5, "Success but low reward!"
                    all_epi_len.append(info.get("episode_length", actual_step_count))

                all_success.append(is_success)
                all_returns.append(episode_return)

                print("Is success: ", is_success)
                print("SR till now: ", sum(all_success) / completed_episodes)
                print(f"Success till now: {sum(all_success)} / {completed_episodes}")
                if len(all_epi_len) > 0:
                    print(f"Epi length till now: {sum(all_epi_len) / len(all_epi_len)}")

                env.reset_end()
                pbar.update(1)
        finally:
            pbar.close()

        log_data = dict()
        all_success_rate = sum(all_success) / self.eval_episodes
        log_data["mean_sr"] = all_success_rate
        log_data["mean_return"] = np.mean(all_returns)
        log_data["mean_epi_length"] = np.mean(all_epi_len) if len(all_epi_len) > 0 else None
        log_data.update(self._summarize_latency("policy_latency", all_policy_latencies))
        cprint(f"test_mean_score: {all_success_rate}", "green")
        cprint(f"mean_returns: {np.mean(all_returns)}", "green")

        env.close()
        return log_data

    def _make_obs_dict(self, obs):
        return {
            "global": obs["global"].astype(np.float32),
            "wrist_0": obs["wrist_0"].astype(np.float32),
            "wrist_1": obs["wrist_1"].astype(np.float32),
            "qpos": obs["qpos"].astype(np.float32),
        }

    def _summarize_latency(self, prefix, values):
        if len(values) == 0:
            return {}
        values = np.asarray(values, dtype=np.float64)
        return {
            f"{prefix}_mean_sec": float(np.mean(values)),
            f"{prefix}_max_sec": float(np.max(values)),
            f"{prefix}_p95_sec": float(np.percentile(values, 95)),
            f"{prefix}_count": int(values.size),
        }
