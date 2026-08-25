import wandb
import numpy as np
import torch
import tqdm
from termcolor import cprint
import pathlib
import dill

from zprl.gym_util.async_vector_env import AsyncVectorEnv
from zprl.env.adroit.adroit import AdroitEnv
from zprl.gym_util.multistep_wrapper import MultiStepWrapper
from zprl.gym_util.video_recording_wrapper import VideoRecordingWrapper
from zprl.real_world.video_recorder import VideoRecorder
from zprl.policy.base_image_policy import BaseImagePolicy
from zprl.common.pytorch_util import dict_apply
from zprl.env_runner.base_image_runner import BaseImageRunner


def _make_init_fn(output_dir, episode_index, render, seed):
    def init_fn(env):
        assert isinstance(env.env, VideoRecordingWrapper)
        env.env.video_recoder.stop()
        env.env.file_path = None
        if render:
            filename = pathlib.Path(output_dir).joinpath(
                'media', f"test_{seed}.mp4")
            filename.parent.mkdir(parents=True, exist_ok=True)
            env.env.file_path = str(filename)
        env.seed(seed)
    return init_fn


class AdroitRunner(BaseImageRunner):
    def __init__(self,
            output_dir,
            eval_episodes=50,
            max_steps=300,
            n_obs_steps=1,
            n_action_steps=5,
            fps=10,
            crf=22,
            tqdm_interval_sec=5.0,
            task_name="door",
            n_envs=25,
            n_epi_vis=5,
            test_start_seed=10000,
            render_device_id=0,
        ):
        super().__init__(output_dir)
        self.task_name = task_name
        if 'pen' in task_name:
            self.success_threshold = 40
        else:
            self.success_threshold = 50

        steps_per_render = max(10 // fps, 1)
        max_steps = max_steps // 2

        assert eval_episodes % n_envs == 0, \
            "eval_episodes must be divisible by n_envs"

        def env_fn():
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    AdroitEnv(
                        env_name=task_name,
                        render_device_id=render_device_id,
                    ),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps,
                        codec='h264',
                        input_pix_fmt='rgb24',
                        crf=crf,
                        thread_type='FRAME',
                        thread_count=1
                    ),
                    file_path=None,
                    steps_per_render=steps_per_render,
                ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps,
                reward_agg_method='sum',
            )
        env_fns = [env_fn for _ in range(n_envs)]
        env_seeds = list()
        env_init_fn_dills = list()

        for i in range(eval_episodes):
            seed = test_start_seed + i
            render = i < n_epi_vis
            init_fn = _make_init_fn(output_dir, i, render, seed)
            env_seeds.append(seed)
            env_init_fn_dills.append(dill.dumps(init_fn))

        env = AsyncVectorEnv(env_fns, dummy_env_fn=env_fn)

        self.env = env
        self.env_seeds = env_seeds
        self.env_init_fn_dills = env_init_fn_dills
        self.eval_episodes = eval_episodes
        self.n_envs = n_envs

        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec
        self._closed = False

    def run(self, policy: BaseImagePolicy):
        device = policy.device
        env = self.env

        n_envs = self.n_envs
        n_epis = self.eval_episodes
        n_chunks = n_epis // n_envs

        all_n_goal_achieved = [[] for _ in range(n_epis)]
        all_rewards = [[] for _ in range(n_epis)]
        all_video_paths = [None for _ in range(n_epis)]

        for i in range(n_chunks):
            start = i * n_envs
            end = (i + 1) * n_envs
            global_slice = slice(start, end)
            local_slice = slice(0, end - start)

            this_init_fns = self.env_init_fn_dills[global_slice]
            env.call_each('run_dill_function',
                args_list=[(x,) for x in this_init_fns])

            obs = env.reset()
            policy.reset()
            done = False

            task = self.task_name
            pbar = tqdm.tqdm(
                total=self.max_steps,
                desc=f"Eval {task} {i+1} / {n_chunks}", leave=False,
                mininterval=self.tqdm_interval_sec)

            while not done:
                obs_dict = dict_apply(
                    obs, lambda x: torch.from_numpy(x).to(device=device))

                with torch.no_grad():
                    obs_dict_input = {
                        'agent_pos': obs_dict['agent_pos'].to(torch.float),
                        'image': obs_dict['image'].to(torch.float),
                    }
                    action_dict = policy.predict_action(obs_dict_input)

                np_action_dict = dict_apply(
                    action_dict, lambda x: x.detach().to('cpu').numpy())
                action = np_action_dict['action']

                obs, reward, done, info = env.step(action)

                for sublist, r in zip(all_rewards[global_slice], reward):
                    sublist.append(r)
                for sublist, d in zip(all_n_goal_achieved[global_slice], info):
                    sublist.append(d["n_goal_achieved"])

                done = np.all(done)
                pbar.update(action.shape[1])
            pbar.close()

            all_video_paths[global_slice] = env.render()[local_slice]

        log_data = dict()

        all_returns = np.array([np.sum(r) for r in all_rewards])
        all_n_goal_achieved = np.array(
            [np.sum(g) for g in all_n_goal_achieved])
        n_success = np.sum(all_n_goal_achieved >= self.success_threshold)
        success_rate = n_success / self.eval_episodes

        log_data['test/mean_return'] = np.mean(all_returns)
        log_data['test/mean_n_goal_achieved'] = np.mean(all_n_goal_achieved)
        log_data['test/mean_score'] = success_rate

        cprint(f"test/mean_score: {success_rate}", 'green')
        cprint(f"test/mean_return: {np.mean(all_returns)}", 'green')

        for i in range(n_epis):
            video_path = all_video_paths[i]
            if video_path is not None:
                log_data[f'test/video_seed_{self.env_seeds[i]}'] = \
                    wandb.Video(video_path)
            log_data[f'test/n_goal_{self.env_seeds[i]}'] = int(
                all_n_goal_achieved[i])
        env.reset()

        return log_data

    def close(self):
        if self._closed:
            return
        self._closed = True
        self.env.close()
