import wandb
import numpy as np
import torch
import tqdm
from termcolor import cprint
import time
import pathlib
import dill

from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
from diffusion_policy.env.metaworld.metaworld_image_wrapper import MetaWorldEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner


class MetaworldRunner(BaseImageRunner):
    def __init__(self,
            output_dir,
            eval_episodes=50,
            max_steps=200,
            n_obs_steps=1,
            n_action_steps=5,
            fps=20,
            crf=22,
            tqdm_interval_sec=5.0,
            task_name="peg-insert-side",
            n_envs=25,
            n_epi_vis=5,
            render_device_id=0,
        ):
        super().__init__(output_dir)
        self.task_name = task_name

        steps_per_render = max(20 // fps, 1)

        assert eval_episodes % n_envs == 0, "eval_episodes must be divisible by n_envs"

        def env_fn():
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    MetaWorldEnv(
                        task_name=task_name,
                        device_id=render_device_id,
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
                reward_agg_method='max',
            )
        def dummy_env_fn():
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    MetaWorldEnv(
                        task_name=task_name,
                        device_id=render_device_id,
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
                reward_agg_method='max',
            )
        env_fns = [env_fn for _ in range(n_envs)]
        env_init_fn_dills = list()

        for i in range(eval_episodes):
            render = i < n_epi_vis

            def init_fn(env, render=render):
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if render:
                    filename = pathlib.Path(output_dir).joinpath(
                        'media', f"test_{i}" + ".mp4")
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename
            env_init_fn_dills.append(dill.dumps(init_fn))

        env = AsyncVectorEnv(env_fns, dummy_env_fn=dummy_env_fn)

        self.env = env
        self.env_init_fn_dills = env_init_fn_dills
        self.eval_episodes = eval_episodes
        self.n_envs = n_envs

        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec

    def run(self, policy: BaseImagePolicy):
        device = policy.device
        dtype = policy.dtype
        env = self.env

        # plan for rollout
        n_envs = self.n_envs
        n_epis = self.eval_episodes
        n_chunks = n_epis // n_envs

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

            # start rollout
            obs = env.reset()
            policy.reset()
            done = False

            task = self.task_name
            pbar = tqdm.tqdm(total=self.max_steps, desc=f"Eval {task} {i+1} / {n_chunks}", leave=False, mininterval=self.tqdm_interval_sec)

            while not done:
                # create obs dict
                np_obs_dict = obs

                # device transfer
                obs_dict = dict_apply(np_obs_dict,
                    lambda x: torch.from_numpy(x).to(device=device))

                # run policy
                with torch.no_grad():
                    obs_dict_input = {}  # flush unused keys
                    obs_dict_input['agent_pos'] = obs_dict['agent_pos'].to(torch.float)
                    obs_dict_input['image'] = obs_dict['image'].to(torch.float)
                    action_dict = policy.predict_action(obs_dict_input)

                # device_transfer
                np_action_dict = dict_apply(action_dict,
                    lambda x: x.detach().to('cpu').numpy())
                action = np_action_dict['action']

                # step env
                obs, reward, done, info = env.step(action)

                # process stats
                for sublist, r in zip(all_rewards[global_slice], reward):
                    sublist.append(r)

                done = np.all(done)

                pbar.update(action.shape[1])
            pbar.close()

            # collect videos
            all_video_paths[global_slice] = env.render()[local_slice]

        # log
        log_data = dict()
        all_returns = np.array([np.max(r) for r in all_rewards])
        all_success_rates = np.sum(all_returns) / self.eval_episodes

        log_data['test/mean_score'] = all_success_rates

        cprint(f"test/mean_score: {all_success_rates}", 'green')

        for i in range(n_epis):
            video_path = all_video_paths[i]
            if video_path is not None:
                video = wandb.Video(video_path)
                log_data[f'test/video_{i}'] = video

            # success flag
            log_data[f'test/reward_{i}'] = int(all_returns[i])
        _ = env.reset()
        del env

        return log_data

    def close(self):
        self.env.close()
