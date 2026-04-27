import os
import wandb
import numpy as np
import torch
import collections
import pathlib
import tqdm
import h5py
import math
import dill

from zprl.gym_util.async_vector_env import AsyncVectorEnv
from zprl.gym_util.multistep_wrapper import SingleStepStackedObsWrapper
from zprl.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder
from zprl.model.common.rotation_transformer import RotationTransformer

from zprl.policy.base_image_policy import BaseImagePolicy
from zprl.common.pytorch_util import dict_apply
from zprl.env_runner.base_image_runner import BaseImageRunner
from zprl.env_runner.robomimic_image_runner import (
    _find_wrapper,
    _get_env_shape_meta,
    create_env,
)
from zprl.env.robomimic.robomimic_image_wrapper import RobomimicImageWrapper
from zprl.env.robomimic.robomimic_image_relative_wrapper import RobomimicImageRelativeWrapper
import robomimic.utils.file_utils as FileUtils


class RobomimicDelayImageRunner(BaseImageRunner):
    """
    Robomimic image runner with single-step env stepping and fixed action delay.
    """

    def __init__(self,
            output_dir,
            dataset_path,
            shape_meta:dict,
            n_train=10,
            n_train_vis=3,
            train_start_idx=0,
            n_test=22,
            n_test_vis=6,
            test_start_seed=10000,
            max_steps=400,
            n_obs_steps=2,
            n_action_steps=8,
            render_obs_key='agentview_image',
            fps=10,
            crf=22,
            past_action=False,
            abs_action=False,
            action_delay_steps=0,
            rtc_mode=False,
            prefix_attention_schedule='exp',
            max_guidance_weight=5.0,
            action_pose_repr=None,
            tqdm_interval_sec=5.0,
            n_envs=None
        ):
        super().__init__(output_dir)

        if n_envs is None:
            n_envs = n_train + n_test

        # assert n_obs_steps <= n_action_steps
        dataset_path = os.path.expanduser(dataset_path)
        robosuite_fps = 20
        steps_per_render = max(robosuite_fps // fps, 1)

        # read from dataset
        env_meta = FileUtils.get_env_metadata_from_dataset(
            dataset_path)
        # disable object state observation
        env_meta['env_kwargs']['use_object_obs'] = False

        if action_pose_repr is None:
            action_pose_repr = 'abs' if abs_action else 'delta'
        env_shape_meta = _get_env_shape_meta(shape_meta, action_pose_repr)
        rotation_transformer = None
        if action_pose_repr in ('abs', 'relative'):
            env_meta['env_kwargs']['controller_configs']['control_delta'] = False
        if action_pose_repr == 'abs':
            rotation_transformer = RotationTransformer('axis_angle', 'rotation_6d')

        def env_fn():
            robomimic_env = create_env(
                env_meta=env_meta,
                shape_meta=env_shape_meta
            )
            # Robosuite's hard reset causes excessive memory consumption.
            # Disabled to run more envs.
            # https://github.com/ARISE-Initiative/robosuite/blob/92abf5595eddb3a845cd1093703e5a3ccd01e77e/robosuite/environments/base.py#L247-L248
            robomimic_env.env.hard_reset = False
            wrapped_env = SingleStepStackedObsWrapper(
                VideoRecordingWrapper(
                    RobomimicImageWrapper(
                        env=robomimic_env,
                        shape_meta=env_shape_meta,
                        init_state=None,
                        render_obs_key=render_obs_key
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
                    steps_per_render=steps_per_render
                ),
                n_obs_steps=n_obs_steps,
                max_episode_steps=max_steps
            )
            if action_pose_repr == 'relative':
                wrapped_env = RobomimicImageRelativeWrapper(
                    env=wrapped_env,
                    shape_meta=shape_meta
                )
            return wrapped_env
        
        # For each process the OpenGL context can only be initialized once
        # Since AsyncVectorEnv uses fork to create worker process,
        # a separate env_fn that does not create OpenGL context (enable_render=False)
        # is needed to initialize spaces.
        def dummy_env_fn():
            robomimic_env = create_env(
                    env_meta=env_meta,
                    shape_meta=env_shape_meta,
                    enable_render=False
                )
            wrapped_env = SingleStepStackedObsWrapper(
                VideoRecordingWrapper(
                    RobomimicImageWrapper(
                        env=robomimic_env,
                        shape_meta=env_shape_meta,
                        init_state=None,
                        render_obs_key=render_obs_key
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
                    steps_per_render=steps_per_render
                ),
                n_obs_steps=n_obs_steps,
                max_episode_steps=max_steps
            )
            if action_pose_repr == 'relative':
                wrapped_env = RobomimicImageRelativeWrapper(
                    env=wrapped_env,
                    shape_meta=shape_meta
                )
            return wrapped_env

        env_fns = [env_fn] * n_envs
        env_seeds = list()
        env_prefixs = list()
        env_init_fn_dills = list()

        # train
        with h5py.File(dataset_path, 'r') as f:
            for i in range(n_train):
                train_idx = train_start_idx + i
                enable_render = i < n_train_vis
                init_state = f[f'data/demo_{train_idx}/states'][0]

                def init_fn(env, init_state=init_state,
                    enable_render=enable_render):
                    video_wrapper = _find_wrapper(env, VideoRecordingWrapper)
                    video_wrapper.video_recoder.stop()
                    video_wrapper.file_path = None
                    if enable_render:
                        filename = pathlib.Path(output_dir).joinpath(
                            'media', f"train_{train_idx}" + ".mp4")
                        filename.parent.mkdir(parents=False, exist_ok=True)
                        filename = str(filename)
                        video_wrapper.file_path = filename

                    # switch to init_state reset
                    image_wrapper = _find_wrapper(env, RobomimicImageWrapper)
                    image_wrapper.init_state = init_state

                env_seeds.append(train_idx)
                env_prefixs.append('train/')
                env_init_fn_dills.append(dill.dumps(init_fn))
        
        # test
        for i in range(n_test):
            seed = test_start_seed + i
            enable_render = i < n_test_vis

            def init_fn(env, seed=seed,
                enable_render=enable_render):
                # setup rendering
                # video_wrapper
                video_wrapper = _find_wrapper(env, VideoRecordingWrapper)
                video_wrapper.video_recoder.stop()
                video_wrapper.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        'media', f"test_{seed}" + ".mp4")
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
                    video_wrapper.file_path = filename

                # switch to seed reset
                image_wrapper = _find_wrapper(env, RobomimicImageWrapper)
                image_wrapper.init_state = None
                env.seed(seed)

            env_seeds.append(seed)
            env_prefixs.append('test/')
            env_init_fn_dills.append(dill.dumps(init_fn))

        env = AsyncVectorEnv(env_fns, dummy_env_fn=dummy_env_fn)

        self.env_meta = env_meta
        self.env = env
        self.env_fns = env_fns
        self.env_seeds = env_seeds
        self.env_prefixs = env_prefixs
        self.env_init_fn_dills = env_init_fn_dills
        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.action_delay_steps = action_delay_steps
        self.rtc_mode = rtc_mode
        self.prefix_attention_schedule = prefix_attention_schedule
        self.max_guidance_weight = max_guidance_weight
        self.past_action = past_action
        self.max_steps = max_steps
        self.rotation_transformer = rotation_transformer
        self.abs_action = abs_action
        self.action_pose_repr = action_pose_repr
        self.tqdm_interval_sec = tqdm_interval_sec

    def run(self, policy: BaseImagePolicy):
        device = policy.device
        env = self.env
        
        # plan for rollout
        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)

        # allocate data
        all_video_paths = [None] * n_inits
        all_rewards = [[] for _ in range(n_inits)] # guarantee independent lists

        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            this_global_slice = slice(start, end)
            this_n_active_envs = end - start
            this_local_slice = slice(0,this_n_active_envs)

            this_init_fns = self.env_init_fn_dills[this_global_slice]
            n_diff = n_envs - len(this_init_fns)
            if n_diff > 0:
                this_init_fns.extend([self.env_init_fn_dills[0]]*n_diff)
            assert len(this_init_fns) == n_envs

            # init envs
            env.call_each('run_dill_function',
                args_list=[(x,) for x in this_init_fns])

            # start rollout
            obs = env.reset()
            policy.reset()

            env_name = self.env_meta['env_name']
            pbar = tqdm.tqdm(total=self.max_steps, desc=f"Eval {env_name}DelayImage {chunk_idx+1}/{n_chunks}",
                leave=False, mininterval=self.tqdm_interval_sec)

            done = False
            d = self.action_delay_steps
            Ta = self.n_action_steps
            current_chunk, nchunk = self._predict_action_chunk(policy, obs, device)
            exec_start = 0
            exec_end = Ta + d
            exec_idx = exec_start
            next_chunk = None
            next_nchunk = None
            prefix_attention_horizon = None
            if self.rtc_mode:
                if nchunk is None:
                    raise RuntimeError("Policy result must include naction_pred_all for RTC eval.")
                prefix_attention_horizon = nchunk.shape[1] - (Ta + d)
                if prefix_attention_horizon < d:
                    raise RuntimeError(
                        f"Invalid RTC horizons: chunk_len={nchunk.shape[1]}, "
                        f"execute_horizon={Ta + d}, action_delay_steps={d}.")

            while not done:
                if (exec_idx == exec_start + Ta) and (next_chunk is None):
                    rtc_context = None
                    if self.rtc_mode:
                        rtc_context = {
                            'prev_naction_chunk': nchunk,
                            'inference_delay': self.action_delay_steps,
                            'prefix_attention_horizon': prefix_attention_horizon,
                            'prefix_attention_schedule': self.prefix_attention_schedule,
                            'max_guidance_weight': self.max_guidance_weight,
                        }
                    next_chunk, next_nchunk = self._predict_action_chunk(
                        policy, obs, device, rtc_context=rtc_context)

                if exec_idx >= exec_end:
                    current_chunk = next_chunk
                    nchunk = next_nchunk
                    next_chunk = None
                    next_nchunk = None
                    exec_start = d
                    exec_end = d + Ta + d
                    exec_idx = exec_start

                action = current_chunk[:, exec_idx]
                if not np.all(np.isfinite(action)):
                    print(action)
                    raise RuntimeError("Nan or Inf action")
                
                # step env
                env_action = action
                if self.action_pose_repr == 'abs':
                    env_action = self.undo_transform_action(action)

                obs, reward, done, info = env.step(env_action)

                # collect rewards moved here
                for sublist, r in zip(all_rewards[this_global_slice], reward):
                    sublist.append(r)

                done = np.all(done)
                exec_idx += 1
                pbar.update(1)
            pbar.close()

            # collect data for this round
            all_video_paths[this_global_slice] = env.render()[this_local_slice]
        _ = env.reset()
        
        # log
        max_rewards = collections.defaultdict(list)
        log_data = dict()
        for i in range(n_inits):
            seed = self.env_seeds[i]
            prefix = self.env_prefixs[i]
            max_reward = np.max(all_rewards[i])
            max_rewards[prefix].append(max_reward)
            log_data[prefix+f'sim_max_reward_{seed}'] = max_reward

            # visualize sim
            video_path = all_video_paths[i]
            if video_path is not None:
                sim_video = wandb.Video(video_path)
                log_data[prefix+f'sim_video_{seed}'] = sim_video
        
        # log aggregate metrics
        for prefix, value in max_rewards.items():
            name = prefix+'mean_score'
            value = np.mean(value)
            log_data[name] = value

        log_data['action_delay_steps'] = self.action_delay_steps
        log_data['rtc_mode'] = self.rtc_mode
        return log_data

    def _predict_action_chunk(self, policy, obs, device, rtc_context=None):
        np_obs_dict = dict(obs)
        obs_dict = dict_apply(np_obs_dict,
            lambda x: torch.from_numpy(x).to(
                device=device))

        with torch.no_grad():
            if rtc_context is None:
                action_dict = policy.predict_action(obs_dict)
            else:
                rtc_context = dict_apply(
                    rtc_context, lambda x: torch.from_numpy(x).to(device=device)
                    if isinstance(x, np.ndarray) else x)
                action_dict = policy.predict_action(obs_dict, rtc_context=rtc_context)

        np_action_dict = dict_apply(action_dict,
            lambda x: x.detach().to('cpu').numpy())

        if 'action_pred_all' not in np_action_dict:
            raise RuntimeError("Policy result must include action_pred_all for delay eval.")

        action = np_action_dict['action_pred_all']
        required_len = self.n_action_steps + 2 * self.action_delay_steps
        if action.shape[1] < required_len:
            raise RuntimeError(
                f"action_pred_all length {action.shape[1]} is shorter than "
                f"required delay window {required_len}.")
        naction = np_action_dict.get('naction_pred_all', None)
        return action, naction

    def undo_transform_action(self, action):
        raw_shape = action.shape
        if raw_shape[-1] == 20:
            # dual arm
            action = action.reshape(-1,2,10)

        d_rot = action.shape[-1] - 4
        pos = action[...,:3]
        rot = action[...,3:3+d_rot]
        gripper = action[...,[-1]]
        rot = self.rotation_transformer.inverse(rot)
        uaction = np.concatenate([
            pos, rot, gripper
        ], axis=-1)

        if raw_shape[-1] == 20:
            # dual arm
            uaction = uaction.reshape(*raw_shape[:-1], 14)

        return uaction
