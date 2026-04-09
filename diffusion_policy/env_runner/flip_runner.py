import copy
import h5py
import numpy as np
import pathlib
import time
import torch
import tqdm
from termcolor import cprint

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env.flip.flip_env import FlipEnv
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.policy.flow_match_vib_unet_image_policy import FlowMatchVibUnetImagePolicy



class FlipRunner(BaseImageRunner):
    def __init__(self,
        output_dir,
        eval_episodes=40,
        max_steps=250,
        n_obs_steps=1,
        n_action_steps=10,
        tqdm_interval_sec=5.0,
        mode='rel',
        key_epi_init=None
    ):
        super().__init__(output_dir)

        self.env = MultiStepWrapper(
            FlipEnv(dt=1./11, mode=mode),
            n_obs_steps=n_obs_steps,
            n_action_steps=n_action_steps,
            max_episode_steps=max_steps,
            reward_agg_method='sum',
            key_epi_init=key_epi_init,
        )

        self.eval_episodes = eval_episodes
        self.max_steps = max_steps
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.tqdm_interval_sec = tqdm_interval_sec

    def _save_episode_to_h5(self, sparse_data, dense_data, save_path, meta=None):
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(save_path, 'w') as f:
            sparse_group = f.create_group('sparse')
            obs_group = sparse_group.create_group('obs_seq')
            for key, value in sparse_data['obs_seq'].items():
                obs_group.create_dataset(
                    key,
                    data=value,
                    compression='gzip',
                    compression_opts=4
                )
            sparse_group.create_dataset(
                'chunk_action',
                data=sparse_data['chunk_action'],
                compression='gzip',
                compression_opts=4
            )
            dense_group = f.create_group('dense_raw')
            dense_group.create_dataset(
                'image',
                data=dense_data['image'],
                compression='gzip',
                compression_opts=4
            )
            dense_group.create_dataset(
                'timestamp',
                data=dense_data['timestamp'],
                compression='gzip',
                compression_opts=4
            )
            if meta is not None:
                for key, value in meta.items():
                    f.attrs[key] = value

    @torch.no_grad()
    def run(self, policy: FlowMatchVibUnetImagePolicy): 
        device = policy.device
        env = self.env
        trajectory_save_dir = pathlib.Path(self.output_dir) / 'trajectories'

        completed_episodes = 0
        all_success = []
        all_returns = []
        all_epi_len = []

        pbar = tqdm.tqdm(
            total=self.eval_episodes,
            desc=f"Eval in Flip Env",
            leave=False, mininterval=self.tqdm_interval_sec)

        while completed_episodes < self.eval_episodes:
            obs = env.reset()
            policy.reset()

            episode_obs_seq = dict()
            episode_action_chunks = list()

            def append_obs_seq(obs_seq):
                for key, value in obs_seq.items():
                    if key not in episode_obs_seq:
                        episode_obs_seq[key] = list()
                    episode_obs_seq[key].append(np.array(value, copy=True))

            append_obs_seq(obs)
            
            actual_step_count = 0
            episode_done = False 
            episode_return  = 0
            time_start = time.time()
            is_success = False
            pre_reward = -1

            while not episode_done:
                time_frame = time.time()
                
                # prepare obs
                obs_dict_input = {}
                ## filter necessary obs (may be unnecessary due to MultiStepWrapper)
                obs_dict_input['image'] = (obs['image']).astype(np.float32)
                obs_dict_input['qpos'] = (obs['qpos']).astype(np.float32)

                obs_dict = dict_apply(
                    obs_dict_input,
                    lambda x: torch.from_numpy(x).to(device=device).unsqueeze(0)
                )

                # inference
                action_dict = policy.predict_action(obs_dict)
                np_action_dict = dict_apply(
                    action_dict, lambda x: x.detach().to('cpu').numpy())
                action = np_action_dict['action'].squeeze(0)  # (Ta, Da)
                episode_action_chunks.append(np.array(action, copy=True))
                time_action = time.time()
                print('Inference time: ', time_action - time_frame)

                obs, reward, done, info = env.step(action.copy())
                print('Env step time: ', time.time() - time_action)
                if done:
                    dense_data = env.env.get_dense_recording()
                    sparse_data = {
                        'obs_seq': {
                            key: np.stack(values, axis=0)
                            for key, values in episode_obs_seq.items()
                        },
                        'chunk_action': np.stack(episode_action_chunks, axis=0)
                    }
                    if len(sparse_data['obs_seq']) > 0 and sparse_data['chunk_action'].size > 0:
                        traj_path = trajectory_save_dir / f'episode_{completed_episodes:05d}.h5'
                        self._save_episode_to_h5(
                            sparse_data=sparse_data,
                            dense_data=dense_data,
                            save_path=traj_path,
                            meta={
                                'episode_id': int(completed_episodes),
                                'success': bool(info['is_success'][-1]),
                                'episode_length': int(info['episode_length']),
                            }
                        )
                        print(f"Saved trajectory to {traj_path}")
                else:
                    append_obs_seq(obs)

                # post-process
                if reward is None:
                    reward = pre_reward

                episode_return += reward
                pre_reward = reward

                actual_step_count += 1
                episode_done = done
                print('Chunk freq:', 1 / (time.time() - time_frame))

            time_end = time.time()
            print('Avg chunk freq: ', actual_step_count / (time_end - time_start))

            completed_episodes += 1
            chunk_is_success = info['is_success']
            is_success = bool(chunk_is_success[-1])  # get last chunk info
            # cross check
            if is_success:
                assert reward > 0.5, "Success but low reward!"
                all_epi_len.append(info['episode_length'])

            all_success.append(is_success)
            all_returns.append(episode_return)

            print("Is success: ", is_success)
            print("SR till now: ", sum(all_success) / completed_episodes)
            print(f"Success till now: {sum(all_success)} / {completed_episodes}")
            if len(all_epi_len) > 0:
                print(f"Epi length till now: {sum(all_epi_len) / len(all_epi_len)}")

            env.reset_end()
            pbar.update(1)
            time.sleep(1.0)

        pbar.close()

        # log
        log_data = dict()

        all_success_rate = sum(all_success) / self.eval_episodes
        log_data['mean_sr'] = all_success_rate
        log_data['mean_return'] = np.mean(all_returns)
        log_data['mean_epi_length'] = np.mean(all_epi_len) if len(all_epi_len) > 0 else None
        cprint(f"test_mean_score: {all_success_rate}", 'green')
        cprint(f"mean_returns: {np.mean(all_returns)}", 'green')

        del env

        return log_data
