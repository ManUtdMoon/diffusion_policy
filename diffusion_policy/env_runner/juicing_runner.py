import copy
import h5py
import numpy as np
import os
import time
import torch
import tqdm
from termcolor import cprint

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env.juicing.juicing_env import JuicingEnv
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.policy.flow_match_vib_unet_image_policy import FlowMatchVibUnetImagePolicy



class JuicingRunner(BaseImageRunner):
    def __init__(self,
        output_dir,
        eval_episodes=30,
        max_steps=500,
        n_obs_steps=1,
        n_action_steps=8,
        tqdm_interval_sec=5.0,
    ):
        super().__init__(output_dir)

        self.env = MultiStepWrapper(
            JuicingEnv(),
            n_obs_steps=n_obs_steps,
            n_action_steps=n_action_steps,
            max_episode_steps=max_steps,
            reward_agg_method='sum',
        )

        self.eval_episodes = eval_episodes
        self.max_steps = max_steps
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.tqdm_interval_sec = tqdm_interval_sec

    @torch.no_grad()
    def run(self, policy: FlowMatchVibUnetImagePolicy): 
        device = policy.device
        env = self.env

        completed_episodes = 0
        all_success = []
        all_returns = []

        pbar = tqdm.tqdm(
            total=self.eval_episodes,
            desc=f"Eval in Juicing Env",
            leave=False, mininterval=self.tqdm_interval_sec)

        while completed_episodes < self.eval_episodes:
            obs = env.reset()
            policy.reset()
            
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
                obs_dict_input['state'] = (obs['state']).astype(np.float32)

                obs_dict = dict_apply(
                    obs_dict_input,
                    lambda x: torch.from_numpy(x).to(device=device).unsqueeze(0)
                )

                # inference
                action_dict = policy.predict_action(obs_dict)
                np_action_dict = dict_apply(
                    action_dict, lambda x: x.detach().to('cpu').numpy())
                action = np_action_dict['action'].squeeze(0)  # (Ta, Da)

                obs, reward, done, info = env.step(action.copy())

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

            all_success.append(is_success)
            all_returns.append(episode_return)

            print("Is success: ", is_success)
            print("SR till now: ", sum(all_success) / completed_episodes)

            env.reset_end()
            pbar.update(1)

        pbar.close()

        # log
        log_data = dict()

        all_success_rate = sum(all_success) / self.eval_episodes
        log_data['mean_sr'] = all_success_rate
        log_data['mean_return'] = np.mean(all_returns)
        cprint(f"test_mean_score: {all_success_rate}", 'green')
        cprint(f"mean_returns: {np.mean(all_returns)}", 'green')

        del env

        return log_data
