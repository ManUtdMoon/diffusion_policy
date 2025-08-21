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
import wandb.sdk.data_types.video as wv

from diffusion_policy.env_runner.robomimic_image_runner import RobomimicImageRunner
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.pytorch_util import dict_apply


class RobomimicImageRunnerWithUncertainty(RobomimicImageRunner):
    """
    Robomimic envs already enforces number of steps.
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
            tqdm_interval_sec=5.0,
            n_envs=None
        ):
        super().__init__(output_dir, dataset_path, shape_meta, n_train, n_train_vis,
            train_start_idx, n_test, n_test_vis, test_start_seed, max_steps,
            n_obs_steps, n_action_steps, render_obs_key, fps, crf, past_action,
            abs_action, tqdm_interval_sec, n_envs)

    def run_with_uncertainty(self, policy: BaseImagePolicy, demo_obs_emb):
        device = policy.device
        dtype = policy.dtype
        env = self.env
        obs_emb_dim = policy.obs_feature_dim
        low_dims = sum([policy.obs_encoder.key_shape_map[k][0] for k in policy.obs_encoder.low_dim_keys])
        rgb_emb_dim = obs_emb_dim - low_dims

        # plan for rollout
        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)

        # allocate data
        all_video_paths = [None] * n_inits
        all_rewards = [[] for _ in range(n_inits)] # guarantee independent lists
        all_uncertainty = [[] for _ in range(n_inits)]

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
            past_action = None
            policy.reset()

            env_name = self.env_meta['env_name']
            pbar = tqdm.tqdm(total=self.max_steps, desc=f"Eval {env_name}Image {chunk_idx+1}/{n_chunks}", 
                leave=False, mininterval=self.tqdm_interval_sec)
            
            done = False
            assert self.max_steps % self.n_action_steps == 0, \
                f"T={self.max_steps} must be divisible by Ta={self.n_action_steps}"
            T_sub = self.max_steps // self.n_action_steps
            for t in range(T_sub):
                # create obs dict
                np_obs_dict = dict(obs)
                if self.past_action and (past_action is not None):
                    # TODO: not tested
                    np_obs_dict['past_action'] = past_action[
                        :,-(self.n_obs_steps-1):].astype(np.float32)
                
                # device transfer
                obs_dict = dict_apply(np_obs_dict, 
                    lambda x: torch.from_numpy(x).to(
                        device=device))

                # run policy
                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict)

                # device_transfer
                np_action_dict = dict_apply(action_dict,
                    lambda x: x.detach().to('cpu').numpy())

                action = np_action_dict['action']
                if not np.all(np.isfinite(action)):
                    print(action)
                    raise RuntimeError("Nan or Inf action")
                
                # dist-to-data starts
                obs_emb = action_dict['obs_emb'][..., -obs_emb_dim:-obs_emb_dim + rgb_emb_dim] # (B,di)
                with torch.no_grad():
                    dist_to_data = torch.cdist(obs_emb, demo_obs_emb) # (B,N)
                    uncertainty = dist_to_data.min(dim=1).values # (B,)
                    uncertainty = uncertainty.cpu().numpy()
                # dist-to-data ends

                # flow loss starts
                # with torch.no_grad():
                #     naction_pred = action_dict['naction_pred'] # (B,H,Da)
                # flow_loss ends

                # step env
                env_action = action
                if self.abs_action:
                    env_action = self.undo_transform_action(action)

                obs, reward, done, info = env.step(env_action)

                # collect rewards moved here
                for sublist, r in zip(all_rewards[this_global_slice], reward):
                    sublist.append(r)
                # collect uncertainty scores
                for sublist, u in zip(all_uncertainty[this_global_slice], uncertainty):
                    sublist.append(u)

                done = np.all(done)
                past_action = action

                # update pbar
                pbar.update(action.shape[1])
            pbar.close()
            assert done

            # collect data for this round
            all_video_paths[this_global_slice] = env.render()[this_local_slice]
            # all_rewards[this_global_slice] = env.call('get_attr', 'reward')[this_local_slice]
        # clear out video buffer
        _ = env.reset()
        
        # log
        max_rewards = collections.defaultdict(list)
        log_data = dict()
        log_data['failure'] = [] # (n_inits,)
        log_data['uncertainty'] = all_uncertainty # (n_inits, T_sub)
        # results reported in the paper are generated using the commented out line below
        # which will only report and average metrics from first n_envs initial condition and seeds
        # fortunately this won't invalidate our conclusion since
        # 1. This bug only affects the variance of metrics, not their mean
        # 2. All baseline methods are evaluated using the same code
        # to completely reproduce reported numbers, uncomment this line:
        # for i in range(len(self.env_fns)):
        # and comment out this line
        for i in range(n_inits):
            seed = self.env_seeds[i]
            prefix = self.env_prefixs[i]
            max_reward = np.max(all_rewards[i])
            max_rewards[prefix].append(max_reward)
            log_data[prefix+f'sim_max_reward_{seed}'] = max_reward
            log_data['failure'].append(max_reward < 0.5)

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

        return log_data
