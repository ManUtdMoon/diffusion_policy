import wandb
import numpy as np
import torch
import tqdm
from termcolor import cprint
import time

from diffusion_policy.env.adroit.adroit import AdroitEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner


class AdroitRunner(BaseImageRunner):
    def __init__(self,
            output_dir,
            eval_episodes=20,
            max_steps=300,
            n_obs_steps=1,
            n_action_steps=5,
            fps=10,
            crf=22,
            tqdm_interval_sec=5.0,
            task_name="door",
            env_num=1,
        ):
        super().__init__(output_dir)
        self.task_name = task_name
        if 'pen' in task_name:
            self.success_threshold = 20
        else:
            self.success_threshold = 25

        steps_per_render = max(10 // fps, 1)

        def env_fn():
            return MultiStepWrapper(
                AdroitEnv(
                    env_name=task_name
                ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps,
                reward_agg_method='sum',
            )

        self.eval_episodes = eval_episodes
        self.env = env_fn()

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

        all_goal_achieved = []
        all_returns = []
        hard_success = 0

        for _ in tqdm.tqdm(
            range(self.eval_episodes),
            desc=f"Eval in Adroit {self.task_name} Env",
            leave=False,
            mininterval=self.tqdm_interval_sec
        ):
            # start rollout
            obs = env.reset()
            policy.reset()

            done = False
            num_goal_achieved = 0
            episode_reward  = 0

            while not done:
                # create obs dict
                np_obs_dict = obs

                # device transfer                
                obs_dict = dict_apply(np_obs_dict,
                    lambda x: torch.from_numpy(x).to(device=device))

                # run policy
                with torch.no_grad():
                    obs_dict_input = {}  # flush unused keys
                    obs_dict_input['agent_pos'] = obs_dict['agent_pos'].unsqueeze(0).to(torch.float)
                    obs_dict_input['image'] = obs_dict['image'].unsqueeze(0).to(torch.float)
                    action_dict = policy.predict_action(obs_dict_input)

                # device_transfer
                np_action_dict = dict_apply(action_dict,
                    lambda x: x.detach().to('cpu').numpy())
                action = np_action_dict['action'].squeeze(0)

                # step env
                obs, reward, done, info = env.step(action)
                episode_reward += reward
                
                # process stats
                num_goal_achieved += np.sum(info['goal_achieved'])
                done = np.all(done)

            all_goal_achieved.append(num_goal_achieved)
            all_returns.append(episode_reward)
            if num_goal_achieved > self.success_threshold:
                hard_success += 1         

        # log
        log_data = dict()
        all_success_rates = hard_success / self.eval_episodes

        log_data['test/mean_return'] = np.mean(all_returns)
        log_data['test/mean_n_goal_achieved'] = np.mean(all_goal_achieved)
        log_data['test/mean_score'] = all_success_rates

        cprint(f"test/mean_score: {all_success_rates}", 'green')
        cprint(f"test/mean_return: {np.mean(all_returns)}", 'green')

        _ = env.reset()
        del env

        return log_data
