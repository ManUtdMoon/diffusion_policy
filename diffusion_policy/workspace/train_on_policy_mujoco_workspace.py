if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
import copy
import random
import wandb
import tqdm
import dill
import h5py
import numpy as np
import gymnasium
import gym
from collections import defaultdict
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.utils import explained_variance

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.residue_policy_ppo import ResiduePolicyPPO
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
from diffusion_policy.gym_util.multistep_wrapper import ObsActionSeqWrapper

OmegaConf.register_new_resolver("eval", eval, replace=True)


def collect_episode_info(infos, result=None):
    if result is None:
        result = defaultdict(list)
    for info in infos:
        if "episode" in info.keys():
            result['episode_return'].append(info["episode"]["r"])
            result['episode_length'].append(info["episode"]["l"])
    return result


@torch.no_grad()
def evaluate_policy(policy: ResiduePolicyPPO, eval_envs, n_trajs, device):
    policy.eval()
    eval_log = dict()
    eval_result = defaultdict(list)
    obs = eval_envs.reset()
    while len(eval_result['episode_return']) < n_trajs:
        obs_tensor = torch.Tensor(obs).to(device=device)
        res_naction_tensor = policy.predict_res_naction(obs_tensor, argmax=True)
        res_naction_flat = res_naction_tensor.cpu().numpy()

        env_action = res_naction_flat
        obs, rewards, dones, infos = eval_envs.step(env_action)

        collect_episode_info(infos, eval_result)

    for k, v in eval_result.items():
        eval_log[f'eval/{k}'] = np.mean(v)
    policy.train()
    print(f"Eval log: {eval_log}")
    return eval_log


class TrainOnPolicyMujocoWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'global_update']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # configure training state
        self.global_step = 0
        self.global_update = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        # configure env
        def env_fn(idx):
            def thunk():
                env = gym.make(cfg.task_name)
                env = gym.wrappers.RecordEpisodeStatistics(env)
                env = gym.wrappers.ClipAction(env)
                # env = ObsActionSeqWrapper(
                #     env,
                #     n_obs_steps=cfg.n_obs_steps,
                #     n_action_steps=cfg.n_action_steps,
                #     max_episode_steps=env.env.env._max_episode_steps,
                # )
                env = gym.wrappers.NormalizeReward(env)
                env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))

                env.action_space.seed(cfg.training.seed + idx)
                env.observation_space.seed(cfg.training.seed + idx)
                return env

            return thunk

        env_fns = [env_fn(i) for i in range(cfg.training.n_envs)]
        envs = AsyncVectorEnv(env_fns, dummy_env_fn=env_fn(0))
        envs.seed(cfg.training.seed)

        eval_env_fns = [env_fn(i + 100_000) for i in range(cfg.training.n_eval_envs)]
        eval_envs = AsyncVectorEnv(eval_env_fns, dummy_env_fn=env_fn(0))
        eval_envs.seed(cfg.training.seed + 100_000)

        ## configure res policy
        obs_dim = np.prod(envs.single_observation_space.shape).item()
        act_seq_dim = np.prod(envs.single_action_space.shape).item()
        self.res_policy: ResiduePolicyPPO = hydra.utils.instantiate(
            cfg.res_policy, obs_dim=obs_dim, action_dim=act_seq_dim)
        print(f"Residue policy with do={obs_dim}, Da={act_seq_dim}, gamma={self.res_policy.gamma}")

        # configure logging
        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging
        )
        wandb.config.update(
            {
                "output_dir": self.output_dir,
            }
        )

        # device transfer, optimizers
        device = torch.device(cfg.training.device)
        self.res_policy.to(device)

        policy_opt = self.res_policy.get_optimizer(policy_lr=cfg.training.policy_lr)
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(policy_opt, start_factor=1.0, end_factor=0.0, total_iters=cfg.training.num_steps // cfg.training.training_freq)

        if cfg.training.debug:
            cfg.training.training_freq = 1000
            cfg.training.num_steps = 5000
            cfg.training.log_every = 1000
            cfg.training.checkpoint_every = 5000
            cfg.training.eval_every = 5000

        # replay buffer
        dummy_obs_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,), dtype=np.float32
        )
        dummy_action_space = gymnasium.spaces.Box(
            low=-1.0, high=1.0,
            shape=(act_seq_dim,), dtype=np.float32
        )
        n_steps_per_rollout = int(cfg.training.training_freq // cfg.training.n_envs)
        rb = RolloutBuffer(
            n_steps_per_rollout,
            dummy_obs_space,
            dummy_action_space,
            device=device,
            n_envs=cfg.training.n_envs,
            gamma=cfg.res_policy.gamma,
            gae_lambda=cfg.res_policy.gae_lambda,
        )

        # simplify necessary cfg
        training_freq = cfg.training.training_freq
        eval_every = cfg.training.eval_every
        log_every = cfg.training.log_every
        checkpoint_every = cfg.training.checkpoint_every
        n_steps = cfg.training.num_steps
        n_envs = cfg.training.n_envs
        n_epochs = cfg.training.n_epochs
        bs = cfg.training.batch_size
        n_updates_per_training = int(training_freq * n_epochs // bs)
        res_scale = cfg.training.res_scale

        ## check parameters for code clarity
        assert (
            log_every % training_freq == 0 and
            checkpoint_every % training_freq == 0 and
            eval_every % training_freq == 0
        ), f"log_every({log_every}), checkpoint_every({checkpoint_every}), eval_every({eval_every}) must be divisible by training_freq({training_freq}) for code clarity."

        # training loop
        obs_seq = envs.reset()
        last_episode_start = np.ones((n_envs,), dtype=bool)
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as logger:
           # initial policy evaluation
            eval_log = evaluate_policy(self.res_policy, eval_envs, cfg.training.n_eval_trajs, device)
            logger.log(eval_log)
            wandb_run.log(eval_log, step=self.global_step)

            while self.global_step < n_steps:
                step_log = dict()
                # 1. collect samples
                rb.reset()
                for _ in tqdm.tqdm(
                        range(n_steps_per_rollout), 
                        desc=f"{self.global_step} / {n_steps} samples collected",
                        leave=False):
                    self.global_step += n_envs

                    # 1.1 Get exploration action
                    with torch.no_grad():
                        obs_seq_tensor = torch.Tensor(obs_seq).to(device=device)

                        # 1.1.2 Get res prediction
                        res_naction_tensor, log_prob, value = self.res_policy.predict_all(obs_seq_tensor)
                        res_naction_flat = res_naction_tensor.cpu().numpy()

                    # 1.2 env_action and step
                    env_action = res_naction_flat
                    if cfg.n_action_steps > 1:
                        env_action = res_naction_flat.reshape((n_envs, cfg.n_action_steps, -1))
                    next_obs_seq, rewards, dones, infos = envs.step(env_action)

                    train_result = collect_episode_info(infos)
                    if len(train_result.get('episode_return', [])) > 0:
                        step_log['info/episode_return'] = np.mean(train_result['episode_return'])
                        step_log['info/episode_length'] = np.mean(train_result['episode_length'])

                    # 1.3 prepare transitions for rb
                    rb.add(
                        obs=obs_seq,
                        action=res_naction_flat,
                        reward=rewards,
                        episode_start=last_episode_start,
                        value=value.flatten(),
                        log_prob=log_prob.flatten(),
                    )

                    ## switch to next step
                    obs_seq = next_obs_seq
                    last_episode_start = dones

                # 2. before training, compute value of the last step for GAE
                with torch.no_grad():
                    next_obs_seq_tensor = torch.Tensor(next_obs_seq).to(device=device)
                    next_value = self.res_policy.predict_value(next_obs_seq_tensor)
                rb.compute_returns_and_advantage(last_values=next_value.flatten(), dones=dones)

                # 3. training
                for _ in tqdm.tqdm(
                        range(n_epochs),
                        desc=f"Update {n_updates_per_training} times",
                        leave=False):
                    for batch in rb.get(bs):
                        self.global_update += 1

                        loss, info = self.res_policy.compute_loss(batch)
                        policy_opt.zero_grad()
                        loss.backward()
                        actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.res_policy.parameters(), cfg.training.max_grad_norm)
                        policy_opt.step()
                lr_scheduler.step()

                # 4. training metrics
                explained_var = explained_variance(rb.values.flatten(), rb.returns.flatten())
                step_log.update({
                    'info/global_step': self.global_step,
                    'info/global_update': self.global_update,

                    'info/value': info['value'],
                    'info/actor_entropy': info['actor_entropy'],
                    'info/approx_kl': info['approx_kl'],
                    'info/clip_frac': info['clip_frac'],
                    'info/res_naction_norm': info['res_naction_norm'],
                    'info/explained_var': explained_var,
                    'info/lr': lr_scheduler.get_last_lr()[0],

                    'loss/critic_loss': info['critic_loss'],
                    'loss/actor_loss': info['actor_loss'],
                    'loss/actor_grad_norm': actor_grad_norm.item(),
                })

                # 5. evaluation
                if self.global_step % eval_every == 0 or self.global_step >= n_steps:
                    eval_log = evaluate_policy(self.res_policy, eval_envs, cfg.training.n_eval_trajs, device)
                    step_log.update(eval_log)

                # 6. logging
                logger.log(step_log)
                if self.global_step % log_every == 0:
                    wandb_run.log(step_log, step=self.global_step)

                # 7. checkpointing
                # if self.global_step % checkpoint_every == 0 or self.global_step >= n_steps:
                #     path = pathlib.Path(self.output_dir) / 'checkpoints' / f'step={self.global_step}.ckpt'
                #     path.parent.mkdir(parents=False, exist_ok=True)
                #     payload = {
                #         'cfg': self.cfg,
                #         'res_policy': self.res_policy.state_dict(),
                #         'optimizer': policy_opt.state_dict()
                #     }
                #     torch.save(payload, path.open('wb'), pickle_module=dill)

        # envs.close()


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainOnPolicyMujocoWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
