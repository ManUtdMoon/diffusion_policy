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
import gym
import gymnasium
import collections
from stable_baselines3.common.buffers import ReplayBuffer

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.flow_match_vib_unet_image_policy import FlowMatchVibUnetImagePolicy
from diffusion_policy.policy.residue_policy import ResiduePolicy
from diffusion_policy.policy.sum_policy import SumPolicy
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.vision.crop_randomizer import CropRandomizerV2
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.env.adroit.adroit import AdroitEnv, AdroitEarlyStopWrapper
from diffusion_policy.env.metaworld.metaworld_image_wrapper import MetaWorldEnv, MetaworldEarlyStopWrapper

OmegaConf.register_new_resolver("eval", eval, replace=True)

class TrainOnlineWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'global_update', 'base_ckpt']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure training state
        self.global_step = 0
        self.global_update = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        # configure policies
        ## load base policy
        base_payload = torch.load(open(cfg.online_task.base_ckpt, 'rb'), pickle_module=dill)
        base_cfg = base_payload['cfg']
        assert base_cfg.task_name == cfg.task_name, \
            f"Base policy task {base_cfg.task_name} does not match current task {cfg.task_name}"
        base_cfg.policy.n_action_steps = cfg.n_action_steps # may be different
        self.base_policy: FlowMatchVibUnetImagePolicy
        self.base_policy = hydra.utils.instantiate(base_cfg.policy)
        self.base_policy.load_state_dict(base_payload['state_dicts']['ema_model'])
        print(f"Loaded base policy from {cfg.online_task.base_ckpt}")
        self.base_policy.eval()
        self.base_policy.requires_grad_(False)

        crop_randomizers = list()
        for m in self.base_policy.modules():
            if isinstance(m, CropRandomizerV2):
                crop_randomizers.append(m)
        def set_rand_crop(mode):
            for m in crop_randomizers:
                m.force_random_crop = mode

        ## configure res policy
        To = cfg.n_obs_steps
        Ta = cfg.n_action_steps
        do = self.base_policy.obs_feature_dim
        da = cfg.shape_meta.action.shape[0]
        Da = Ta * da  # action chunk dim

        self.res_policy: ResiduePolicy = hydra.utils.instantiate(
            cfg.res_policy, obs_dim=do, action_dim=Da)
        print(f"Residue policy with do={do}, Da={Da}, gamma={self.res_policy.gamma}")

        ## sum policy
        sum_policy = SumPolicy(
            res_scale=cfg.training.res_scale,
            obs_emb_dim=do,
            action_dim=da,
            n_action_steps=cfg.n_action_steps,
            base_policy=self.base_policy,
            res_policy=self.res_policy
        )

        # configure env
        ## eval, only average score needed
        eval_env_runner: BaseImageRunner = hydra.utils.instantiate(
            cfg.online_task.env_runner,
            output_dir=self.output_dir)
        ## train
        env_type = "adroit" if "adroit" in cfg.task_name.lower() else "metaworld"
        max_steps = cfg.online_task.env_runner.max_steps
        if env_type == 'adroit':
            max_steps //= 2 # repeat = 2 in adroit

        task_name = cfg.online_task.task_name
        render_device_id = cfg.online_task.env_runner.render_device_id

        def make_env_fn(env_type, task_name, render_device_id, n_obs_steps, n_action_steps, max_steps):
            def env_fn():
                if env_type == 'adroit':
                    env = AdroitEarlyStopWrapper(AdroitEnv(
                        env_name=task_name,
                        render_device_id=render_device_id,
                    ))
                elif env_type == 'metaworld':
                    env = MetaworldEarlyStopWrapper(MetaWorldEnv(
                        task_name=task_name,
                        device_id=render_device_id,
                    ))
                else:
                    raise ValueError(f"Unsupported env_type: {env_type}")
                
                return MultiStepWrapper(
                    env,
                    n_obs_steps=n_obs_steps,
                    n_action_steps=n_action_steps,
                    max_episode_steps=max_steps,
                    reward_agg_method='discounted_sum',
                )
            return env_fn

        env_fn = make_env_fn(env_type, task_name, render_device_id, To, Ta, max_steps)
        dummy_env_fn = make_env_fn(env_type, task_name, render_device_id, To, Ta, max_steps)
        
        env_fns = [env_fn for _ in range(cfg.training.n_envs)]
        envs = AsyncVectorEnv(env_fns, dummy_env_fn=dummy_env_fn)

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
        self.base_policy.to(device)
        self.res_policy.to(device)

        optimizers = self.res_policy.get_optimizer(
            policy_lr=cfg.training.policy_lr,
            q_lr=cfg.training.q_lr
        )
        q_opt = optimizers['q_optimizer']
        actor_opt = optimizers['actor_optimizer']
        alpha_opt = optimizers['alpha_optimizer']

        # replay buffer
        dummy_obs_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(do,), dtype=np.float32
        )
        dummy_buf_action_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(Da * 3,), dtype=np.float32
        )
        rb = ReplayBuffer(
            cfg.training.buffer_size,
            dummy_obs_space,
            dummy_buf_action_space,
            device=device,
            n_envs=cfg.training.n_envs,
            handle_timeout_termination=False,
        )

        if cfg.training.debug:
            cfg.training.num_steps = 5000
            cfg.training.prog_explore = 1000
            cfg.training.learning_start = 1000
            cfg.training.checkpoint_every = 1000
            cfg.training.eval_every = 5000
            cfg.training.log_every = 1000

        # simplify necessary cfg
        training_freq = cfg.training.training_freq
        log_every = cfg.training.log_every
        eval_every = cfg.training.eval_every
        checkpoint_every = cfg.training.checkpoint_every
        utd = cfg.training.utd
        n_steps = cfg.training.num_steps
        n_envs = cfg.online_task.n_envs
        n_updates_per_training = int(training_freq * utd)
        learning_start = cfg.training.learning_start
        res_scale = cfg.training.res_scale

        ## check parameters for code clarity
        assert (
            log_every % training_freq == 0 and
            eval_every % training_freq == 0 and
            checkpoint_every % training_freq == 0 and
            learning_start % training_freq == 0 and
            eval_every % log_every == 0
        ), f"log_every({log_every}), eval_every({eval_every}), checkpoint_every({checkpoint_every}), learning_start({learning_start}) must be divisible by training_freq({training_freq}) for code clarity."

        # training loop
        recent_done_successes = collections.deque(maxlen=100)
        def get_recent_success_stats():
            count = len(recent_done_successes)
            rate = float(np.mean(recent_done_successes)) if count > 0 else 0.0
            return count, rate

        SUCCESS_TRHES = cfg.online_task.success_threshold

        obs_seq = envs.reset()
        obs_seq_tensor = dict_apply(
            obs_seq, lambda x: torch.from_numpy(x).to(device=device))
        base_dict = self.base_policy.predict_action(obs_seq_tensor)
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as logger:
            set_rand_crop(False)
            sum_policy.eval()
            eval_log = eval_env_runner.run(sum_policy)
            sum_policy.train()
            set_rand_crop(True)
            logger.log(eval_log)
            wandb_run.log(eval_log, step=self.global_step)

            while self.global_step < n_steps:
                step_log = dict()
                # collect samples
                for _ in tqdm.tqdm(
                        range(training_freq // n_envs), 
                        desc=f"{self.global_step} / {n_steps} samples collected",
                        leave=False):
                    self.global_step += n_envs

                    ## retrieve from base cache
                    obs_emb_tensor = base_dict['obs_emb'][:, -do:].detach()
                    base_naction_tensor = base_dict['naction'].detach()
                    base_naction_flat = base_naction_tensor.flatten(start_dim=1).cpu().numpy()  # (B, Ta*da)

                    ## pi-dec progressive exploration
                    res_ratio = min(
                        max(self.global_step, 0) / cfg.training.prog_explore, 1)
                    ## uncomment to disable progressive exploration
                    # res_ratio = 1.0

                    ## prepare masks for progressive exploration
                    if self.global_step < learning_start:
                        # Mask all residues during warmup phase: base only
                        res_masks = torch.ones(n_envs, device=device, dtype=torch.bool)
                    else:
                        # Progressive exploration: pi-dec style
                        res_masks = torch.rand(n_envs, device=device) >= res_ratio

                    ## forward sum policy with cached base info
                    sum_dict = sum_policy.predict_train_action(
                        base_naction_tensor,
                        obs_emb_tensor,
                        res_mask=res_masks
                    )
                    sum_dict = dict_apply(
                        sum_dict, lambda x: x.detach().cpu().numpy())
                    res_naction_flat = sum_dict['res_naction_flat']
                    action = sum_dict['action']

                    ## env_action and step
                    next_obs_seq, rewards, dones, infos = envs.step(action.copy())
                    for reward, done, info in zip(rewards, dones, infos):
                        if done:
                            if env_type == 'adroit':
                                recent_done_successes.append(
                                    info["accumulated_goal_achieved"] >= SUCCESS_TRHES
                                )
                            elif env_type == 'metaworld':
                                recent_done_successes.append(
                                    reward >= SUCCESS_TRHES
                                )

                    ## reward preprocess
                    # rewards *= cfg.training.reward_scale

                    ## prepare transitions for rb
                    next_obs_seq_tensor = dict_apply(
                        next_obs_seq, lambda x: torch.from_numpy(x).to(device=device))
                    next_base_dict = self.base_policy.predict_action(next_obs_seq_tensor)

                    if cfg.training.bootstrap_at_done == 'never':
                        next_obs_emb_tensor = next_base_dict['obs_emb'][:, -do:].detach()
                        next_base_naction_tensor = next_base_dict['naction'].detach()

                        stop_bootstrap = dones
                    else:
                        assert cfg.training.bootstrap_at_done == 'truncated'
                        terminations = dones
                        truncations = np.array(
                            [d.get('TimeLimit.truncated', [False])[0] for d in infos], 
                            dtype=bool
                        )

                        stop_bootstrap = np.logical_and(terminations, np.logical_not(truncations))

                        real_next_obs = {
                            k: v.copy() for k, v in next_obs_seq.items()
                        }
                        for i, need in enumerate(truncations):
                            if need:
                                for k in next_obs_seq.keys():
                                    real_next_obs[k][i] = infos[i]['final_observation'][k]
                        
                        # for real obs in buffer
                        real_next_obs_tensor = dict_apply(
                            real_next_obs,
                            lambda x: torch.from_numpy(x).to(device=device))
                        real_next_base_dict = self.base_policy.predict_action(
                            real_next_obs_tensor)
                        next_obs_emb_tensor = real_next_base_dict['obs_emb'][:, -do:].detach()
                        next_base_naction_tensor = real_next_base_dict['naction'].detach()

                    actions_to_save = np.concatenate(
                        [
                            res_naction_flat,
                            base_naction_flat,
                            next_base_naction_tensor.flatten(start_dim=1).cpu().numpy()
                        ],
                        axis=-1
                    )

                    rb.add(
                        obs=obs_emb_tensor.cpu().numpy(),
                        next_obs=next_obs_emb_tensor.cpu().numpy(),
                        action=actions_to_save,
                        reward=rewards,
                        done=stop_bootstrap,
                        infos=[{}]
                    )

                    ## switch to next step
                    base_dict = next_base_dict

                if self.global_step < learning_start:
                    # Warmup phase: skip training, only collect data
                    continue

                # training
                for _ in tqdm.tqdm(
                        range(n_updates_per_training),
                        desc=f"Update {n_updates_per_training} times",
                        leave=False):
                    self.global_update += 1

                    ## fetch data
                    batch = rb.sample(cfg.training.batch_size)

                    ## update critics
                    critic_loss, critic_info = self.res_policy.compute_critic_loss(batch, None)
                    q_opt.zero_grad()
                    critic_loss.backward()
                    q1_grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.res_policy.qs.parameters(), cfg.training.max_grad_norm)
                    q_opt.step()

                    ## update target
                    if self.global_update % cfg.training.target_freq == 0:
                        self.res_policy.target_update()
                    
                    ## update policy
                    if self.global_update % cfg.training.policy_freq == 0:
                        actor_loss, actor_info = self.res_policy.compute_actor_loss(batch)
                        actor_opt.zero_grad()
                        actor_loss.backward()
                        actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                            self.res_policy.actor.parameters(),
                            cfg.training.max_grad_norm
                        )
                        actor_opt.step()

                        alpha = self.res_policy.init_alpha
                        if cfg.res_policy.auto_alpha:
                            alpha_loss = self.res_policy.compute_alpha_loss(batch)
                            alpha_opt.zero_grad()
                            alpha_loss.backward()
                            alpha_opt.step()
                            alpha = self.res_policy.log_alpha.exp().item()

                ## training metrics
                recent_done_count, recent_done_sr = get_recent_success_stats()

                step_log = {
                    'info/global_step': self.global_step,
                    'info/global_update': self.global_update,

                    'info/res_ratio': res_ratio,
                    'info/q_target': critic_info['q_target'],
                    'info/q_predicted': critic_info['q_predicted'],
                    'info/q_predicted_min': critic_info['q_predicted_min'],
                    'info/q_predicted_max': critic_info['q_predicted_max'],
                    'info/actor_entropy': actor_info['actor_entropy'],
                    'info/rewards': critic_info['rewards'],
                    'info/reward_max': critic_info['reward_max'],
                    'info/reward_min': critic_info['reward_min'],
                    'info/dones': critic_info['dones'],
                    'info/res_naction_norm': actor_info['res_naction_norm'],
                    'info/base_norm': actor_info['base_norm'],
                    'info/recent_done_sr': recent_done_sr,
                    'info/recent_done_count': recent_done_count,

                    'loss/critic_loss': critic_loss.item() / 2.0,
                    'loss/actor_loss': actor_loss.item(),
                    'loss/q1_grad_norm': q1_grad_norm.item(),
                    'loss/actor_grad_norm': actor_grad_norm.item(),
                    'loss/alpha': alpha,
                }
                if cfg.res_policy.auto_alpha:
                    step_log['loss/alpha_loss'] = alpha_loss.item()

                # evaluation
                sum_policy.eval()
                if self.global_step > 0 and self.global_step % eval_every == 0:
                    set_rand_crop(False)
                    eval_log = eval_env_runner.run(sum_policy)
                    step_log.update(eval_log)
                    set_rand_crop(True)
                sum_policy.train()

                # logging
                logger.log(step_log)
                if self.global_step % log_every == 0:
                    wandb_run.log(step_log, step=self.global_step)

                # checkpointing
                # if self.global_step % checkpoint_every == 0:
                #     path = pathlib.Path(self.output_dir) / 'checkpoints' / f'step={self.global_step}.ckpt'
                #     path.parent.mkdir(parents=False, exist_ok=True)
                #     payload = {
                #         'cfg': self.cfg,
                #         'res_policy': self.res_policy.state_dict(),
                #     }
                #     torch.save(payload, path.open('wb'), pickle_module=dill)


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")),
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainOnlineWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
