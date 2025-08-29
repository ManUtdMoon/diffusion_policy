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
import gymnasium as gym
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.utils import explained_variance

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.flow_match_unet_image_policy import FlowMatchUnetImagePolicy
from diffusion_policy.policy.residue_policy_ppo import ResiduePolicyPPO
from diffusion_policy.policy.sum_policy import SumPolicy
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
from diffusion_policy.env_runner.robomimic_image_runner import create_env
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.env.robomimic.robomimic_image_wrapper import RobomimicImageWrapper
import robomimic.utils.file_utils as FileUtils

OmegaConf.register_new_resolver("eval", eval, replace=True)

class TrainOnPolicyRobomimicWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'global_update', 'base_ckpt']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

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
        self.base_policy: FlowMatchUnetImagePolicy
        self.base_policy = hydra.utils.instantiate(base_cfg.policy)
        self.base_policy.load_state_dict(base_payload['state_dicts']['ema_model'])
        print(f"Loaded base policy from {cfg.online_task.base_ckpt}")
        self.base_policy.eval()
        self.base_policy.requires_grad_(False)

        ## load demo_obs_emb from base policy
        checkpoint = cfg.online_task.base_ckpt
        ckpt_dir = pathlib.Path(checkpoint).parent
        ckpt_name = pathlib.Path(checkpoint).stem
        demo_emb_path = ckpt_dir / f'{ckpt_name}_obs_emb_action.pt'
        demo_emb = torch.load(open(demo_emb_path, 'rb'), pickle_module=dill)

        ## configure res policy
        obs_emb_dim = self.base_policy.obs_feature_dim # do, Do=To*do
        act_dim = cfg.shape_meta.action.shape[0] # da
        act_seq_dim = cfg.n_action_steps * act_dim # Da=Ta*da
        self.res_policy: ResiduePolicyPPO = hydra.utils.instantiate(
            cfg.res_policy, obs_dim=obs_emb_dim, action_dim=act_seq_dim)
        print(f"Residue policy with do={obs_emb_dim}, Da={act_seq_dim}, gamma={self.res_policy.gamma}")

        ## sum policy
        sum_policy = SumPolicy(
            res_scale=cfg.training.res_scale,
            obs_emb_dim=obs_emb_dim,
            action_dim=act_dim,
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
        ### fetch env_meta
        dataset_path = os.path.expanduser(cfg.online_task.dataset_path)
        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
        env_meta['env_kwargs']['use_object_obs'] = False
        if cfg.online_task.abs_action:
            env_meta['env_kwargs']['controller_configs']['control_delta'] = False
        
        shape_meta = cfg.shape_meta

        def env_fn():
            robomimic_env = create_env(
                env_meta=env_meta, 
                shape_meta=shape_meta
            )
            robomimic_env.env.hard_reset = False
            return MultiStepWrapper(
                RobomimicImageWrapper(
                    env=robomimic_env,
                    shape_meta=shape_meta,
                    init_state=None,
                    render_obs_key=cfg.online_task.env_runner.render_obs_key
                ),
                n_obs_steps=cfg.n_obs_steps,
                n_action_steps=cfg.n_action_steps,
                max_episode_steps=cfg.online_task.env_runner.max_steps
            )
        def dummy_env_fn():
            robomimic_env = create_env(
                env_meta=env_meta, 
                shape_meta=shape_meta,
                enable_render=False
            )
            return MultiStepWrapper(
                RobomimicImageWrapper(
                    env=robomimic_env,
                    shape_meta=shape_meta,
                    init_state=None,
                    render_obs_key=cfg.online_task.env_runner.render_obs_key
                ),
                n_obs_steps=cfg.n_obs_steps,
                n_action_steps=cfg.n_action_steps,
                max_episode_steps=cfg.online_task.env_runner.max_steps
            )
        env_fns = [env_fn] * cfg.training.n_envs
        envs = AsyncVectorEnv(env_fns, dummy_env_fn=dummy_env_fn)

        ### uncomment to set all train env as in-demo env
        # env_init_fn_dills = list()
        # with h5py.File(dataset_path, 'r') as f:
        #     for i in range(cfg.training.n_envs):
        #         init_state = f[f'data/demo_{i}/states'][0]

        #         def init_fn(env, init_state=init_state):
        #             assert isinstance(env.env, RobomimicImageWrapper)
        #             env.env.init_state = init_state
        #         env_init_fn_dills.append(dill.dumps(init_fn))
        # envs.call_each('run_dill_function', 
        #     args_list=[(x,) for x in env_init_fn_dills])

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
        ## extract rgb_emb from demo_obs_emb
        lowdim_dim = sum([self.base_policy.obs_encoder.key_shape_map[k][0] for k in self.base_policy.obs_encoder.low_dim_keys])
        rgb_emb_dim = obs_emb_dim - lowdim_dim
        demo_rgb_emb = demo_emb['obs_emb'][..., -obs_emb_dim:-obs_emb_dim + rgb_emb_dim].to(device)  # (N, di)

        policy_opt = self.res_policy.get_optimizer(policy_lr=cfg.training.policy_lr)

        # replay buffer
        dummy_obs_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_emb_dim,), dtype=np.float32
        )
        dummy_buf_action_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(act_seq_dim * 2,), dtype=np.float32
        )  # only store res_naction & base_naction in on-policy
        dummy_action_space = gym.spaces.Box(
            low=-1.0, high=1.0,
            shape=(act_seq_dim,), dtype=np.float32
        )
        n_steps_per_rollout = int(cfg.training.training_freq // cfg.training.n_envs)
        rb = RolloutBuffer(
            n_steps_per_rollout,
            dummy_obs_space,
            dummy_buf_action_space,
            device=device,
            n_envs=cfg.training.n_envs,
            gamma=cfg.res_policy.gamma,
            gae_lambda=cfg.res_policy.gae_lambda,
        )

        if cfg.training.debug:
            cfg.training.num_steps = 5000
            cfg.training.log_every = 1000
            cfg.training.checkpoint_every = 5000
            cfg.training.eval_every = 5000

        # simplify necessary cfg
        training_freq = cfg.training.training_freq
        log_every = cfg.training.log_every
        eval_every = cfg.training.eval_every
        checkpoint_every = cfg.training.checkpoint_every
        n_steps = cfg.training.num_steps
        n_envs = cfg.online_task.n_envs
        n_epochs = cfg.training.n_epochs
        bs = cfg.training.batch_size
        n_updates_per_training = int(training_freq * n_epochs // bs)
        res_scale = cfg.training.res_scale

        ## check parameters for code clarity
        assert (
            log_every % training_freq == 0 and
            eval_every % training_freq == 0 and
            checkpoint_every % training_freq == 0 and
            eval_every % log_every == 0
        ), f"log_every({log_every}), eval_every({eval_every}), checkpoint_every({checkpoint_every}) must be divisible by training_freq({training_freq}) for code clarity."

        # action preprocess: from action to env action
        rot_tf = None
        if cfg.online_task.abs_action:
            rot_tf = RotationTransformer('axis_angle', 'rotation_6d')

        def undo_transform_action(action):
            if rot_tf is None:
                return action

            # undo rotation transformation
            raw_shape = action.shape
            if raw_shape[-1] == 20:  # dual arm
                action = action.reshape((-1, 2, 10))
            
            d_rot = action.shape[-1] - 4
            pos = action[..., :3]
            rot = action[..., 3:3+d_rot]
            gripper = action[..., [-1]]
            rot = rot_tf.inverse(rot)
            uaction = np.concatenate([pos, rot, gripper], axis=-1)
            if raw_shape[-1] == 20:  # dual arm
                uaction = uaction.reshape((*raw_shape[:-1], 14))
            return uaction

        learning_started = False

        # training loop
        obs_seq = envs.reset()
        last_episode_start = np.ones((n_envs,), dtype=bool)
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as logger:
            # Initial evaluation
            sum_policy.eval()
            eval_log = eval_env_runner.run(sum_policy)
            sum_policy.train()
            logger.log(eval_log)
            wandb_run.log(eval_log, step=self.global_step)

            while self.global_step < n_steps:
                # 1. collect samples
                rb.reset()
                for _ in tqdm.tqdm(
                        range(n_steps_per_rollout), 
                        desc=f"{self.global_step} / {n_steps} samples collected",
                        leave=False):
                    self.global_step += n_envs

                    # 1.1 Get exploration action
                    with torch.no_grad():
                        # 1.1.1 Get base prediction
                        obs_seq_tensor = dict_apply(
                            obs_seq, lambda x: torch.from_numpy(x).to(device=device)
                        )
                        base_dict = self.base_policy.predict_action(obs_seq_tensor)
                        obs_emb_tensor = base_dict['obs_emb'][:, -obs_emb_dim:].detach()
                        base_naction_tensor = base_dict['naction'].detach()
                        base_naction_flat = base_naction_tensor.flatten(start_dim=1).cpu().numpy()  # (B, Ta*da)

                        # 1.1.2 Get res prediction
                        res_input = obs_emb_tensor
                        if self.res_policy.actor_input == 'obs_action':
                            res_input = torch.cat(
                                [obs_emb_tensor, base_naction_tensor.flatten(start_dim=1)], dim=-1)
                        res_naction_tensor, log_prob, value = self.res_policy.predict_all(res_input)
                        res_naction_flat = res_naction_tensor.flatten(start_dim=1).cpu().numpy()

                        # 1.1.3 Combine two policies
                        sum_naction_tensor = res_scale * res_naction_tensor.reshape_as(base_naction_tensor) + base_naction_tensor
                        action = self.base_policy.normalizer['action'].unnormalize(sum_naction_tensor).cpu().numpy()

                    # 1.2 env_action and step
                    env_action = undo_transform_action(action)
                    next_obs_seq, rewards, dones, infos = envs.step(env_action)

                    ## reward preprocess
                    ## 1. fixed epi len performs better with positive rewards
                    # rewards -= 1.0

                    # 1.3 prepare transitions for rb
                    actions_to_save = np.concatenate(
                        [
                            res_naction_flat,
                            base_naction_flat,
                        ],
                        axis=-1
                    )
                    rb.add(
                        obs=obs_emb_tensor.cpu().numpy(),
                        action=actions_to_save,
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
                    obs_seq_tensor = dict_apply(
                        obs_seq, lambda x: torch.from_numpy(x).to(device=device))
                    base_dict = self.base_policy.predict_action(obs_seq_tensor)
                    obs_emb_tensor = base_dict['obs_emb'][:, -obs_emb_dim:].detach()
                    base_naction_tensor = base_dict['naction'].detach()
                    
                    policy_input = obs_emb_tensor
                    if self.res_policy.actor_input == 'obs_action':
                        policy_input = torch.cat([obs_emb_tensor, base_naction_tensor.flatten(start_dim=1)], dim=-1)
                    next_value = self.res_policy.predict_value(policy_input)
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

                ## compute d2d of this batch
                with torch.no_grad():
                    batch_rgb = batch.observations[..., :rgb_emb_dim]
                    batch_d2d = torch.cdist(
                        batch_rgb, demo_rgb_emb
                    ).min(dim=1).values # (B,)

                # 4. training metrics
                explained_var = explained_variance(rb.values.flatten(), rb.returns.flatten())
                step_log = {
                    'info/global_step': self.global_step,
                    'info/global_update': self.global_update,

                    'info/value': info['value'],
                    'info/actor_entropy': info['actor_entropy'],
                    'info/approx_kl': info['approx_kl'],
                    'info/clip_frac': info['clip_frac'],
                    'info/uncertainty': batch_d2d.mean().item(),
                    'info/res_naction_norm': info['res_naction_norm'],
                    'info/explained_var': explained_var,

                    'loss/critic_loss': info['critic_loss'],
                    'loss/actor_loss': info['actor_loss'],
                    'loss/actor_grad_norm': actor_grad_norm.item(),
                }

                # 5. evaluation
                if self.global_step % eval_every == 0:
                    sum_policy.eval()
                    eval_log = eval_env_runner.run(sum_policy)
                    step_log.update(eval_log)
                    sum_policy.train()

                # 6. logging
                logger.log(step_log)
                if self.global_step % log_every == 0:
                    wandb_run.log(step_log, step=self.global_step)

                # 7. checkpointing
                if self.global_step % checkpoint_every == 0:
                    path = pathlib.Path(self.output_dir) / 'checkpoints' / f'step={self.global_step}.ckpt'
                    path.parent.mkdir(parents=False, exist_ok=True)
                    payload = {
                        'cfg': self.cfg,
                        'res_policy': self.res_policy.state_dict(),
                        'optimizer': policy_opt.state_dict()
                    }
                    torch.save(payload, path.open('wb'), pickle_module=dill)

        # envs.close()


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainOnPolicyRobomimicWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
