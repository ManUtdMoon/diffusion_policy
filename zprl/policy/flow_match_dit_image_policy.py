from typing import Dict
import torch
import torch.nn.functional as F
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler

from zprl.model.common.normalizer import LinearNormalizer
from zprl.policy.base_image_policy import BaseImagePolicy
from zprl.model.diffusion.dit import DiTNoiseNet
from zprl.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from zprl.common.pytorch_util import dict_apply

class FlowMatchDitImagePolicy(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: FlowMatchEulerDiscreteScheduler,
            obs_encoder: MultiImageObsEncoder,
            horizon, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            # model parameters
            num_blocks=6,
            hidden_dim=384,
            time_dim=128,
            dropout=0.1,
            dim_feedforward=2048,
            nhead=8,
            activation="gelu",
            # parameters passed to step
            **kwargs):
        super().__init__()

        # parse shapes
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        # get feature dim
        obs_feature_dim = obs_encoder.output_shape()[0]

        # create diffusion model
        input_dim = action_dim
        cond_dim = obs_feature_dim * n_obs_steps

        model = DiTNoiseNet(
            input_dim=input_dim,
            input_length=horizon,
            cond_dim=cond_dim,
            time_dim=time_dim,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            dropout=dropout,
            dim_feedforward=dim_feedforward,
            nhead=nhead,
            activation=activation
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.kwargs = kwargs

        assert num_inference_steps is not None, "num_inference_steps must be specified for FlowMatchDitImagePolicy"
        self.num_inference_steps = num_inference_steps

        self.timesteps = self.noise_scheduler.timesteps.clone()
    
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data,
            global_cond=None,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
    
        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. predict model output
            model_output = model(trajectory, t, global_cond)

            # 2. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample       

        return trajectory
    
    def encode_obs(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        nobs = self.normalizer.normalize(obs_dict)
        B, To = next(iter(nobs.values())).shape[:2]
        batched_nobs = dict_apply(
            nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
        obs_emb = self.obs_encoder(batched_nobs) # (B*To, do)
        obs_emb = obs_emb.reshape(B, -1) # (B, Do=To*do)
        return obs_emb
    
    def conditional_predict(self, obs_emb: torch.Tensor) -> Dict[str, torch.Tensor]:
        B = obs_emb.shape[0]
        T = self.horizon
        Da = self.action_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # condition through global feature
        global_cond = obs_emb
        # empty data for action
        cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)

        # run sampling
        nsample = self.conditional_sample(
            cond_data,
            global_cond=global_cond,
            **self.kwargs)
        
        # unnormalize prediction
        naction_pred = nsample
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred,
            'naction': naction_pred[:,start:end],
            'naction_pred': naction_pred,
            'obs_emb': global_cond,
        }
        return result

    def conditional_predict_from_noise(self, obs_emb: torch.Tensor, noise: torch.Tensor) -> Dict[str, torch.Tensor]:
        B = obs_emb.shape[0]
        T = self.horizon
        Da = self.action_dim
        To = self.n_obs_steps

        device = self.device
        dtype = self.dtype

        assert noise.shape == (B, T, Da)
        trajectory = noise.to(device=device, dtype=dtype)
        scheduler = self.noise_scheduler

        scheduler.set_timesteps(self.num_inference_steps)
        for t in scheduler.timesteps:
            model_output = self.model(trajectory, t, obs_emb)
            trajectory = scheduler.step(
                model_output, t, trajectory,
                **self.kwargs
                ).prev_sample

        naction_pred = trajectory
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]

        result = {
            'action': action,
            'action_pred': action_pred,
            'naction': naction_pred[:,start:end],
            'naction_pred': naction_pred,
            'obs_emb': obs_emb,
        }
        return result

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # condition through global feature
        this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs)
        # reshape back to B, Do
        global_cond = nobs_features.reshape(B, -1)
        # empty data for action
        cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)

        # run sampling
        nsample = self.conditional_sample(
            cond_data, 
            global_cond=global_cond,
            **self.kwargs)
        
        # unnormalize prediction
        naction_pred = nsample[..., :Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred,
            'naction': naction_pred[:,start:end],
            'naction_pred': naction_pred,
            'obs_emb': global_cond,
        }
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # normalize input
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        batch_size = nactions.shape[0]

        # handle different ways of passing observation
        global_cond = None
        trajectory = nactions
        # reshape B, T, ... to B*T
        this_nobs = dict_apply(nobs, 
            lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs)
        # reshape back to B, Do
        global_cond = nobs_features.reshape(batch_size, -1)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        if self.timesteps.device != trajectory.device:
            self.timesteps = self.timesteps.to(trajectory.device)
        timestep_idxs = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()
        timesteps = self.timesteps[timestep_idxs]
        # Add noise to the clean images by linear interpolation
        # (this is the forward flow process)
        noisy_trajectory = self.noise_scheduler.scale_noise(
            trajectory, timesteps, noise)

        # Predict the velocity field: from sample to noise
        pred = self.model(noisy_trajectory, timesteps, global_cond)

        # target u = x1 - x0
        target = noise - trajectory

        loss = F.mse_loss(pred, target)
        return loss
