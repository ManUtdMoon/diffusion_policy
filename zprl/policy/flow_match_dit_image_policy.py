from typing import Dict
import torch
import torch.nn.functional as F
from zprl.model.common.normalizer import LinearNormalizer
from zprl.policy.base_image_policy import BaseImagePolicy
from zprl.model.diffusion.dit import DiTNoiseNet
from zprl.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from zprl.common.pytorch_util import dict_apply
from zprl.policy.prev_action_util import append_prev_action_cond, get_prev_action_cond, split_prev_action

class FlowMatchDitImagePolicy(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
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
            n_prev_action_steps=0):
        super().__init__()

        # parse shapes
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        # get feature dim
        obs_feature_dim = obs_encoder.output_shape()[0]
        prev_action_cond_dim = int(n_prev_action_steps) * action_dim

        # create diffusion model
        input_dim = action_dim
        cond_dim = obs_feature_dim * n_obs_steps + prev_action_cond_dim

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
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.n_prev_action_steps = n_prev_action_steps
        assert num_inference_steps is not None, "num_inference_steps must be specified for FlowMatchDitImagePolicy"
        self.num_inference_steps = num_inference_steps

    

    # ========= inference  ============
    def conditional_sample(self, 
            condition_data,
            global_cond=None,
            generator=None
            ):
        model = self.model

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
    
        timesteps = torch.linspace(
            1.0, 0.0, self.num_inference_steps + 1,
            device=trajectory.device,
            dtype=torch.float32)
        dt = -1.0 / self.num_inference_steps

        for t in timesteps[:-1]:
            # 1. predict model output
            model_output = model(trajectory, t, global_cond)

            # 2. compute previous image: x_t -> x_t-1
            trajectory = trajectory + dt * model_output

        return trajectory
    
    def encode_obs(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        obs_dict, prev_action, prev_action_valid_mask = split_prev_action(obs_dict)
        prev_action_cond = get_prev_action_cond(prev_action, prev_action_valid_mask, self.n_prev_action_steps, self.normalizer['action'])
        nobs = self.normalizer.normalize(obs_dict)
        B, To = next(iter(nobs.values())).shape[:2]
        batched_nobs = dict_apply(
            nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
        obs_emb = self.obs_encoder(batched_nobs) # (B*To, do)
        obs_emb = obs_emb.reshape(B, -1) # (B, Do=To*do)
        obs_emb = append_prev_action_cond(obs_emb, prev_action_cond)
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
            global_cond=global_cond)
        
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

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        # normalize input
        obs_dict, prev_action, prev_action_valid_mask = split_prev_action(obs_dict)
        prev_action_cond = get_prev_action_cond(prev_action, prev_action_valid_mask, self.n_prev_action_steps, self.normalizer['action'])
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
        global_cond = append_prev_action_cond(global_cond, prev_action_cond)
        # empty data for action
        cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)

        # run sampling
        nsample = self.conditional_sample(
            cond_data, 
            global_cond=global_cond)
        
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

    def forward(self, batch):
        return self.compute_loss(batch)

    def compute_loss(self, batch):
        # normalize input
        obs_dict, prev_action, prev_action_valid_mask = split_prev_action(batch['obs'])
        prev_action_cond = get_prev_action_cond(prev_action, prev_action_valid_mask, self.n_prev_action_steps, self.normalizer['action'])
        nobs = self.normalizer.normalize(obs_dict)
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
        global_cond = append_prev_action_cond(global_cond, prev_action_cond)

        # Sample noise that we'll add to the images
        noise = torch.randn_like(trajectory)
        # Sample a random continuous flow time. t=1 is noise, t=0 is sample.
        timesteps = torch.rand(
            (batch_size,), device=trajectory.device, dtype=trajectory.dtype)
        timesteps_b = timesteps[:, None, None]
        noisy_trajectory = (1 - timesteps_b) * trajectory + timesteps_b * noise

        # Predict the velocity field: from sample to noise
        pred = self.model(noisy_trajectory, timesteps, global_cond)

        # target u = x1 - x0
        target = noise - trajectory

        loss = F.mse_loss(pred, target)
        return loss
