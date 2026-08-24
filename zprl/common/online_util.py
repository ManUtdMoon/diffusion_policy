from omegaconf import OmegaConf


def prepare_base_policy_config(base_cfg, n_action_steps, num_inference_steps):
    """Patch a legacy checkpoint config and apply online inference overrides."""
    base_cfg_yaml = OmegaConf.to_yaml(base_cfg)
    if 'diffusion_policy' in base_cfg_yaml:
        base_cfg = OmegaConf.create(
            base_cfg_yaml.replace('diffusion_policy', 'zprl'))

    base_cfg.policy.n_action_steps = n_action_steps
    base_cfg.policy.num_inference_steps = num_inference_steps
    return base_cfg


def get_crop_randomizers(policy):
    """Find crop modules by capability so both V2 and V3 are supported."""
    return [
        module for module in policy.modules()
        if hasattr(module, 'force_random_crop')
    ]
