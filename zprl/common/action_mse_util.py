import torch


def action_mse_per_sample(pred_action, gt_action, action_mse_groups):
    squared_error = (pred_action - gt_action).square()
    metrics = {
        "": squared_error.flatten(start_dim=1).mean(dim=1)
    }

    action_dim = squared_error.shape[-1]
    for name, ranges in action_mse_groups.items():
        group_errors = list()
        for start, end in ranges:
            end = action_dim if end is None else end
            if start < 0 or end > action_dim or start >= end:
                raise ValueError(
                    f"Invalid action MSE range [{start}, {end}) for action_dim={action_dim}"
                )
            group_errors.append(squared_error[..., start:end])
        metrics[name] = torch.cat(group_errors, dim=-1).flatten(start_dim=1).mean(dim=1)

    return metrics
