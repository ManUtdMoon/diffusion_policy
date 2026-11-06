import torch


U = tuple(j / 100 for j in range(5, 100, 5))


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


def action_mse_quantiles(pred, gt):
    """Return [B, 19] MSE quantiles for [B, K, D] predictions and [B, D] targets."""
    if pred.ndim != 3 or gt.shape != (pred.shape[0], pred.shape[2]):
        raise ValueError("Expected pred [B, K, D] and gt [B, D] with matching B and D")
    error = (pred.float() - gt.float()[:, None, :]).square().mean(dim=-1)
    levels = error.new_tensor(U)
    return torch.quantile(error, levels, dim=1, interpolation="linear").transpose(0, 1)
