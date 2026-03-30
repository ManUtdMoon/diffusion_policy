import math
from torch.optim.lr_scheduler import LambdaLR
from diffusers.optimization import (
    Union, SchedulerType, Optional,
    Optimizer, TYPE_TO_SCHEDULER_FUNCTION
)


def get_cosine_with_min_lr_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.0,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
) -> LambdaLR:
    """Cosine schedule that decays to min_lr_ratio * initial_lr instead of 0."""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_value = max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_value

    return LambdaLR(optimizer, lr_lambda, last_epoch)


# Register custom scheduler type
_CUSTOM_SCHEDULERS = {
    'cosine_with_min_lr': get_cosine_with_min_lr_schedule_with_warmup,
}


def get_scheduler(
    name: Union[str, SchedulerType],
    optimizer: Optimizer,
    num_warmup_steps: Optional[int] = None,
    num_training_steps: Optional[int] = None,
    **kwargs
):
    """
    Added kwargs vs diffuser's original implementation

    Unified API to get any scheduler from its name.

    Args:
        name (`str` or `SchedulerType`):
            The name of the scheduler to use.
        optimizer (`torch.optim.Optimizer`):
            The optimizer that will be used during training.
        num_warmup_steps (`int`, *optional*):
            The number of warmup steps to do. This is not required by all schedulers (hence the argument being
            optional), the function will raise an error if it's unset and the scheduler type requires it.
        num_training_steps (`int``, *optional*):
            The number of training steps to do. This is not required by all schedulers (hence the argument being
            optional), the function will raise an error if it's unset and the scheduler type requires it.
    """
    # Check custom schedulers first
    if name in _CUSTOM_SCHEDULERS:
        if num_warmup_steps is None:
            raise ValueError(f"{name} requires `num_warmup_steps`, please provide that argument.")
        if num_training_steps is None:
            raise ValueError(f"{name} requires `num_training_steps`, please provide that argument.")
        return _CUSTOM_SCHEDULERS[name](
            optimizer, num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps, **kwargs)

    name = SchedulerType(name)
    schedule_func = TYPE_TO_SCHEDULER_FUNCTION[name]
    if name == SchedulerType.CONSTANT:
        return schedule_func(optimizer, **kwargs)

    # All other schedulers require `num_warmup_steps`
    if num_warmup_steps is None:
        raise ValueError(f"{name} requires `num_warmup_steps`, please provide that argument.")

    if name == SchedulerType.CONSTANT_WITH_WARMUP:
        return schedule_func(optimizer, num_warmup_steps=num_warmup_steps, **kwargs)

    # All other schedulers require `num_training_steps`
    if num_training_steps is None:
        raise ValueError(f"{name} requires `num_training_steps`, please provide that argument.")

    return schedule_func(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps, **kwargs)
