import math


def configure_training_budget(training, loader_length):
    """Return the total batch budget and derive epochs from the prepared loader."""
    if loader_length <= 0:
        raise ValueError('Training dataloader must not be empty')
    max_steps = training.max_train_steps
    if max_steps is None:
        if training.num_epochs is None or training.num_epochs <= 0:
            raise ValueError('num_epochs must be positive when max_train_steps is null')
        return training.num_epochs * loader_length
    if max_steps <= 0:
        raise ValueError('max_train_steps must be positive or null')
    training.num_epochs = math.ceil(max_steps / loader_length)
    return max_steps
