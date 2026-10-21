from abc import abstractmethod

from stable_baselines3.common.buffers import ReplayBuffer


class HistoryReplayBufferBase(ReplayBuffer):
    """SB3 storage for history buffers that implement their own n-step sampling.

    SB3 2.4.1 has no NStepReplayBuffer. Only storage and the n_steps/gamma
    attributes are shared here; subclasses must supply the complete sampler.
    The legacy ReactiveNStepReplayBuffer still uses upstream SB3 unchanged.
    """

    def __init__(self, *args, n_steps: int = 3, gamma: float = 0.99, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_steps = n_steps
        self.gamma = gamma
        if self.optimize_memory_usage:
            raise NotImplementedError('History replay does not support optimize_memory_usage=True')

    @abstractmethod
    def _get_samples(self, batch_inds, env=None):
        raise NotImplementedError
