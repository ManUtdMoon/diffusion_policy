import numpy as np

from zprl.env.robomimic.robomimic_subtask_wrapper import SubtaskWrapper


class TransportSubtaskWrapper(SubtaskWrapper):
    def __init__(self, env, subtask_config, gamma=None):
        super().__init__(env, subtask_config, gamma=gamma)
        assert self.stages == ('trash_in_trash_bin', 'payload_in_target_bin')
        self.task = env
        while not hasattr(self.task, 'transport'):
            self.task = self.task.env

    def _get_stage_predicates(self):
        return np.asarray(
            [getattr(self.task.transport, stage) for stage in self.stages],
            dtype=np.bool_)

    def _stage_bonus(self, completion_delta):
        # Task success already carries its own reward, whichever stage finishes last.
        if self.task._check_success():
            return 0.0
        return super()._stage_bonus(completion_delta)
