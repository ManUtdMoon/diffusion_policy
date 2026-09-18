import numpy as np

from zprl.env.robomimic.robomimic_subtask_wrapper import SubtaskWrapper


class ToolHangSubtaskWrapper(SubtaskWrapper):
    def __init__(self, env, subtask_config, gamma=None):
        super().__init__(env, subtask_config, gamma=gamma)
        assert self.stages == ('frame_assembled',)
        self.task = env
        while not hasattr(self.task, '_check_frame_assembled'):
            self.task = self.task.env

    def _get_stage_predicates(self):
        return np.asarray([self.task._check_frame_assembled()], dtype=np.bool_)
