import numpy as np

from zprl.env.robomimic.robomimic_subtask_wrapper import SubtaskWrapper, get_subtask_dim


class SquareSubtaskWrapper(SubtaskWrapper):
    def __init__(self, env, subtask_config, gamma=None):
        super().__init__(env, subtask_config, gamma=gamma)
        self.hover_threshold = float(subtask_config.hover_threshold)
        assert self.stages in (('grasp',), ('grasp', 'hover'))

        self.task = env
        while not hasattr(self.task, 'staged_rewards'):
            self.task = self.task.env
        assert self.task.single_object_mode == 2
        assert self.task.nut_id == self.task.nut_to_id['square']

    def _get_stage_predicates(self):
        _, r_grasp, _, r_hover = self.task.staged_rewards()
        predicate_map = {
            'grasp': r_grasp > 0.0,
            'hover': r_hover >= self.hover_threshold,
        }
        return np.asarray([predicate_map[stage] for stage in self.stages], dtype=np.bool_)
