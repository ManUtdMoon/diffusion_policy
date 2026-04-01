from typing import Dict
import torch
import torch.nn as nn
from zprl.model.common.module_attr_mixin import ModuleAttrMixin
from zprl.model.common.normalizer import LinearNormalizer

class BaseImagePolicy(ModuleAttrMixin):
    # init accepts keyword argument shape_meta, see config/task/*_image.yaml

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict:
            str: B,To,*
        return: B,Ta,Da
        """
        raise NotImplementedError()

    def encode_frame(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Encode single-step observations into frame embeddings.

        Args:
            obs_dict: dict of tensors with shape (B, *)
        Returns:
            frame_emb: (B, do) per-timestep observation embedding
        """
        raise NotImplementedError()

    # reset state for stateful policies
    def reset(self):
        pass

    # ========== training ===========
    # no standard training interface except setting normalizer
    def set_normalizer(self, normalizer: LinearNormalizer):
        raise NotImplementedError()
