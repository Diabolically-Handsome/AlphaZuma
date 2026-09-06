"""Inference-only compatibility for action-head-only SB3 checkpoints.

The action-head-only training routes intentionally optimized only
``action_net.weight`` and ``action_net.bias``.  Their archived PPO zip files
therefore contain an optimizer state whose parameter group is smaller than the
fresh full-policy optimizer constructed by ``MaskablePPO.load``.  Optimizer
state is irrelevant for inference, but SB3 tries to restore it before returning
the model and otherwise rejects the checkpoint.

This startup hook preserves strict loading for every module and tensor while
ignoring only the specific optimizer parameter-group-size mismatch.  Any other
optimizer-load error is re-raised unchanged.  The original checkpoint bytes
are never modified.
"""

from __future__ import annotations

import sys

import torch


_ORIGINAL_LOAD_STATE_DICT = torch.optim.Optimizer.load_state_dict


def _inference_load_state_dict(self, state_dict):
    try:
        return _ORIGINAL_LOAD_STATE_DICT(self, state_dict)
    except ValueError as error:
        if str(error) != (
            "loaded state dict contains a parameter group that doesn't match "
            "the size of optimizer's group"
        ):
            raise
        print(
            "INFERENCE_OPTIMIZER_COMPAT: ignored archived optimizer "
            "parameter-group-size mismatch; policy tensors remain strict",
            file=sys.stderr,
            flush=True,
        )
        return None


torch.optim.Optimizer.load_state_dict = _inference_load_state_dict
