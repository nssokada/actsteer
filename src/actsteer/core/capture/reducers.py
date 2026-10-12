"""Attention-mask-aware reductions for residual activations."""

from __future__ import annotations

from typing import Any


def hidden_from_output(output: Any) -> Any:
    """Extract a block's residual tensor without assuming tuple-only outputs."""

    if isinstance(output, (tuple, list)):
        if not output:
            raise ValueError("transformer block returned an empty sequence")
        return output[0]
    return output


def replace_hidden(output: Any, hidden: Any) -> Any:
    """Replace the residual tensor while preserving the block output container."""

    if isinstance(output, tuple):
        return (hidden, *output[1:])
    if isinstance(output, list):
        return [hidden, *output[1:]]
    return hidden


def masked_mean(hidden: Any, mask: Any) -> Any:
    """Reduce ``[batch, tokens, hidden]`` over selected token positions."""

    if hidden.ndim != 3:
        raise ValueError(f"expected hidden state [B,T,H], got shape {tuple(hidden.shape)}")
    if mask.ndim == 3 and mask.shape[-1] == 1:
        mask = mask[..., 0]
    if mask.ndim != 2 or tuple(mask.shape) != tuple(hidden.shape[:2]):
        raise ValueError(
            f"token mask shape {tuple(mask.shape)} does not match hidden state "
            f"{tuple(hidden.shape[:2])}"
        )
    weights = mask.to(device=hidden.device, dtype=hidden.dtype)
    counts = weights.sum(dim=1, keepdim=True)
    if bool((counts == 0).any().item()):
        raise ValueError("every example must select at least one token")
    return (hidden * weights.unsqueeze(-1)).sum(dim=1) / counts
