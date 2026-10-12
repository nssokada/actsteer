"""Explicit dose semantics for single- and multi-layer steering."""

from __future__ import annotations

import math
from collections.abc import Mapping
from enum import Enum
from typing import Any


class StrengthPolicy(str, Enum):
    TOTAL_BUDGET = "total_budget"
    PER_LAYER = "per_layer"


def allocate_amplitudes(
    strength: float,
    calibration_scales: Mapping[int, float],
    *,
    policy: StrengthPolicy = StrengthPolicy.TOTAL_BUDGET,
) -> dict[int, float]:
    """Translate a dimensionless strength into one activation-space dose/layer."""

    if not calibration_scales:
        raise ValueError("at least one calibrated layer is required")
    invalid = {layer: scale for layer, scale in calibration_scales.items() if scale <= 0}
    if invalid:
        raise ValueError(f"calibration scales must be positive: {invalid}")
    divisor = math.sqrt(len(calibration_scales)) if policy is StrengthPolicy.TOTAL_BUDGET else 1.0
    return {
        layer: float(strength) * float(scale) / divisor
        for layer, scale in calibration_scales.items()
    }


def build_torch_deltas(
    vectors: Mapping[int, Any],
    amplitudes: Mapping[int, float],
    *,
    device: Any,
    dtype: Any | None = None,
) -> dict[int, Any]:
    """Create normalized PyTorch deltas ready for ``SteeringController``."""

    try:
        import torch
    except ImportError as error:  # pragma: no cover - model environment only
        raise ImportError("steering deltas require the ActSteer model dependencies") from error
    if set(vectors) != set(amplitudes):
        raise ValueError("vectors and amplitudes must name exactly the same layers")
    deltas: dict[int, Any] = {}
    for layer, vector in vectors.items():
        tensor = torch.as_tensor(vector, device=device, dtype=torch.float32)
        if tensor.ndim != 1:
            raise ValueError(f"steering vector at layer {layer} must be one-dimensional")
        norm = tensor.norm()
        if not bool(torch.isfinite(norm).item()) or float(norm.item()) <= 1e-12:
            raise ValueError(f"steering vector is degenerate at layer {layer}")
        delta = float(amplitudes[layer]) * tensor / norm
        deltas[layer] = delta.to(dtype=dtype) if dtype is not None else delta
    return deltas
