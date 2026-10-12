"""Model- and layer-specific activation scale calibration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray


class CalibrationMethod(str, Enum):
    PROJECTION_SD = "projection_sd"
    RESIDUAL_NORM = "residual_norm"


@dataclass(frozen=True)
class Calibration:
    method: CalibrationMethod
    scales: Mapping[int, float]
    sample_counts: Mapping[int, int]


def calibrate_layers(
    activations: Mapping[int, NDArray[np.floating]],
    *,
    method: CalibrationMethod | str,
    vectors: Mapping[int, NDArray[np.floating]] | None = None,
) -> Calibration:
    """Compute projection-SD (primary) or mean residual-norm scales."""

    method = CalibrationMethod(method)
    if not activations:
        raise ValueError("at least one activation layer is required")
    scales: dict[int, float] = {}
    counts: dict[int, int] = {}
    for layer, raw_values in sorted(activations.items()):
        values = np.asarray(raw_values, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] < 1:
            raise ValueError(f"layer {layer} activations must have shape [examples, hidden]")
        if method is CalibrationMethod.PROJECTION_SD:
            if vectors is None or layer not in vectors:
                raise ValueError(f"projection-SD calibration needs a vector at layer {layer}")
            vector = np.asarray(vectors[layer], dtype=np.float64)
            norm = float(np.linalg.norm(vector))
            if norm <= 1e-12:
                raise ValueError(f"calibration vector is degenerate at layer {layer}")
            projections = values @ (vector / norm)
            if len(projections) < 2:
                raise ValueError("projection-SD calibration requires at least two examples")
            scale = float(projections.std(ddof=1))
        else:
            scale = float(np.linalg.norm(values, axis=1).mean())
        if not np.isfinite(scale) or scale <= 1e-12:
            raise ValueError(f"calibration is degenerate at layer {layer}: {scale}")
        scales[layer] = scale
        counts[layer] = values.shape[0]
    return Calibration(method=method, scales=scales, sample_counts=counts)
