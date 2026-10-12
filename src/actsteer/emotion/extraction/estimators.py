"""Emotion-vector estimators over reduced per-example activations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.floating]


class ExtractionPreset(str, Enum):
    CONTRASTIVE_MEAN = "contrastive_mean"
    SOFRONIEW_GLOBAL = "sofroniew_global"
    SOFRONIEW_NEUTRAL = "sofroniew_neutral"


@dataclass(frozen=True)
class ReducedActivations:
    """Aligned ``[examples, hidden]`` activation matrices for several layers."""

    example_ids: tuple[str, ...]
    concepts: tuple[str, ...]
    layers: Mapping[int, FloatArray]

    def __post_init__(self) -> None:
        count = len(self.example_ids)
        if count == 0 or len(self.concepts) != count:
            raise ValueError("example_ids and concepts must be aligned and non-empty")
        if len(set(self.example_ids)) != count:
            raise ValueError("activation example IDs must be unique")
        if not self.layers:
            raise ValueError("at least one activation layer is required")
        hidden_size: int | None = None
        for layer, values in self.layers.items():
            array = np.asarray(values)
            if array.ndim != 2 or array.shape[0] != count:
                raise ValueError(
                    f"layer {layer} must have shape [{count}, hidden], got {array.shape}"
                )
            if not np.isfinite(array).all():
                raise ValueError(f"layer {layer} contains non-finite activations")
            hidden_size = hidden_size or array.shape[1]
            if array.shape[1] != hidden_size:
                raise ValueError("all activation layers must have the same hidden size")


@dataclass(frozen=True)
class VectorExtraction:
    preset: ExtractionPreset
    vectors: Mapping[int, Mapping[str, FloatArray]]
    centers: Mapping[int, FloatArray]
    neutral_pcs: Mapping[int, FloatArray]
    neutral_concept: str
    pc_variance_threshold: float


def _neutral_pcs(values: FloatArray, variance_threshold: float) -> FloatArray:
    if not 0.0 <= variance_threshold <= 1.0:
        raise ValueError("PC variance threshold must lie in [0, 1]")
    hidden = values.shape[1]
    if variance_threshold == 0.0 or values.shape[0] < 2:
        return np.empty((0, hidden), dtype=np.float64)
    centered = values.astype(np.float64) - values.mean(axis=0, keepdims=True)
    _, singular_values, right = np.linalg.svd(centered, full_matrices=False)
    variance = singular_values**2
    total = float(variance.sum())
    if total <= 0.0:
        return np.empty((0, hidden), dtype=np.float64)
    cumulative = np.cumsum(variance) / total
    count = int(np.searchsorted(cumulative, variance_threshold, side="left")) + 1
    return right[:count]


def _unit(vector: FloatArray, *, concept: str, layer: int) -> FloatArray:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError(f"cannot normalize degenerate vector for {concept!r} at layer {layer}")
    return (vector / norm).astype(np.float32)


def extract_emotion_vectors(
    captures: ReducedActivations,
    *,
    preset: ExtractionPreset | str,
    neutral_concept: str = "neutral",
    concepts: Sequence[str] | None = None,
    pc_variance_threshold: float = 0.5,
) -> VectorExtraction:
    """Extract unit emotion directions using one of the three v0.1 presets."""

    preset = ExtractionPreset(preset)
    labels = np.asarray(captures.concepts, dtype=object)
    available = set(captures.concepts)
    if neutral_concept not in available:
        raise ValueError(f"neutral concept {neutral_concept!r} is absent from captures")
    selected = tuple(concepts) if concepts is not None else tuple(
        sorted(available - {neutral_concept})
    )
    if not selected:
        raise ValueError("at least one non-neutral concept is required")
    missing = set(selected) - available
    if missing:
        raise ValueError(f"requested concepts are absent from captures: {sorted(missing)}")
    if neutral_concept in selected:
        raise ValueError("neutral concept cannot also be an extraction target")

    vectors: dict[int, dict[str, FloatArray]] = {}
    centers: dict[int, FloatArray] = {}
    neutral_pcs: dict[int, FloatArray] = {}
    for layer, raw_values in sorted(captures.layers.items()):
        values = np.asarray(raw_values, dtype=np.float64)
        neutral_values = values[labels == neutral_concept]
        neutral_mean = neutral_values.mean(axis=0)
        means = {concept: values[labels == concept].mean(axis=0) for concept in selected}

        if preset is ExtractionPreset.SOFRONIEW_GLOBAL:
            center = np.stack(tuple(means.values())).mean(axis=0)
        else:
            center = neutral_mean
        if preset is ExtractionPreset.CONTRASTIVE_MEAN:
            pcs = np.empty((0, values.shape[1]), dtype=np.float64)
        else:
            pcs = _neutral_pcs(neutral_values, pc_variance_threshold)

        layer_vectors: dict[str, FloatArray] = {}
        for concept, mean in means.items():
            direction = mean - center
            if pcs.shape[0]:
                direction = direction - pcs.T @ (pcs @ direction)
            layer_vectors[concept] = _unit(direction, concept=concept, layer=layer)
        vectors[layer] = layer_vectors
        centers[layer] = center.astype(np.float32)
        neutral_pcs[layer] = pcs.astype(np.float32)

    return VectorExtraction(
        preset=preset,
        vectors=vectors,
        centers=centers,
        neutral_pcs=neutral_pcs,
        neutral_concept=neutral_concept,
        pc_variance_threshold=(
            0.0 if preset is ExtractionPreset.CONTRASTIVE_MEAN else pc_variance_threshold
        ),
    )
