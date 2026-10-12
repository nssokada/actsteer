"""Model-independent layer selectors resolved before execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class LayerSelector(Protocol):
    def resolve(self, num_layers: int) -> tuple[int, ...]: ...


def _require_num_layers(num_layers: int) -> None:
    if num_layers <= 0:
        raise ValueError(f"num_layers must be positive, got {num_layers}")


@dataclass(frozen=True)
class AbsoluteLayers:
    indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.indices:
            raise ValueError("at least one absolute layer is required")
        if any(index < 0 for index in self.indices):
            raise ValueError(f"layer indices must be non-negative: {self.indices}")
        if len(set(self.indices)) != len(self.indices):
            raise ValueError(f"layer indices must be unique: {self.indices}")

    def resolve(self, num_layers: int) -> tuple[int, ...]:
        _require_num_layers(num_layers)
        invalid = [index for index in self.indices if index >= num_layers]
        if invalid:
            raise ValueError(
                f"layers {invalid} out of range for model with {num_layers} layers"
            )
        return self.indices


@dataclass(frozen=True)
class RelativeLayers:
    """Select depths on [0, 1], rounded onto [0, num_layers - 1]."""

    depths: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.depths:
            raise ValueError("at least one relative depth is required")
        if any(depth < 0.0 or depth > 1.0 for depth in self.depths):
            raise ValueError(f"relative depths must lie in [0, 1]: {self.depths}")

    def resolve(self, num_layers: int) -> tuple[int, ...]:
        _require_num_layers(num_layers)
        last = num_layers - 1
        resolved = tuple(dict.fromkeys(round(depth * last) for depth in self.depths))
        return resolved


@dataclass(frozen=True)
class RelativeWindow:
    """Select every layer whose normalized depth lies in an inclusive window."""

    start: float
    stop: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.start <= self.stop <= 1.0):
            raise ValueError(
                f"relative layer window must satisfy 0 <= start <= stop <= 1; "
                f"got ({self.start}, {self.stop})"
            )

    def resolve(self, num_layers: int) -> tuple[int, ...]:
        _require_num_layers(num_layers)
        if num_layers == 1:
            return (0,)
        last = num_layers - 1
        resolved = tuple(
            index
            for index in range(num_layers)
            if self.start <= index / last <= self.stop
        )
        if not resolved:
            center = (self.start + self.stop) / 2
            return (round(center * last),)
        return resolved

