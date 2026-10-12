"""Semantic model locations independent of concrete module paths."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ActivationStream(str, Enum):
    RESIDUAL = "residual"


class ActivationPoint(str, Enum):
    PRE = "pre"
    POST = "post"


@dataclass(frozen=True)
class ActivationSite:
    stream: ActivationStream = ActivationStream.RESIDUAL
    point: ActivationPoint = ActivationPoint.POST

