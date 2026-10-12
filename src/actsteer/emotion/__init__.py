"""Emotion-specific datasets, extraction protocols, and calibration tools.

This package may depend on :mod:`actsteer.core`. Core modules must not import
from this package.
"""

__all__: list[str] = []
from actsteer.emotion.calibration import Calibration, CalibrationMethod, calibrate_layers
from actsteer.emotion.datasets import from_codrai_records, load_codrai_parquet
from actsteer.emotion.extraction import (
    ExtractionPreset,
    ReducedActivations,
    VectorExtraction,
    extract_emotion_vectors,
)

__all__ = [
    "Calibration",
    "CalibrationMethod",
    "ExtractionPreset",
    "ReducedActivations",
    "VectorExtraction",
    "calibrate_layers",
    "extract_emotion_vectors",
    "from_codrai_records",
    "load_codrai_parquet",
]
