from __future__ import annotations

import math
import unittest

import numpy as np

from actsteer.core.interventions import StrengthPolicy, allocate_amplitudes
from actsteer.emotion import (
    CalibrationMethod,
    ExtractionPreset,
    ReducedActivations,
    calibrate_layers,
    extract_emotion_vectors,
)


class ExtractionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.values = np.asarray(
            [
                [2.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 4.0, 0.0],
                [0.0, 0.0, -1.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        self.captures = ReducedActivations(
            example_ids=tuple(f"example-{index}" for index in range(6)),
            concepts=("fear", "fear", "happy", "happy", "neutral", "neutral"),
            layers={2: self.values},
        )

    def test_contrastive_mean_is_direct_mean_difference(self) -> None:
        result = extract_emotion_vectors(
            self.captures,
            preset=ExtractionPreset.CONTRASTIVE_MEAN,
        )
        np.testing.assert_allclose(result.vectors[2]["fear"], [1.0, 0.0, 0.0])
        self.assertEqual(result.neutral_pcs[2].shape, (0, 3))

    def test_sofroniew_removes_neutral_principal_component(self) -> None:
        result = extract_emotion_vectors(
            self.captures,
            preset=ExtractionPreset.SOFRONIEW_NEUTRAL,
            pc_variance_threshold=0.5,
        )
        vector = result.vectors[2]["fear"]
        pc = result.neutral_pcs[2][0]
        self.assertAlmostEqual(float(vector @ pc), 0.0, places=6)
        self.assertAlmostEqual(float(np.linalg.norm(vector)), 1.0, places=6)

    def test_projection_sd_calibration(self) -> None:
        calibration = calibrate_layers(
            {2: self.values},
            method=CalibrationMethod.PROJECTION_SD,
            vectors={2: np.asarray([1.0, 0.0, 0.0])},
        )
        self.assertAlmostEqual(calibration.scales[2], self.values[:, 0].std(ddof=1))

    def test_total_budget_does_not_grow_with_layer_count(self) -> None:
        scales = {2: 2.0, 4: 4.0}
        amplitudes = allocate_amplitudes(3.0, scales)
        standardized_norm = math.sqrt(
            sum((amplitudes[layer] / scales[layer]) ** 2 for layer in scales)
        )
        self.assertAlmostEqual(standardized_norm, 3.0)
        legacy = allocate_amplitudes(3.0, scales, policy=StrengthPolicy.PER_LAYER)
        self.assertEqual(legacy, {2: 6.0, 4: 12.0})

    def test_missing_neutral_fails(self) -> None:
        captures = ReducedActivations(
            example_ids=("a", "b"),
            concepts=("fear", "fear"),
            layers={0: np.asarray([[1.0, 0.0], [2.0, 0.0]])},
        )
        with self.assertRaisesRegex(ValueError, "neutral concept"):
            extract_emotion_vectors(captures, preset="contrastive_mean")


if __name__ == "__main__":
    unittest.main()
