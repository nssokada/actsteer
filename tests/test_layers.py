from __future__ import annotations

import unittest

from actsteer.core.capture.layers import AbsoluteLayers, RelativeLayers, RelativeWindow


class LayerSelectorTests(unittest.TestCase):
    def test_absolute_layers_fail_out_of_range(self) -> None:
        with self.assertRaisesRegex(ValueError, "out of range"):
            AbsoluteLayers((0, 4)).resolve(num_layers=4)

    def test_relative_depths_resolve_against_model(self) -> None:
        self.assertEqual(RelativeLayers((0.0, 0.5, 1.0)).resolve(5), (0, 2, 4))

    def test_relative_window_uses_normalized_depth(self) -> None:
        self.assertEqual(RelativeWindow(0.4, 0.6).resolve(6), (2, 3))


if __name__ == "__main__":
    unittest.main()

