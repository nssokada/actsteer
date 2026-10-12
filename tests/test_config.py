from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from actsteer.core.config import ActSteerConfig


class ConfigTests(unittest.TestCase):
    def test_yaml_round_trip(self) -> None:
        config = ActSteerConfig.model_validate(
            {
                "name": "smoke",
                "model": {"id": "example/model", "dtype": "bfloat16"},
                "artifacts": {"root": "/tmp/actsteer-artifacts"},
                "seed": 7,
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            config.to_yaml(path)
            loaded = ActSteerConfig.from_yaml(path)
        self.assertEqual(config, loaded)

    def test_unknown_fields_fail(self) -> None:
        with self.assertRaises(ValueError):
            ActSteerConfig.model_validate(
                {
                    "name": "bad",
                    "model": {"id": "example/model"},
                    "artifacts": {"root": "/tmp/artifacts"},
                    "unexpected": True,
                }
            )


if __name__ == "__main__":
    unittest.main()

