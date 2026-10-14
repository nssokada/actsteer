from __future__ import annotations

import importlib.util
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from actsteer.core.artifacts import ArtifactStore
from actsteer.validation.accept import REQUIRED_CHECKS, accept

HAS_MODELS = all(importlib.util.find_spec(name) for name in ("torch", "transformers"))


class AcceptanceTests(unittest.TestCase):
    def test_aggregate_verifies_artifacts_checks_and_exact_model_identity(self):
        # Synthetic unit data tests the gate, never constitutes checkpoint evidence.
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(directory)
            reports, models = [], []
            for family in ("gemma", "llama", "qwen"):
                model = dict(
                    id=family,
                    revision="b" * 40,
                    tokenizer_revision="b" * 40,
                    mode="chat",
                    dtype="bfloat16",
                )
                code = dict(sha="a" * 40, clean=True)
                ref = store.write(
                    "instrumentation",
                    {"code": code, "model": model},
                    {"test.txt": "synthetic unit test"},
                )
                models.append(dict(family=family, **model))
                reports.append(
                    dict(
                        family=family,
                        kind="real_checkpoint",
                        status="pass",
                        real_checkpoint_gate=True,
                        code=code,
                        model=model,
                        checks=[dict(name=name, status="pass") for name in REQUIRED_CHECKS],
                        memory_final={"available": True},
                        environment={"gpus": ["NVIDIA H100"]},
                        numerics=dict(
                            dtype="torch.bfloat16", atol=0, rtol=0, deterministic_algorithms=True
                        ),
                        artifact=str(ref.path),
                        artifact_id=ref.artifact_id,
                    )
                )
            self.assertEqual(accept(reports, models, "a" * 40)["status"], "validated")
            missing = deepcopy(reports)
            missing[0]["checks"].pop()
            with self.assertRaisesRegex(ValueError, "checks missing"):
                accept(missing, models, "a" * 40)
            wrong = deepcopy(reports)
            wrong[0]["model"]["tokenizer_revision"] = "c" * 40
            with self.assertRaisesRegex(ValueError, "provenance mismatch"):
                accept(wrong, models, "a" * 40)
            with self.assertRaisesRegex(ValueError, "clean Git SHA"):
                accept(reports, models, "d" * 40)
            (Path(reports[0]["artifact"]) / "test.txt").write_text("tampered")
            with self.assertRaisesRegex(ValueError, "checksum"):
                accept(reports, models, "a" * 40)

    def test_fixture_reports_cannot_satisfy_gate(self):
        reports = [
            {"family": family, "kind": "local_fixture", "status": "pass"}
            for family in ("gemma", "llama", "qwen")
        ]
        models = [{"family": family} for family in ("gemma", "llama", "qwen")]
        with self.assertRaisesRegex(ValueError, "real-checkpoint"):
            accept(reports, models, "a" * 40)

    def test_missing_family_cannot_satisfy_gate(self):
        with self.assertRaisesRegex(ValueError, "three"):
            accept([], [], "a" * 40)

    def test_dirty_source_cannot_satisfy_gate(self):
        reports = [
            {
                "family": family,
                "kind": "real_checkpoint",
                "status": "pass",
                "real_checkpoint_gate": True,
                "code": {"sha": "a" * 40, "clean": False},
            }
            for family in ("gemma", "llama", "qwen")
        ]
        models = [{"family": family} for family in ("gemma", "llama", "qwen")]
        with self.assertRaisesRegex(ValueError, "clean Git SHA"):
            accept(reports, models, "a" * 40)


@unittest.skipUnless(HAS_MODELS, "requires optional model dependencies")
class InstrumentationValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch

        from actsteer.core.config import ModelConfig
        from actsteer.core.models import load_model
        from actsteer.validation.fixtures import write_fixture

        torch.set_num_threads(2)
        cls.directory = tempfile.TemporaryDirectory()
        fixture = Path(cls.directory.name) / "model"
        write_fixture("llama", fixture)
        cls.adapter = load_model(
            ModelConfig(id=str(fixture), dtype="float32"),
            device_map="cpu",
            attn_implementation="eager",
        )

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_harness_detects_missing_intervention(self):
        import torch

        from actsteer.core.interventions import SteeringController
        from actsteer.validation.smoke import exercise, hook_inventory

        inventory = hook_inventory(self.adapter.model)
        report = {"checks": []}
        with patch.object(SteeringController, "_hook", return_value=lambda *_args: None):
            with torch.inference_mode(), self.assertRaises(AssertionError):
                exercise(
                    self.adapter, report, Path(self.directory.name), mode="chat", atol=0, rtol=0
                )
        self.assertEqual(inventory, hook_inventory(self.adapter.model))
        self.assertEqual(report["checks"][-1]["name"], "inactive_mask_generation_and_cleanup")
        self.assertNotIn(
            "encode_release_layer_token_and_cache_selectivity",
            [check["name"] for check in report["checks"]],
        )

    def test_reduced_capture_matches_masked_rows_and_cleans_after_error(self):
        import torch

        from actsteer.core.capture.capture import capture_reduced_forward
        from actsteer.core.capture.reducers import masked_mean
        from actsteer.core.capture.tokens import LastValidToken, resolve_token_mask
        from actsteer.validation.smoke import hook_inventory

        hidden = torch.arange(24.0).reshape(2, 3, 4)
        attention = torch.tensor([[0, 1, 1], [1, 1, 0]])
        mask = resolve_token_mask(LastValidToken(), attention)
        torch.testing.assert_close(masked_mean(hidden, mask), hidden[[0, 1], [2, 1]])
        inventory = hook_inventory(self.adapter.model)
        with self.assertRaisesRegex(ValueError, "at least one token"):
            capture_reduced_forward(
                self.adapter,
                {"input_ids": torch.tensor([[5, 6, 7]])},
                layers=[1, 2],
                token_mask=torch.zeros(1, 3),
            )
        self.assertEqual(inventory, hook_inventory(self.adapter.model))


if __name__ == "__main__":
    unittest.main()
