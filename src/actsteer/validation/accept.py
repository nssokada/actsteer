"""Fail closed unless all pinned families passed on one clean Git revision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from actsteer.core.artifacts import ArtifactRef, ArtifactStore

REQUIRED_CHECKS = {
    "named_spans_and_token_selectors",
    "plain_text_rendering",
    "layer_selectors_and_adapter",
    "unhooked_baseline",
    "baseline_repeat",
    "deterministic_baseline_generation",
    "reduced_pre_and_post_capture",
    "direction_calibration_and_total_budget",
    "zero_strength_logits",
    "zero_strength_generation_and_cleanup",
    "inactive_mask_logits",
    "inactive_mask_generation_and_cleanup",
    "encode_release_layer_token_and_cache_selectivity",
    "response_only_layer_token_and_cache_selectivity",
    "continuous_layer_token_and_cache_selectivity",
    "single_layer_selectivity",
    "normal_exception_and_partial_install_cleanup",
    "baseline_after_cleanup",
    "baseline_generation_after_cleanup",
    "immutable_artifact_reload_and_deterministic_reuse",
}


def accept(reports: list[dict], models: list[dict], sha: str):
    if len(reports) != len(models) or len(models) != 3:
        raise ValueError("exactly three selected model reports are required")
    if {r["family"] for r in reports} != {"gemma", "llama", "qwen"}:
        raise ValueError("one Gemma, Llama, and Qwen report required")
    for report in reports:
        (row,) = [row for row in models if row["family"] == report["family"]]
        if (
            report["kind"] != "real_checkpoint"
            or report["status"] != "pass"
            or not report["real_checkpoint_gate"]
        ):
            raise ValueError(f"{row['family']}: no passing real-checkpoint evidence")
        if report["code"]["sha"] != sha or not report["code"]["clean"]:
            raise ValueError("reports must identify the requested clean Git SHA")
        for key in ("id", "revision", "tokenizer_revision", "mode", "dtype"):
            if report["model"][key] != row[key]:
                raise ValueError(f"model provenance mismatch: {key}")
        if not REQUIRED_CHECKS <= {c["name"] for c in report["checks"] if c["status"] == "pass"}:
            raise ValueError("required checks missing")
        if any(c["status"] != "pass" for c in report["checks"]):
            raise ValueError("a model check failed")
        if not report["memory_final"]["available"]:
            raise ValueError("GPU memory diagnostics missing")
        if not any("H100" in name or "H200" in name for name in report["environment"]["gpus"]):
            raise ValueError("H100/H200 evidence missing")
        numerics = report["numerics"]
        if (
            numerics["dtype"] != "torch.bfloat16"
            or numerics["atol"] != 0
            or numerics["rtol"] != 0
            or not numerics["deterministic_algorithms"]
        ):
            raise ValueError("acceptance requires deterministic BF16 and exact no-op equality")
        path = Path(report["artifact"])
        ref = ArtifactRef("instrumentation", report["artifact_id"], path)
        manifest = ArtifactStore(path.parent.parent).verify(ref)
        if manifest.inputs["code"] != report["code"] or manifest.inputs["model"] != report["model"]:
            raise ValueError("artifact provenance does not match report")
    return {"status": "validated", "sha": sha, "families": sorted(r["family"] for r in reports)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("configs/validation-models.json"))
    parser.add_argument("--sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("reports", type=Path, nargs="+")
    args = parser.parse_args(argv)
    try:
        result = accept(
            [json.loads(p.read_text()) for p in args.reports],
            json.loads(args.manifest.read_text())["models"],
            args.sha,
        )
    except Exception as error:
        result = {"status": "not_validated", "sha": args.sha, "error": str(error)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result))
    return 0 if result["status"] == "validated" else 1


if __name__ == "__main__":
    raise SystemExit(main())
