"""Scheduler-neutral mechanical checks against fixtures or pinned HF checkpoints.

Run ``python -m actsteer.validation.smoke --help``. A fixture pass is explicitly
not a real-checkpoint pass. Instrumentation directions have no scientific meaning.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import re
import subprocess
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from actsteer.core.artifacts import ArtifactStore
from actsteer.core.calibration import CalibrationMethod, calibrate_layers
from actsteer.core.capture.capture import capture_reduced_forward
from actsteer.core.capture.layers import AbsoluteLayers, RelativeLayers
from actsteer.core.capture.reducers import hidden_from_output
from actsteer.core.capture.sites import ActivationPoint, ActivationSite
from actsteer.core.capture.tokens import (
    ExplicitTokenSpan,
    LastValidToken,
    TaggedSpan,
    resolve_token_mask,
)
from actsteer.core.config import ModelConfig
from actsteer.core.interventions import InterventionWindow, SteeringController
from actsteer.core.interventions.strength import allocate_amplitudes, build_torch_deltas
from actsteer.core.models import load_model
from actsteer.core.prompts import PromptSection, PromptSpec

SEED = 1729
NEW_TOKENS = 4


def git_identity() -> dict[str, Any]:
    # Identify the code being imported, even when launched outside the checkout.
    root = Path(__file__).resolve().parents[3]

    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()

    try:
        status = git("status", "--porcelain", "--untracked-files=all")
        return {"sha": git("rev-parse", "HEAD"), "clean": not status, "status": status}
    except (OSError, subprocess.CalledProcessError):
        return {"sha": None, "clean": False, "status": "source checkout unavailable"}


def environment() -> dict[str, Any]:
    versions = {}
    for package in (
        "actsteer",
        "torch",
        "transformers",
        "accelerate",
        "numpy",
        "tokenizers",
        "safetensors",
        "huggingface-hub",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    result = {"python": sys.version, "platform": platform.platform(), "packages": versions}
    try:
        import torch

        result["cuda"] = torch.version.cuda
        result["cuda_available"] = torch.cuda.is_available()
        result["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        result["cudnn"] = torch.backends.cudnn.version()
    except ImportError:
        result["cuda_available"] = False
        result["gpus"] = []
    return result


def memory() -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        return {"available": False, "reason": "CUDA unavailable; CPU fixture only"}
    torch.cuda.synchronize()
    return {
        "available": True,
        "allocated_bytes": torch.cuda.memory_allocated(),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "reserved_bytes": torch.cuda.memory_reserved(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


def hook_inventory(model) -> dict[str, tuple[int, int]]:
    return {
        name: (len(module._forward_pre_hooks), len(module._forward_hooks))
        for name, module in model.named_modules()
    }


def exercise(adapter, report: dict, output: Path, *, mode: str, atol: float, rtol: float):
    import numpy as np
    import torch
    from transformers import GenerationConfig

    model = adapter.model
    device = model.get_input_embeddings().weight.device
    original_hooks = hook_inventory(model)
    checks = report["checks"]

    def passed(name, **details):
        checks.append({"name": name, "status": "pass", **details})

    def equal(name, actual, expected):
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
        passed(name, max_abs_error=float((actual.float() - expected.float()).abs().max()))

    def cleanup():
        assert hook_inventory(model) == original_hooks, "hook inventory changed"

    def render(words):
        prompt = PromptSpec(
            (PromptSection("payload", words), PromptSection("instruction", "Repeat the last word."))
        )
        rendered = (
            adapter.render_tagged_chat(prompt)
            if mode == "chat"
            else adapter.render_tagged_text(prompt)
        )
        ids = torch.tensor([rendered.token_ids], device=device)
        inputs = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
        spans = {name: [span] for name, span in rendered.token_spans.items()}
        mask = resolve_token_mask(
            TaggedSpan("payload"), inputs["attention_mask"], tagged_spans=spans
        )
        start, stop = rendered.token_spans["payload"]
        assert 0 < int(mask.sum()) < ids.shape[1]
        assert torch.equal(
            mask, resolve_token_mask(ExplicitTokenSpan(start, stop), inputs["attention_mask"])
        )
        last = resolve_token_mask(LastValidToken(), inputs["attention_mask"])
        assert int(last.sum()) == 1 and bool(last[0, -1])
        return rendered, inputs, mask

    rendered, inputs, mask = render("alpha beta gamma delta")
    report["prompt"] = {
        "text": rendered.text,
        "token_ids": rendered.token_ids,
        "spans": rendered.token_spans,
        "decoded_spans": {
            name: adapter.tokenizer.decode(rendered.token_ids[start:stop])
            for name, (start, stop) in rendered.token_spans.items()
        },
    }
    passed("named_spans_and_token_selectors")
    # Always exercise the base/plain path as well as the requested input mode.
    plain = adapter.render_tagged_text(PromptSpec((PromptSection("payload", "alpha beta"),)))
    assert plain.token_ids and plain.token_spans["payload"]
    passed("plain_text_rendering")
    layers = RelativeLayers((0.25, 0.75)).resolve(adapter.num_layers)
    assert len(layers) == 2 and adapter.num_layers >= 3
    assert AbsoluteLayers(layers).resolve(adapter.num_layers) == layers
    report["layers"] = layers
    report["adapter"] = type(adapter).__name__
    report["model_type"] = model.config.model_type
    report["parameter_count"] = sum(p.numel() for p in model.parameters())
    report["numerics"] = {
        "dtype": str(next(model.parameters()).dtype),
        "attention_implementation": model.config._attn_implementation,
        "atol": atol,
        "rtol": rtol,
        "seed": SEED,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "tf32": torch.backends.cuda.matmul.allow_tf32,
        "generation": {"do_sample": False, "new_tokens": NEW_TOKENS, "use_cache": True},
    }
    passed("layer_selectors_and_adapter")

    def forward():
        return model(**inputs, use_cache=False).logits.detach().cpu()

    generation_config = GenerationConfig(
        do_sample=False,
        max_new_tokens=NEW_TOKENS,
        min_new_tokens=NEW_TOKENS,
        use_cache=True,
        pad_token_id=adapter.tokenizer.pad_token_id,
        eos_token_id=None,
        bos_token_id=adapter.tokenizer.bos_token_id,
    )

    def generate():
        return model.generate(**inputs, generation_config=generation_config).detach().cpu()

    baseline = forward()
    assert bool(torch.isfinite(baseline).all())
    baseline_tokens = generate()
    assert baseline_tokens.shape[1] == inputs["input_ids"].shape[1] + NEW_TOKENS
    passed("unhooked_baseline", generated_token_ids=baseline_tokens.tolist())
    equal("baseline_repeat", forward(), baseline)
    assert torch.equal(generate(), baseline_tokens)
    passed("deterministic_baseline_generation")

    # Four neutral lexical fixtures provide independent calibration observations.
    captures = {layer: [] for layer in layers}
    for words in (
        "alpha beta gamma delta",
        "one two three four",
        "delta gamma beta alpha",
        "four three two one",
    ):
        _, sample, selected = render(words)
        capture_output, reduced = capture_reduced_forward(
            adapter, {**sample, "use_cache": False}, layers=layers, token_mask=selected
        )
        del capture_output
        for layer in layers:
            assert reduced[layer].shape == (1, adapter.hidden_size)
            assert reduced[layer].device.type == "cpu" and reduced[layer].dtype == torch.float32
            assert bool(torch.isfinite(reduced[layer]).all())
            captures[layer].append(reduced[layer].numpy()[0])
        cleanup()
    captures = {layer: np.stack(values) for layer, values in captures.items()}
    # Independent direct-index oracle checks both capture sites and the reduction.
    capture_errors = []
    for point in ActivationPoint:
        observed = {}
        handles = []

        def observe(layer, hidden, observed=observed):
            selected = hidden[0, mask[0]].double()
            observed[layer] = (
                selected.mean(0, keepdim=True).cpu(),
                selected.abs().mean(0, keepdim=True).cpu(),
                torch.finfo(hidden.dtype).eps,
            )

        try:
            for layer in layers:
                block = adapter.get_block(layer)
                if point is ActivationPoint.PRE:
                    handles.append(
                        block.register_forward_pre_hook(
                            lambda _m, values, k=layer: observe(k, values[0])
                        )
                    )
                else:
                    handles.append(
                        block.register_forward_hook(
                            lambda _m, _i, value, k=layer: observe(k, hidden_from_output(value))
                        )
                    )
            _, reduced = capture_reduced_forward(
                adapter,
                {**inputs, "use_cache": False},
                layers=layers,
                token_mask=mask,
                site=ActivationSite(point=point),
            )
            for layer in layers:
                reference, magnitude, eps = observed[layer]
                # Capture reduces in model dtype; allow rounding relative to a
                # float64 direct-index mean, including cancellation near zero.
                error = (reduced[layer].double() - reference).abs()
                bound = 4 * eps * magnitude + 1e-7
                assert bool((error <= bound).all()), "capture differs from selected-token mean"
                capture_errors.append(
                    {
                        "site": point.value,
                        "layer": layer,
                        "max_abs_error": float(error.max()),
                        "max_rounding_bound": float(bound.max()),
                    }
                )
        finally:
            for handle in handles:
                handle.remove()
    cleanup()
    passed(
        "reduced_pre_and_post_capture",
        shapes={k: list(v.shape) for k, v in captures.items()},
        rounding_bound="per component: 4 * dtype_eps * mean(abs(selected)) + 1e-7",
        errors=capture_errors,
    )
    rng = np.random.default_rng(SEED)
    vectors = {layer: rng.normal(size=adapter.hidden_size).astype(np.float32) for layer in layers}
    vectors = {layer: vector / np.linalg.norm(vector) for layer, vector in vectors.items()}
    calibration = calibrate_layers(
        captures, method=CalibrationMethod.PROJECTION_SD, vectors=vectors
    )
    residual = calibrate_layers(captures, method=CalibrationMethod.RESIDUAL_NORM)
    amplitudes = allocate_amplitudes(4.0, calibration.scales)
    assert np.isclose(sum((amplitudes[k] / calibration.scales[k]) ** 2 for k in layers), 16)
    deltas = build_torch_deltas(
        vectors, amplitudes, device=device, dtype=next(model.parameters()).dtype
    )
    passed(
        "direction_calibration_and_total_budget",
        projection_sd=dict(calibration.scales),
        residual_norm=dict(residual.scales),
        sample_counts=dict(calibration.sample_counts),
        amplitudes=amplitudes,
        direction="seeded Gaussian unit vector; instrumentation only",
    )

    for name, values, selected in (
        ("zero_strength", {k: torch.zeros_like(v) for k, v in deltas.items()}, mask),
        ("inactive_mask", deltas, torch.zeros_like(mask)),
    ):
        with SteeringController(adapter, values).active(selected):
            equal(name + "_logits", forward(), baseline)
            assert torch.equal(generate(), baseline_tokens)
        cleanup()
        passed(name + "_generation_and_cleanup")

    # Bracket the steering hook at EVERY block. This isolates the injected change
    # from normal downstream propagation; comparing whole model states cannot do that.
    def audited_generation(window, selected_deltas):
        before = {}
        events = []
        handles = []
        pass_number = [-1]
        controller = SteeringController(adapter, selected_deltas, window=window)

        def count_pass(_module, _args, kwargs):
            pass_number[0] += 1
            ids = kwargs["input_ids"]
            if pass_number[0] == 0:
                assert ids.shape[1] == mask.shape[1]
            else:
                assert ids.shape[1] == 1, "generation did not use single-token cached decoding"
                cache = kwargs.get("past_key_values")
                assert cache is not None, "decoding has no KV cache"
                if hasattr(cache, "get_seq_length"):
                    assert int(cache.get_seq_length()) > 0

        def observe_before(layer):
            def observe(_module, _args, value):
                before[layer] = hidden_from_output(value).detach().clone()

            return observe

        def observe_after(layer):
            def observe(_module, _args, value):
                actual = hidden_from_output(value)
                prior = before.pop(layer)
                prefill = pass_number[0] == 0
                active = layer in selected_deltas and (
                    (prefill and window is not InterventionWindow.RESPONSE_ONLY)
                    or (not prefill and window is not InterventionWindow.ENCODE_RELEASE)
                )
                expected = prior
                if active:
                    selected = mask[..., None] if prefill else torch.ones_like(prior[..., :1])
                    expected = (
                        prior + selected.to(prior) * selected_deltas[layer].to(prior)[None, None]
                    )
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                changed = (actual != prior).any(dim=-1)
                expected_tokens = mask if prefill else torch.ones_like(changed)
                assert torch.equal(
                    changed, expected_tokens if active else torch.zeros_like(changed)
                ), "injection is absent, rounded away, or changed an unselected token"
                events.append(
                    {
                        "pass": pass_number[0],
                        "layer": layer,
                        "phase": "prefill" if prefill else "cached_decode",
                        "changed_tokens": int(changed.sum()),
                        "active": active,
                    }
                )

            return observe

        try:
            handles.append(model.register_forward_pre_hook(count_pass, with_kwargs=True))
            for layer in range(adapter.num_layers):
                handles.append(
                    adapter.get_block(layer).register_forward_hook(observe_before(layer))
                )
            with controller.active(mask):
                for layer in range(adapter.num_layers):
                    handles.append(
                        adapter.get_block(layer).register_forward_hook(observe_after(layer))
                    )
                generated = generate()
        finally:
            for handle in handles:
                handle.remove()
        cleanup()
        assert pass_number[0] == NEW_TOKENS - 1
        assert len(events) == NEW_TOKENS * adapter.num_layers
        return generated, events

    steered = None
    for window in InterventionWindow:
        generated, events = audited_generation(window, deltas)
        passed(
            window.value + "_layer_token_and_cache_selectivity",
            events=events,
            generated_token_ids=generated.tolist(),
        )
        if window is InterventionWindow.ENCODE_RELEASE:
            steered = generated
    _, events = audited_generation(
        InterventionWindow.ENCODE_RELEASE, {layers[0]: deltas[layers[0]]}
    )
    passed("single_layer_selectivity", events=events)

    # Force failure INSIDE a model forward, after hooks were installed.
    def fail(*_args):
        raise RuntimeError("deliberate validation exception")

    failure = adapter.get_block(layers[-1]).register_forward_hook(fail)
    try:
        try:
            with SteeringController(adapter, deltas).active(mask):
                forward()
        except RuntimeError as error:
            assert str(error) == "deliberate validation exception"
        else:
            raise AssertionError("forced exception did not fire")
        # capture's finally must also run if an earlier block already captured.
        try:
            capture_reduced_forward(adapter, inputs, layers=layers, token_mask=mask)
        except RuntimeError as error:
            assert str(error) == "deliberate validation exception"
        else:
            raise AssertionError("capture exception did not fire")
    finally:
        failure.remove()
    cleanup()
    # Invalid later layer must clean up partial hook installation too.
    try:
        with SteeringController(
            adapter, {layers[0]: deltas[layers[0]], adapter.num_layers: deltas[layers[0]]}
        ).active(mask):
            raise AssertionError("invalid layer accepted")
    except IndexError:
        pass
    cleanup()
    passed("normal_exception_and_partial_install_cleanup")
    equal("baseline_after_cleanup", forward(), baseline)
    assert torch.equal(generate(), baseline_tokens)
    passed("baseline_generation_after_cleanup")

    payloads = {}
    for layer in layers:
        for label, value in (("capture", captures[layer]), ("vector", vectors[layer])):
            buffer = io.BytesIO()
            np.save(buffer, value, allow_pickle=False)
            payloads[f"{label}-{layer}.npy"] = buffer.getvalue()
    metadata = {
        "layers": layers,
        "calibration_scales": dict(calibration.scales),
        "amplitudes": amplitudes,
        "seed": SEED,
        "purpose": "instrumentation only",
    }
    payloads["metadata.json"] = metadata
    provenance = {key: report[key] for key in ("code", "model", "kind", "numerics", "prompt")}
    provenance["environment"] = report["environment"]
    store = ArtifactStore(output / "artifacts")
    ref = store.write("instrumentation", provenance, payloads)
    assert store.write("instrumentation", provenance, payloads) == ref
    store.verify(ref)
    loaded_metadata = json.loads((ref.path / "metadata.json").read_text())
    loaded_vectors = {}
    for layer in layers:
        loaded_capture = np.load(ref.path / f"capture-{layer}.npy", allow_pickle=False)
        np.testing.assert_array_equal(loaded_capture, captures[layer])
        loaded_vectors[layer] = np.load(ref.path / f"vector-{layer}.npy", allow_pickle=False)
        np.testing.assert_array_equal(loaded_vectors[layer], vectors[layer])
    loaded_deltas = build_torch_deltas(
        loaded_vectors,
        {int(k): v for k, v in loaded_metadata["amplitudes"].items()},
        device=device,
        dtype=next(model.parameters()).dtype,
    )
    assert steered is not None
    with SteeringController(adapter, loaded_deltas).active(mask):
        assert torch.equal(generate(), steered)
    cleanup()
    report["artifact"] = str(ref.path)
    report["artifact_id"] = ref.artifact_id
    passed("immutable_artifact_reload_and_deterministic_reuse")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", action="store_true", help="offline random CPU model only")
    parser.add_argument("--family", required=True, choices=["gemma", "llama", "qwen"])
    parser.add_argument("--manifest", type=Path, default=Path("configs/validation-models.json"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mode", choices=["chat", "text"], default="chat")
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--rtol", type=float, default=0.0)
    args = parser.parse_args(argv)
    args.output = args.output.resolve()
    # Never overwrite prior evidence. Use a fresh run directory for each attempt.
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema_version": 1,
        "started_at": datetime.now(UTC).isoformat(),
        "kind": "local_fixture" if args.fixture else "real_checkpoint",
        "status": "fail",
        "family": args.family,
        "code": git_identity(),
        "environment": environment(),
        "checks": [],
        "command": sys.argv,
        "stage": "preflight",
        "real_checkpoint_gate": False,
    }
    try:
        import torch

        if not __debug__:
            raise RuntimeError("validation must run without Python -O / PYTHONOPTIMIZE")
        if not (0 <= args.atol < float("inf") and 0 <= args.rtol < float("inf")):
            raise ValueError("tolerances must be finite and non-negative")
        torch.manual_seed(SEED)
        torch.set_num_threads(4)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if args.fixture:
            from actsteer.validation.fixtures import write_fixture

            fixture_path = args.output / "fixture"
            write_fixture(args.family, fixture_path)
            config = ModelConfig(id=str(fixture_path), dtype="float32", mode=args.mode)
            report["model"] = {
                "id": f"random-{args.family}",
                "revision": None,
                "tokenizer_revision": None,
                "seed": SEED,
                "mode": args.mode,
                "fixture_sha256": {
                    p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted(fixture_path.iterdir())
                },
            }
        else:
            rows = json.loads(args.manifest.read_text())["models"]
            (row,) = [row for row in rows if row["family"] == args.family]
            config = ModelConfig.model_validate({k: v for k, v in row.items() if k != "family"})
            report["model"] = config.model_dump()
            for revision in (config.revision, config.tokenizer_revision):
                if not revision or not re.fullmatch("[0-9a-f]{40}", revision):
                    raise ValueError("real checkpoints require immutable model/tokenizer SHAs")
            if not report["code"]["clean"]:
                raise RuntimeError("real-checkpoint evidence requires a clean source Git checkout")
            if not torch.cuda.is_available():
                raise RuntimeError("real-checkpoint gate requires an allocated H100/H200 CUDA GPU")
            if not any(name in torch.cuda.get_device_name(0) for name in ("H100", "H200")):
                raise RuntimeError("real-checkpoint gate requires H100 or H200")
            torch.cuda.reset_peak_memory_stats()
        report["stage"] = "load_model"
        adapter = load_model(
            config, device_map="cpu" if args.fixture else "cuda:0", attn_implementation="eager"
        )
        assert adapter.family.value == args.family
        if not args.fixture:
            assert adapter.model.config._commit_hash == config.revision
            assert all(p.device.type == "cuda" for p in adapter.model.parameters())
            report["model"]["resolved_model_revision"] = adapter.model.config._commit_hash
            # Tokenizer files are loaded by exact SHA by load_model, never a moving branch.
            report["model"]["resolved_tokenizer_revision"] = config.tokenizer_revision
        report["memory_after_load"] = memory()
        report["stage"] = "exercise"
        with torch.inference_mode():
            exercise(adapter, report, args.output, mode=config.mode, atol=args.atol, rtol=args.rtol)
        if git_identity() != report["code"]:
            raise RuntimeError("source checkout changed during validation")
        report["status"] = "pass"
        report["real_checkpoint_gate"] = not args.fixture
    except Exception as error:
        report["error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
            "guidance": "Check dependencies, CUDA allocation, HF checkpoint access "
            "and cache space; inspect the last passing check and traceback.",
        }
    finally:
        try:
            report["memory_final"] = memory()
        except Exception as error:
            report["memory_final"] = {"error": str(error)}
        report["finished_at"] = datetime.now(UTC).isoformat()
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "kind": report["kind"],
                "report": str(args.output / "report.json"),
                "error": report.get("error"),
            }
        )
    )
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
