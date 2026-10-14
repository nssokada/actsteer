# Instrumentation validation

A revision is accepted only after all three pinned real checkpoints pass on
H100/H200 and `actsteer.validation.accept` emits `status: validated` for that
same clean Git SHA. A local fixture pass, accessible model metadata, a submitted
job, or a prepared command is not acceptance. Keep the JSON reports and immutable
artifacts with the downstream study's dependency pin.

**Validated run (2026-10-12):** all three pinned checkpoints passed 20/20 checks
on NVIDIA H100 80GB HBM3 in BF16 at clean source SHA
`a960c22107d720cfc515e00b45c126c184135559`. See the
[portable validation record](validation/2026-10-12-a960c221.md) for results,
environment, and evidence checksums. This remains the experimental pin; newer
documentation commits are not additional GPU-validated source revisions.

The harness is scheduler-neutral. It uses a seeded Gaussian unit vector and four
small lexical prompts solely to test instrumentation. It contains no behavioral
classifier, study dataset, hypothesis, or scientific interpretation.

## Local smoke

Use Python 3.11 or later from the repository root:

```bash
python -m pip install -e '.[models,dev]' -c requirements/validation-constraints.txt
bash scripts/validate-local.sh runs/local-validation
```

Use a new output directory on each attempt. The script runs the unit suite,
Ruff, Pyright, the config CLI, six offline CPU smoke runs (Gemma/Llama/Qwen ×
chat/text), and rejection of fixture reports by the real-checkpoint gate.
The fixture saves randomly initialized tiny Transformers models and a synthetic
fast tokenizer, then reloads them through ActSteer's normal `load_model` API.
No checkpoint weights or production tokenizers are used in these local runs.
`pip-freeze.txt`, individual logs, and each run's `report.json` are the evidence.

A single local run is:

```bash
python -m actsteer.validation.smoke --fixture --family gemma --mode chat \
  --output runs/gemma-fixture
```

## What the model harness proves

Every successful run records and checks:

- Environment, imported source Git SHA and cleanliness, adapter, model identity,
  tokenizer revision, dtype, attention implementation, seed, and tolerances.
- Native chat rendering with named `payload` and `instruction` spans, decoded
  spans for inspection, plain-text rendering, absolute/relative layer selectors,
  and token/interval/tag selectors.
- Finite unhooked logits and repeatable greedy generation, zero-strength and
  inactive-mask equivalence, plus baseline repetition after cleanup.
- CPU float32 reduced captures at two selected depths, both pre- and post-block
  sites, checked against a float64 direct-index selected-token mean.
- Projection-SD and residual-norm calibration and matched total-budget multi-layer
  amplitudes; deterministic unit directions test instrumentation only.
- Single- and multi-layer steering; independent hooks bracket the addition at
  every block and check its exact tensor value and changed-token mask. They
  measure the injection itself; normal downstream effects can propagate.
- Encode-release, response-only, and continuous windows through an actual cached
  `generate` call. Every run observes prefill and three single-token decode
  passes with a nonempty KV cache. Encode-release must inject nothing during
  cached decoding. Response-only skips prefill, including its first-token logits.
- Hook-inventory restoration after success, exceptions inside model forwards,
  capture errors, and partial installation failure.
- Peak/current CUDA allocation and reservation, immutable checksummed NumPy
  captures/vectors plus JSON calibration, reload equality, idempotent writes,
  and identical generated token IDs when reusing reloaded directions.

Baseline/no-op comparisons default to **atol=0, rtol=0**, and acceptance requires
these exact tolerances. Injection-site additions also use exact equality.
The independent reduction oracle allows model-dtype rounding, per component:
`4 * dtype_epsilon * mean(abs(selected_activations)) + 1e-7`; reports retain the
observed errors and bounds. This does not relax no-op or token-selectivity checks.
Real runs use unquantized BF16, eager attention, deterministic PyTorch algorithms,
TF32 disabled, seed 1729, and four new greedy tokens. Deterministic failures are
failures, not silently relaxed. Run without Python `-O` / `PYTHONOPTIMIZE`.

## Pinned checkpoints and access preflight

`configs/validation-models.json` fixes model and tokenizer to the same immutable
revision (resolved from Hugging Face on 2026-10-12):

| Family | Checkpoint | Revision |
|---|---|---|
| Gemma | `google/gemma-2-9b-it` | `11c9b309abf73637e4b6f9a3fa1e92e615547819` |
| Llama | `meta-llama/Llama-3.1-8B-Instruct` | `0e9e39f249a16976918f6564b8830bc894c89659` |
| Qwen | `Qwen/Qwen2.5-7B-Instruct` | `a09a35458c702b33eeacc393d103063234e8bc28` |

```bash
python -m actsteer.validation.prepare --output runs/checkpoint-access.json
```

This checks the server's revision and access to config, weight index, and
production tokenizer metadata. It does not load weights or satisfy the gate.
Use `--download` on a network-enabled HPC login node to populate `HF_HOME`
before an offline compute-node job. For gated repositories, authenticate with
`hf auth login` using an account that has the required model access.
Never put authentication tokens in reports or job scripts.

## Reproduce the validated HPC run

Use a separate, clean checkout of the **tested source**, not the documentation
branch's current HEAD. From a network-enabled host:

```bash
export ACTSTEER_SHA=a960c22107d720cfc515e00b45c126c184135559
git clone https://github.com/nssokada/actsteer.git "$HOME/actsteer-validation"
cd "$HOME/actsteer-validation"
git checkout --detach "$ACTSTEER_SHA"
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[models,dev]' -c requirements/validation-constraints.txt
```

The recorded environment used Python 3.11.15 and CUDA 13.0; match the
[recorded versions](validation/2026-10-12-a960c221.md) when reproducing it.
Use a cluster-approved persistent cache location with space for the checkpoints.
The example paths below are outside the checkout; replace them if necessary.
For gated repositories, obtain access and authenticate with `hf auth login`.

```bash
export HF_HOME="$HOME/hf-cache"
export ACTSTEER_RESULTS_ROOT="$HOME/actsteer-validation-results"
mkdir -p "$HF_HOME" "$ACTSTEER_RESULTS_ROOT"
python -m actsteer.validation.prepare --download \
  --output "runs/checkpoint-download-$ACTSTEER_SHA.json"
export ACTSTEER_RESULTS="$ACTSTEER_RESULTS_ROOT/$ACTSTEER_SHA-$(date -u +%Y%m%dT%H%M%SZ)"
test "$(git rev-parse HEAD)" = "$ACTSTEER_SHA"
test -z "$(git status --porcelain --untracked-files=all)"
sbatch --output="$ACTSTEER_RESULTS.slurm-%j.out" scripts/validate-slurm.sh
# For a separate H200 run, replace the preceding submission with:
# sbatch --gres=gpu:nvidia_h200:1 --output="$ACTSTEER_RESULTS.slurm-%j.out" scripts/validate-slurm.sh
```

**The `--output` override is required.** The pinned wrapper's default log path is
inside the checkout. Slurm creates that file before the script runs, so the
clean-working-tree preflight rejects the resulting untracked file. Both results
and scheduler logs must be outside the checkout. Create the results **parent**
directory before submitting; the wrapper creates the new run directory itself.
Use a fresh run path for every submission and preserve earlier evidence.
This documentation corrects the invocation; the tested executable is unchanged.

The wrapper requests one H100, 8 CPUs, 96 GB host RAM, and a 2-hour limit. It runs
all families sequentially even if one fails, then runs aggregate acceptance.
Adjust partition, typed GPU resource, and account flags to the site's allocation.
H200 is supported by the gate but was not exercised in this recorded run.
Inspect `squeue -u "$USER"`, `sacct -j JOBID`, and
`"$ACTSTEER_RESULTS.slurm-JOBID.out"`. A completed job alone is insufficient:
check all three reports and `acceptance.json`.

On an already allocated H100, the equivalent scheduler-neutral commands are
below. Use the same activated environment and fresh, absolute results path from
above; do not run this block in addition to submitting the wrapper for that path.

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir "$ACTSTEER_RESULTS"
python -m pip freeze > "$ACTSTEER_RESULTS/pip-freeze.txt"
nvidia-smi > "$ACTSTEER_RESULTS/nvidia-smi.txt"
for family in gemma llama qwen; do
  python -m actsteer.validation.smoke --family "$family" \
    --manifest configs/validation-models.json --output "$ACTSTEER_RESULTS/$family"
done
python -m actsteer.validation.accept --sha "$ACTSTEER_SHA" \
  --manifest configs/validation-models.json \
  --output "$ACTSTEER_RESULTS/acceptance.json" \
  "$ACTSTEER_RESULTS/gemma/report.json" "$ACTSTEER_RESULTS/llama/report.json" \
  "$ACTSTEER_RESULTS/qwen/report.json"
```

The observed result was:

```json
{
  "status": "validated",
  "sha": "a960c22107d720cfc515e00b45c126c184135559",
  "families": ["gemma", "llama", "qwen"]
}
```

## Evidence retention and copied reports

Keep the complete results directory, including every family's `artifacts/`,
`pip-freeze.txt`, `nvidia-smi.txt`, and the sibling scheduler log. Run acceptance
on the execution host before copying: the original reports contain absolute
artifact paths. After copying, preserve report bytes and remap artifact locations
only in memory as shown in the [validation record](validation/2026-10-12-a960c221.md).
Verify checksums before auditing the copied evidence.

Store raw evidence in access-controlled durable storage. Reports, manifests,
environment listings, and logs can contain private filesystem or infrastructure
identifiers, so inspect them before any public upload. The public validation
summary and checksum index are portable; they do not replace the complete archive.
