# Instrumentation validation

A revision is accepted only after all three pinned real checkpoints pass on
H100/H200 and `actsteer.validation.accept` emits `status: validated` for that
same clean Git SHA. A local fixture pass, accessible model metadata, a submitted
job, or a prepared command is not acceptance. Keep the JSON reports and immutable
artifacts with the downstream study's dependency pin.

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

## Exact HPC path

The local checkout need not be published to GitHub. Export the candidate commit:

```bash
ACTSTEER_SHA=$(git rev-parse HEAD)
git bundle create runs/actsteer-validation.bundle HEAD
scp runs/actsteer-validation.bundle nokada@login.hpc.caltech.edu:actsteer-validation.bundle
ssh nokada@login.hpc.caltech.edu
```

On the HPC host, import that bundle into a new checkout:

```bash
git init "$HOME/actsteer-validation"
git -C "$HOME/actsteer-validation" fetch "$HOME/actsteer-validation.bundle" HEAD
git -C "$HOME/actsteer-validation" checkout --detach FETCH_HEAD
cd "$HOME/actsteer-validation"
export ACTSTEER_SHA=$(git rev-parse HEAD)
# Compare this SHA with the candidate reported by the local run before proceeding.
```

Use an available Python 3.11+ environment (the repository requires >=3.11).
Cluster module availability and account association must be checked after login;
no module name or allocation account is guessed by the script.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[models,dev]' -c requirements/validation-constraints.txt
export HF_HOME="/resnick/scratch/$USER/hf-actsteer"
mkdir -p "$HF_HOME" "$HOME/actsteer-validation-results"
hf auth login
python -m actsteer.validation.prepare --download \
  --output "runs/checkpoint-download-$ACTSTEER_SHA.json"
export ACTSTEER_RESULTS="$HOME/actsteer-validation-results/$ACTSTEER_SHA-$(date -u +%Y%m%dT%H%M%SZ)"
sbatch scripts/validate-slurm.sh
# H200 alternative:
# sbatch --gres=gpu:nvidia_h200:1 scripts/validate-slurm.sh
```

The Slurm wrapper requests one H100, 8 CPUs, 96 GB host RAM, and 2 hours. It runs
all families sequentially even if one fails, then runs the aggregate gate. Add
`--account=YOUR_ACTUAL_ACCOUNT` only if required by the user's allocation. Inspect
`squeue -u "$USER"`, the `actsteer-validation-JOBID.out` log, and `sacct -j JOBID`
after submission. A job ID or exit code without the reports is insufficient.

The GPU partition and typed H100/H200 resource tokens follow the
[Caltech GPU documentation](https://www.hpc.caltech.edu/docs/documentation/gpu-computing.html).
The documented login host and password/Duo flow are in the
[Caltech quick start](https://www.hpc.caltech.edu/docs/documentation/quick-start.html).
These are documented commands, not proof of the user's allocation or authentication.

On an already allocated GPU, the exact scheduler-neutral commands are:

```bash
python -m actsteer.validation.smoke --family gemma --output "$ACTSTEER_RESULTS/gemma"
python -m actsteer.validation.smoke --family llama --output "$ACTSTEER_RESULTS/llama"
python -m actsteer.validation.smoke --family qwen --output "$ACTSTEER_RESULTS/qwen"
python -m actsteer.validation.accept --sha "$ACTSTEER_SHA" \
  --output "$ACTSTEER_RESULTS/acceptance.json" \
  "$ACTSTEER_RESULTS/gemma/report.json" "$ACTSTEER_RESULTS/llama/report.json" \
  "$ACTSTEER_RESULTS/qwen/report.json"
```

Keep the complete results directory, including `artifacts/` inside each family,
`pip-freeze.txt`, and `nvidia-smi.txt`. Run acceptance on the execution host before
copying results; report artifact paths are absolute. Verify `acceptance.json`
identifies the intended clean SHA and all three families. Only then update the
coordination repository's `STATUS.md` with the validated SHA. The toolkit itself
has no dependency on that coordination repository.

## Current execution status (2026-10-12)

Local checks pass; the real-checkpoint gate remains **not satisfied**.
No production checkpoint was loaded and no GPU job was submitted in this session.

Observed blockers:

- This execution environment is macOS ARM64 with no CUDA GPU.
- A read-only `ssh -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=yes
  nokada@login.hpc.caltech.edu ...` probe reached the server but failed with
  `Permission denied (publickey,keyboard-interactive,hostbased)` (exit 255).
- Gemma and Llama access preflight returned `401 GatedRepoError`; Qwen metadata
  access passed. Remote credentials may differ, but are not established here.

The remaining path is authenticated HPC login, authorized checkpoint download,
then the three real runs and aggregate acceptance command above. CPU fixtures
must not be presented as validated production checkpoints.
