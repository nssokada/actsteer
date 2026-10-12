# ActSteer Architecture and Migration Plan

**Status:** Implementation in progress  
**Created:** 2026-09-28  
**Target repository:** `/Users/nokada/Documents/CALTECH/actsteer`  
**Source repositories:**

- `/Users/nokada/Documents/CALTECH/llm_foraging` — existing implementation and reference study
- `/Users/nokada/Documents/CALTECH/agent-emotions` — research questions, literature, and paper planning

## 1. Purpose

ActSteer will be a locally installable, GitHub-hosted Python research toolkit for
activation capture, vector extraction, and residual-stream steering in open-weight
causal language models. It will provide a general core plus an emotion-specific
framework in one repository.

The toolkit must support reusable scientific workflows without absorbing the
domain logic of every downstream experiment. `llm_foraging` will remain the
canonical foraging study and become ActSteer's first downstream consumer.

### Implementation checkpoint

| Work unit | Status | Notes |
|---|---|---|
| WU-01 | complete | Repository, installable package, MIT license, Conda/requirements scaffolding, typed root config, CLI shell, and lean tests created 2026-09-28 |
| WU-02 | complete | Normalized records, generic JSONL/Parquet/HF-row ingestion, stable subsets/fingerprints, tagged native-chat spans, and optional Codrai adapter implemented |
| WU-03 | in progress | Gemma/Llama/Qwen dispatch, native text/chat rendering, semantic block access, and local fake-model tests complete; tiny Transformers and H100 family smokes remain |
| WU-04 | in progress | Reduced forward capture, token selectors/reducers, and immutable checksum-verifiable artifact store implemented; safetensors/Parquet capture persistence and resume remain |
| WU-05 | in progress | Three extraction presets and both calibration methods implemented and unit tested; artifact serialization and Gemma parity remain |
| WU-06 | in progress | Multi-layer dose policies and three intervention windows implemented with hook-cleanup tests; real-model zero-strength and generation checks remain |
| WU-07–WU-10 | pending | Not started |

### v0.1 success criterion

ActSteer is usable when the same end-to-end workflow runs on supported Gemma,
Llama, and Qwen instruction models:

```text
user-provided emotion dataset
  -> validated normalized examples
  -> reduced residual-stream activation capture
  -> configurable emotion-vector extraction
  -> model-calibrated residual steering
  -> token/span-targeted intervention
  -> candidate-logprob or raw-generation readout
  -> versioned artifacts with complete provenance
```

The v0.1 reference run will use a small subset such as `afraid`, `sad`, and
`happy` plus neutral examples. The number of concepts is user configuration,
not a Codrai-specific package constant.

## 2. Agreed decisions

### Repository and distribution

- Create a new standalone repository named `actsteer`.
- Use one repository and one Python distribution.
- Organize the distribution into `actsteer.core` and `actsteer.emotion`.
- Host the source on GitHub and install it from a clone or pinned Git commit.
- Do not publish to PyPI or Conda for v0.1.
- Use an MIT license.
- Keep `llm_foraging` intact as a study repository; migrate it incrementally.
- Require scientific/numerical compatibility, not legacy API or file-format
  compatibility.

### Environment and model scope

- Provide `pyproject.toml` for the installable package.
- Provide a Conda environment/requirements workflow for CUDA and HPC use.
- Use Hugging Face Transformers and native PyTorch hooks.
- Do not require NNsight or TransformerLens.
- Support base and instruction-tuned causal models; validate v0.1 primarily on
  instruction-tuned models.
- Validate v0.1 on 8B-class Gemma, Llama, and Qwen models.
- Use model-native chat templates for instruction models and literal text for
  base models.
- Add no implicit system prompt. Save semantic messages, rendered prompts, and
  token IDs.
- Target BF16/FP16 on one H100 or H200. Defer quantization.
- Use the Caltech Slurm `gpu` partition with typed GPU resources.

### Scientific scope

- Make residual-stream capture and steering first-class.
- Defer attention-head, MLP, and neuron-level intervention.
- Support single-layer and multi-layer steering.
- Use single-layer steering as the most interpretable control.
- Make normalized total-budget steering the standard multi-layer policy.
- Provide an explicit per-layer-strength policy for Sofroniew-compatible runs.
- Calibrate steering strength to each model/layer's activation distribution.
- Use neutral examples from the same source dataset for calibration by default.
- Preserve the current all-data Codrai extraction method; do not impose a split.
- Require separate external stimuli for vector validation and behavioral
  evaluation.

### Interfaces and execution

- Make the typed Python API primary for novel experiments.
- Provide CLI commands for stable, repeated workflows such as extraction,
  steering matrices, planning, shard execution, and consolidation.
- Allow validated YAML configuration for standard matrix-style workflows.
- Keep scientific configuration separate from Slurm resource profiles.
- Render Slurm scripts but require explicit `sbatch` submission.
- Make the local filesystem the authoritative tracker. Defer W&B.
- Leave downstream construct classification, statistics, and plotting to
  downstream experiment repositories.

### Data and artifacts

- Accept both plain text and structured chat messages.
- Keep the core independent of Codrai.
- Provide generic adapters for records, Parquet/JSONL, and Hugging Face
  datasets; provide Codrai as an optional emotion-dataset adapter.
- Ingest and validate user-provided data. Do not generate synthetic datasets in
  v0.1.
- Persist reduced per-example activations.
- Make full token-level activations opt-in and normally temporary.
- Recompute Gemma artifacts under ActSteer rather than importing legacy `.pt`
  bundles.
- Store reusable datasets, captures, vectors, and calibrations in a shared,
  immutable artifact root.
- Store experiment-specific outputs in project-owned run directories.

## 3. Non-goals for v0.1

- Publishing a package to PyPI or Conda.
- Supporting arbitrary hidden modules beyond the residual stream.
- Low-bit quantization.
- Distributed training or multi-node model execution.
- A general statistical-analysis or plotting framework.
- Built-in LLM judging or emotion classification of generated text.
- Synthetic emotion-dataset generation.
- W&B/MLflow integration.
- Compatibility with legacy absolute paths, environment variables, or `.pt`
  bundle layouts.
- Migrating paper prose, plots, notebooks, FID task logic, or study-specific
  regressions out of `llm_foraging`.

## 4. Architecture

### 4.1 Dependency direction

```text
actsteer.emotion
      |
      v
actsteer.core
```

`actsteer.core` must never import from `actsteer.emotion`. Downstream projects
may import either or both.

### 4.2 Proposed repository layout

```text
actsteer/
├── LICENSE
├── README.md
├── pyproject.toml
├── environment.yml
├── requirements/
│   ├── base.txt
│   └── dev.txt
├── src/
│   └── actsteer/
│       ├── __init__.py
│       ├── cli.py
│       ├── core/
│       │   ├── config.py
│       │   ├── records.py
│       │   ├── datasets.py
│       │   ├── prompts.py
│       │   ├── models/
│       │   │   ├── base.py
│       │   │   ├── transformers.py
│       │   │   ├── gemma.py
│       │   │   ├── llama.py
│       │   │   └── qwen.py
│       │   ├── capture/
│       │   │   ├── sites.py
│       │   │   ├── tokens.py
│       │   │   ├── reducers.py
│       │   │   └── capture.py
│       │   ├── interventions/
│       │   │   ├── targets.py
│       │   │   ├── windows.py
│       │   │   ├── strength.py
│       │   │   └── runtime.py
│       │   ├── readouts/
│       │   │   ├── candidate.py
│       │   │   ├── numeric.py
│       │   │   ├── generation.py
│       │   │   └── activation.py
│       │   ├── artifacts/
│       │   │   ├── manifests.py
│       │   │   ├── store.py
│       │   │   └── fingerprints.py
│       │   └── execution/
│       │       ├── plan.py
│       │       ├── shard.py
│       │       ├── consolidate.py
│       │       └── slurm.py
│       └── emotion/
│           ├── records.py
│           ├── datasets/
│           │   ├── codrai.py
│           │   └── tabular.py
│           ├── extraction/
│           │   ├── estimators.py
│           │   ├── transforms.py
│           │   └── presets.py
│           ├── calibration.py
│           └── protocols.py
├── configs/
│   └── execution/
│       ├── caltech-h100.yaml
│       └── caltech-h200.yaml
├── examples/
│   ├── extract_emotion_vectors.py
│   ├── encode_release.py
│   └── configs/
├── tests/
└── docs/
    ├── architecture.md
    ├── artifacts.md
    ├── model-support.md
    ├── slurm.md
    └── plans/
        └── architecture-and-migration.md
```

The exact module count may shrink during implementation. The boundaries are
more important than mirroring this tree mechanically.

### 4.3 Core domain objects

#### Dataset examples

```python
ConceptExample(
    id="codrai-afraid-0042",
    concept="afraid",
    text="...",                 # mutually exclusive with messages
    messages=None,
    reference="emotional",
    pair_id="optional-pair",
    split=None,
    metadata={"topic": "..."},
)
```

Validation requirements:

- stable unique ID;
- exactly one of `text` or `messages`;
- non-empty concept and content;
- declared reference/control semantics;
- pair IDs required only by estimators that use them;
- neutral examples required only by protocols that use neutral centering,
  neutral-PC removal, or neutral calibration.

#### Activation sites and token selectors

```python
ActivationSite(stream="residual", point="post", layer=20)
```

Required token selectors:

- all valid tokens;
- last valid/decision token;
- last `k` tokens;
- tokens after an offset;
- prompt tokens;
- generated tokens;
- tokens belonging to a chat role;
- tagged prompt-section spans;
- explicit character/token spans;
- custom Python selector.

#### Layer selectors

- `AbsoluteLayers([16, 20, 24])`
- `RelativeLayers([0.4, 0.5, 0.6])`
- `RelativeWindow(0.4, 0.65)`
- `SelectedLayers(artifact_id)`

Relative selectors must resolve to explicit model-specific indices before a run
starts. The resolved indices are saved in the run manifest.

#### Intervention windows

- `prefill_all`
- `decision_token`
- `generated`
- `response_only` = decision token plus generated tokens
- `continuous` = prompt, decision token, and generated tokens
- `tagged_span` for encode-release

The canonical emotion-state protocol is encode-release:

```text
steer tagged environment span during prefill
  -> remove intervention
  -> process question/decision without steering
  -> measure unsteered behavior
```

Response-only and continuous interventions are controls. A small temporal-window
validation precedes large sweeps. Persistence experiments may insert 0, 16, 64,
or 256 unsteered tokens between the steered environment and the decision.

#### Strength policies

- `ProjectionStandardDeviations(z)` — primary cross-model calibration;
- `ResidualNormFraction(c)` — migration/Sofroniew compatibility;
- `RawMagnitude(alpha)` — allowed only when explicitly requested and recorded;
- `PerLayerStrength(...)` — same calibrated dose at every target layer;
- `TotalBudget(..., weighting="uniform_l2")` — normalized multi-layer dose;
- `LayerWeights({...})` — explicit experimental weights.

Every run records raw injection norm, residual-norm fraction, projection-standard
deviations, activation norms before/after, and any norm-preservation setting.

### 4.4 Model adapter contract

Each supported adapter must provide:

- model and tokenizer loading;
- architecture metadata;
- native chat-template rendering;
- semantic residual pre/post access where available;
- explicit residual-post access for steering;
- attention-mask-aware token handling;
- hook output reconstruction without breaking tuples or cache behavior;
- prefill versus decode phase detection;
- candidate token/sequence scoring;
- generation with intervention lifecycle management;
- capability reporting for base/chat support.

The public API exposes semantic targets, not paths such as
`model.model.layers[i]`.

### 4.5 Extraction components and presets

Extraction is composition rather than a single hard-coded algorithm:

```python
ExtractionProtocol(
    token_selector=AfterOffset(50),
    token_reducer=Mean(),
    sample_reducer=Mean(),
    reference=NeutralMean(),
    transforms=[RemoveNeutralPCs(variance=0.50), L2Normalize()],
)
```

Required v0.1 presets:

| Preset | Reference | Neutral-PC removal | Normalization |
|---|---|---:|---:|
| `contrastive_mean` | neutral mean | no | L2 |
| `sofroniew_global` | mean of selected concept means | yes | L2 |
| `sofroniew_neutral` | neutral mean | yes | L2 |

`sofroniew_global` must record the complete centering population because adding
or removing concepts changes every vector.

### 4.6 Readout boundary

ActSteer produces standardized model measurements, not project conclusions.

Required readouts:

- candidate token/sequence log probabilities;
- numeric-token distributions;
- raw generated text, token IDs, and optional log probabilities;
- requested activation captures/projections.

Emotion classification, behavioral coding, regression, inference, plotting, and
paper claims remain downstream.

## 5. Artifact design

### 5.1 Shared immutable artifacts

The shared artifact root contains expensive reusable prerequisites:

```text
<artifact_root>/
├── datasets/
├── captures/
├── vectors/
├── calibrations/
└── layer-selections/
```

Project-owned run directories contain experimental outputs:

```text
<project>/runs/<run_id>/
├── submitted_config.yaml
├── resolved_config.yaml
├── plan.json
├── environment.json
├── provenance.json
├── submit.slurm
├── logs/
├── shards/
├── consolidated/
└── status.json
```

Shared artifacts are never mutated by an experiment. A changed input or method
creates a new fingerprint and artifact directory.

### 5.2 Provenance inputs

Fingerprints/manifests must cover:

- model and tokenizer IDs and revisions;
- dtype;
- dataset fingerprint and selected examples;
- semantic messages and rendered prompt template;
- activation site and resolved layers;
- token selector and reducer;
- extraction estimator/transforms;
- centering population;
- calibration dataset and projection statistics;
- intervention window and strength policy;
- package/Git revision and dirty-state flag;
- seeds and relevant Transformers/PyTorch/CUDA versions.

### 5.3 File formats

- Parquet for tabular records and run measurements.
- Safetensors for dense vectors and tensor shards.
- JSON/YAML for manifests and configurations.
- Avoid new monolithic PyTorch-pickle `.pt` bundles.

Persist reduced per-example activations with grain:

```text
example x layer x hidden dimension
```

Do not persist full token-by-layer activations unless explicitly requested for a
bounded set of layers/spans.

## 6. Python, CLI, and Slurm workflows

### 6.1 Python-first custom work

```python
from actsteer.emotion import ReducedActivations, extract_emotion_vectors

captures = ReducedActivations(
    example_ids=example_ids,
    concepts=concepts,
    layers=reduced_layers,
)
vectors = extract_emotion_vectors(captures, preset="sofroniew_neutral")
```

Novel experiments remain normal Python files in downstream repositories.

### 6.2 Standard CLI work

```bash
actsteer emotion validate-data --config experiments/vectors.yaml
actsteer emotion extract --config experiments/vectors.yaml
actsteer plan experiments/fear-sweep.yaml
actsteer slurm render runs/fear-sweep/plan.json \
  --profile configs/execution/caltech-h100.yaml
sbatch runs/fear-sweep/submit.slurm
actsteer consolidate runs/fear-sweep
```

The CLI parses and validates inputs, then calls the same Python API. It contains
no independent extraction or steering logic.

### 6.3 Execution rules

- One model process per H100/H200 GPU.
- Use Slurm arrays for independent shards.
- Group work units to avoid unnecessary model reloads.
- Give every cell a deterministic ID.
- Write one immutable output per shard/cell; never let array workers append to
  one shared CSV.
- Resume by detecting valid completed cells.
- Consolidation fails loudly on missing, duplicated, incompatible, or corrupted
  shards.
- Render Slurm scripts; never submit without an explicit user `sbatch` action.

## 7. Migration map from `llm_foraging`

### Move or reimplement in ActSteer

| Existing source | ActSteer destination | Treatment |
|---|---|---|
| `src/llm_foraging/gemma2.py` | `core/models/` | Generalize; remove Gemma constants and torch polyfill when unnecessary |
| `src/llm_foraging/dispatch.py` | `core/models/` + config | Replace environment-variable global dispatch with explicit config |
| `src/llm_foraging/emotion_vectors.py` | `emotion/extraction/` + artifacts | Replace `.pt` loader with versioned vector artifact |
| `scripts/setup/extract_emotion_vectors.py` | `core/capture/` + `emotion/extraction/` | Split capture, estimator, transforms, persistence, and CLI |
| steering hook in `scripts/sweep_steering.py` | `core/interventions/` | Reimplement with semantic sites, windows, and strength policies |
| generic token-logit logic | `core/readouts/candidate.py` | Support candidate sequences, not only single-token A/B |
| generic numeric-token logic | `core/readouts/numeric.py` | Generalize digit distribution readout |
| manual sweep/resume patterns | `core/execution/` | Replace append-to-CSV with planned immutable shards |

### Keep in `llm_foraging`

| Existing source | Reason to keep downstream |
|---|---|
| `stimuli_v2.py` and FID prompt builders | Study/task semantics |
| FID configuration grids | Experimental design |
| fear/calm families used as hypotheses | Study-specific construct choices |
| `readout_binary.py` regression and validation gates | FID analysis semantics |
| cumulative-sweep phase logic | Study protocol, built on ActSteer execution primitives |
| danger probe analysis | Study analysis |
| `plots_section3.py`, `plots_section4.py` | Paper figures and statistics |
| notebooks, manuscript, results, and data layout | Reproducibility record for the paper |

### Migration principle

Copy scientific behavior, not legacy structure. Do not preserve hard-coded
`/home/nokada/projects/llm_foraging`, `sys.path` edits, stale path conventions,
or global `LLMF_MODEL_ID` state.

## 8. Implementation work units

Each work unit ends with a usable artifact and a verification gate. A failed
gate stops that unit; it does not trigger opportunistic redesign.

### WU-01 — Repository scaffold and contracts

**Outputs**

- `pyproject.toml`, MIT `LICENSE`, Conda environment, README skeleton.
- Core typed records/config objects.
- Initial architecture and artifact-schema docs.

**Verification**

- `pip install -e .` succeeds in the target Conda environment.
- `python -c "import actsteer"` succeeds.
- Configuration round-trips through YAML/JSON.
- No core module imports `actsteer.emotion`.

### WU-02 — Dataset normalization and prompt rendering

**Outputs**

- `ConceptExample`, validators, record/Parquet/Hugging Face adapters.
- Text and chat-message support.
- Tagged prompt sections and character-to-token span resolution.
- Optional Codrai adapter.

**Verification**

- Invalid mixed text/messages and duplicate IDs fail loudly.
- Tagged spans resolve correctly after native chat templating.
- Dataset fingerprint changes when content or selection changes.
- Core tests do not import Codrai-specific code.

### WU-03 — Model adapters

**Outputs**

- Common Transformers adapter contract.
- Gemma, Llama, and Qwen adapters.
- Base/plain-text and instruct/chat rendering paths.

**Verification**

- Tiny-config CPU tests confirm residual-site resolution and tensor shape.
- Hooks preserve output type and are removed after context exit or failure.
- Padding masks and last-valid-token selection work for mixed-length batches.
- Manual H100 smoke succeeds on one selected model from each family.

### WU-04 — Reduced activation capture and artifact store

**Outputs**

- Batched, attention-mask-aware residual capture.
- Token selectors and reducers.
- Immutable shared artifact store and manifests.
- Safetensors tensor shards plus Parquet example index.

**Verification**

- Reduced capture matches a direct reference calculation on a tiny fixture.
- Interrupted capture resumes without duplicating examples.
- Reordered input does not silently collide with an existing fingerprint.
- No full token tensor is persisted under the reduced-capture default.

### WU-05 — Emotion extraction and calibration

**Outputs**

- `contrastive_mean`, `sofroniew_global`, and `sofroniew_neutral` presets.
- Neutral-PC transforms and L2 normalization.
- Projection-distribution and residual-norm calibration artifacts.

**Verification**

- All saved vectors have unit norm within tolerance.
- `contrastive_mean` equals the direct mean difference on a fixture.
- PC-removed vectors are orthogonal to removed PCs within tolerance.
- A small Gemma run matches the old extractor numerically within documented
  tolerance when configured identically.

### WU-06 — Intervention runtime

**Outputs**

- Single- and multi-layer residual addition.
- Window/span targeting, including encode-release.
- Strength policies and per-run diagnostics.

**Verification**

- Strength zero exactly matches the unhooked logits within tolerance.
- Only selected layers and tokens change on an instrumented tiny model.
- Encode-release removes hooks before decision/response computation.
- Matched total-budget single/multi-layer doses have the declared norm.
- Exceptions never leave hooks installed.

### WU-07 — Readouts and experiment runner

**Outputs**

- Candidate token/sequence, numeric, generation, and activation readouts.
- Typed experiment plans with deterministic cell IDs.
- Immutable shard writer, resume logic, and fail-loud consolidation.

**Verification**

- Candidate-sequence scores match direct teacher-forced log probabilities.
- A/B counterbalancing can be implemented downstream without special core code.
- Duplicate/missing shards make consolidation fail.
- Resuming skips only complete, checksum-valid cells.

### WU-08 — CLI and Caltech Slurm rendering

**Outputs**

- Stable extraction/planning/shard/consolidation commands.
- H100/H200 execution profiles.
- Reviewable array-job script generation.

**Verification**

- CLI and Python API resolve to identical configurations.
- Generated scripts request `--partition=gpu` and typed H100/H200 GRES.
- No command submits a job without an explicit user action.
- A small two-shard run consolidates successfully on Caltech HPC.

### WU-09 — Three-family v0.1 acceptance run

**Outputs**

- Small shared dataset/capture/vector/calibration artifacts for selected Gemma,
  Llama, and Qwen models.
- Baseline, encode-release, response-only, and continuous results on a small
  neutral/emotion and FID fixture.
- `docs/model-support.md` compatibility record.

**Verification**

- All three families complete capture, extraction, steering, and both required
  readout types.
- Resolved token masks are manually inspectable and correct.
- Baseline is unchanged by installing an inactive intervention.
- Output quality does not catastrophically fail at the chosen smoke strength.
- Failures are reported as unsupported capabilities, never silently skipped.

### WU-10 — `llm_foraging` pilot migration

**Outputs**

- A new downstream driver using ActSteer for model load, vectors, intervention,
  and candidate log probabilities.
- Small parity report comparing old and new Gemma behavior.
- No rewrite of existing paper analyses.

**Verification**

- Old results and scripts remain available.
- New driver has no absolute repo path or `sys.path` insertion.
- Representative extraction, steering, and A/B readouts match within declared
  tolerances.
- Study-specific FID logic remains outside ActSteer.

## 9. Contingencies and stop rules

### Pre-authorized contingencies

| Trigger | Action |
|---|---|
| A model family has a different residual-output structure | Add/adjust only that family adapter; do not leak conditionals into core experiment code |
| Native chat template makes a tagged span ambiguous | Save rendered prompt/token map, fail validation, and require an explicit section marker or selector |
| Batched capture exceeds host memory | Reduce batch size or shard count; do not change the scientific reduction |
| Projection-SD calibration is numerically degenerate | Record diagnostics and fall back only when the experiment explicitly selects residual-norm scaling |
| Multi-layer steering overwhelms behavior | Lower total budget; do not silently switch to per-layer scaling |
| An H100 job times out | Resume immutable shards; do not append to or overwrite complete shards |
| Old/new Gemma results differ | Isolate prompt rendering, token selection, site, dtype, and scaling in that order; document any intentional difference |

### Park conditions

Stop and request a design decision rather than improvising if:

- supporting a model requires non-residual or architecture-specific scientific
  semantics rather than a mechanical adapter;
- an artifact schema change would invalidate already produced v0.1 artifacts;
- a dataset lacks the controls required by its selected extraction preset;
- a requested feature expands into attention/MLP intervention, quantization,
  distributed model execution, synthetic generation, or analysis tooling;
- a migration step would overwrite or rewrite existing `llm_foraging` results;
- network access, model licenses, or Hugging Face authentication are unavailable.

## 10. Lean testing standard

Testing should protect scientific correctness without becoming the project.

Required:

- fast unit tests for masks, layer resolution, strength normalization, hook
  cleanup, fingerprints, configuration, and shard consolidation;
- one tiny-model end-to-end capture -> vector -> steering -> readout test;
- one manual H100 smoke script covering Gemma, Llama, and Qwen;
- one-time Gemma migration parity checks.

Not required:

- large-model GitHub Actions;
- scheduled HPC compatibility suites;
- elaborate coverage targets;
- long-term maintenance of the legacy implementation as a second backend.

## 11. Documentation requirements

The public GitHub repository should showcase the workflow without pretending to
be a production platform.

The README must show:

1. local Conda and editable installation;
2. a minimal extraction example with a user-provided dataset;
3. encode-release steering;
4. a standard CLI matrix sweep;
5. Slurm rendering and manual submission;
6. supported models and explicit limitations.

Documentation must explain:

- extraction presets and their scientific differences;
- token/span and intervention-window semantics;
- first-response-token behavior during generation;
- single- versus multi-layer dose semantics;
- artifact provenance and shared-versus-project-owned storage;
- base versus instruction-model prompting;
- what ActSteer intentionally leaves downstream.

## 12. Remaining implementation-time choices

These do not block the architecture plan but must be pinned before WU-03/WU-09:

- exact Hugging Face model IDs and immutable revisions for Gemma, Llama, and
  Qwen;
- supported Python, PyTorch, Transformers, and CUDA versions;
- Caltech group-storage artifact root;
- final small v0.1 dataset subset and prompts;
- numerical tolerances for Gemma migration parity;
- the GitHub organization/account and remote URL.

No other unresolved decision should be delegated to an unattended run.
