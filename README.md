# ActSteer

ActSteer is a research toolkit for capturing language-model activations,
extracting steering directions, and applying targeted residual-stream
interventions. It provides a general activation-engineering core and an
emotion-specific framework.

The project is under active development. The architecture and migration plan
is in [`docs/plans/architecture-and-migration.md`](docs/plans/architecture-and-migration.md).

## Scope

ActSteer is designed around:

- Hugging Face causal language models and native PyTorch hooks;
- Gemma, Llama, and Qwen model adapters;
- plain-text and structured chat inputs;
- reduced per-example activation capture;
- composable vector-extraction protocols;
- token-, span-, phase-, and layer-targeted residual steering;
- Python-first custom experiments and CLI-driven standard workflows;
- immutable shared artifacts and project-owned experiment runs;
- transparent Slurm script generation for Caltech HPC.

ActSteer does not own task-specific classification, statistics, plotting, or
scientific interpretation.

## Development installation

```bash
conda env create -f environment.yml
conda activate actsteer
```

For an existing compatible environment:

```bash
python -m pip install -e ".[models,data,dev]"
```

The package is intended to be installed from a local clone or pinned Git
revision. It is not currently published to a package registry.

## Current status

The first implementation milestone includes:

- validated text/chat datasets, deterministic fingerprints, and an optional
  Codrai adapter;
- Gemma, Llama, and Qwen Transformers adapters with native chat templates;
- named prompt-section to token-span resolution;
- attention-mask-aware reduced residual capture;
- `contrastive_mean`, `sofroniew_global`, and `sofroniew_neutral` extraction;
- projection-SD and residual-norm calibration;
- matched total-budget and legacy per-layer steering strengths;
- encode-release, response-only, and continuous residual interventions;
- immutable content-addressed artifacts; and
- a small validation CLI.

The implementation is locally tested without model weights. Tiny-model and
H100/H200 acceptance runs, readouts, experiment planning, and Slurm rendering
remain in progress; see the migration plan for the exact gates.

## Minimal Python workflow

Normalize any records into the package schema:

```python
from actsteer.core import ConceptDataset

dataset = ConceptDataset.from_tabular_records(
    rows,
    id_column="id",
    concept_column="emotion",
    text_column="story",
)
print(dataset.fingerprint)
```

Extract emotion directions from reduced per-example activations. Layer keys are
model block indices and each array has shape `[examples, hidden]`:

```python
from actsteer.emotion import ReducedActivations, extract_emotion_vectors

captures = ReducedActivations(
    example_ids=tuple(row.id for row in dataset),
    concepts=tuple(row.concept for row in dataset),
    layers={20: layer20_array, 24: layer24_array},
)
result = extract_emotion_vectors(
    captures,
    preset="sofroniew_neutral",
    neutral_concept="neutral",
)
fear_at_20 = result.vectors[20]["fear"]
```

Standard steering uses a model-calibrated total dose that does not grow merely
because more layers are selected:

```python
from actsteer.core.interventions import (
    InterventionWindow,
    SteeringController,
    allocate_amplitudes,
    build_torch_deltas,
)

amplitudes = allocate_amplitudes(1.0, calibration.scales)
deltas = build_torch_deltas(vectors, amplitudes, device=model.device)
controller = SteeringController(
    adapter,
    deltas,
    window=InterventionWindow.ENCODE_RELEASE,
)

with controller.active(environment_token_mask):
    generated = adapter.model.generate(**model_inputs)
```

The CLI is intentionally limited to stable workflows and calls these same
Python APIs:

```bash
actsteer config validate examples/configs/smoke.yaml
actsteer data validate examples/data/emotions.jsonl
```
