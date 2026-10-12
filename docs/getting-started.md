# Getting started

This guide walks through ActSteer's current Python APIs. It uses a single
prompt for clarity; real extraction runs should batch many examples and keep a
separate evaluation dataset.

## Install from a clone

```bash
git clone https://github.com/nssokada/actsteer.git
cd actsteer
conda env create -f environment.yml
conda activate actsteer
```

Or install into an existing Python 3.11 environment:

```bash
python -m pip install -e ".[models,data]"
```

Some Hugging Face checkpoints require accepting their license and authenticating
with Hugging Face before they can be downloaded.

## 1. Represent a concept dataset

ActSteer's core schema is not emotion-specific. Each example has an ID, a
concept label, and either plain text or structured messages.

```python
from actsteer.core import ConceptDataset

dataset = ConceptDataset.from_records(
    [
        {"id": "fear-1", "concept": "fear", "text": "A predator is nearby."},
        {"id": "neutral-1", "concept": "neutral", "text": "A lamp is on a desk."},
    ]
)

print(dataset.summary)
print(dataset.fingerprint)
```

For custom tables, map your own column names instead of rewriting the source
data:

```python
dataset = ConceptDataset.from_tabular_records(
    rows,
    id_column="example_id",
    concept_column="emotion",
    text_column="story",
    metadata_columns=("source", "topic"),
)
```

## 2. Load a supported model

```python
from actsteer.core.config import ModelConfig
from actsteer.core.models import load_model

adapter = load_model(
    ModelConfig(
        id="<hugging-face-model-id>",
        dtype="bfloat16",
        mode="chat",
    )
)
```

Model loading is explicit: ActSteer does not insert a system message or change
the checkpoint's native chat template.

## 3. Render and target a prompt section

Named sections make the scientific target explicit. This example captures only
the environment description, not the question.

```python
import torch

from actsteer.core.capture import TaggedSpan, resolve_token_mask
from actsteer.core.prompts import PromptSection, PromptSpec

prompt = PromptSpec(
    sections=(
        PromptSection("environment", "A predator is approaching through the trees."),
        PromptSection("question", "What should you do?"),
    )
)
rendered = adapter.render_tagged_chat(prompt)

device = next(adapter.model.parameters()).device
input_ids = torch.tensor([rendered.token_ids], device=device)
attention_mask = torch.ones_like(input_ids)
environment_mask = resolve_token_mask(
    TaggedSpan("environment"),
    attention_mask,
    tagged_spans={"environment": [rendered.token_spans["environment"]]},
)
```

## 4. Capture reduced residual activations

The default capture path reduces token activations on the model device and
returns one vector per example and layer.

```python
from actsteer.core.capture import capture_reduced_forward

_, captured = capture_reduced_forward(
    adapter,
    {"input_ids": input_ids, "attention_mask": attention_mask},
    layers=(12, 16, 20),
    token_mask=environment_mask,
)

layer_16_vector = captured[16]  # shape: [batch, hidden]
```

## 5. Extract emotion directions

After collecting aligned examples, assemble the reduced matrices and choose an
explicit extraction preset:

```python
from actsteer.emotion import ReducedActivations, extract_emotion_vectors

captures = ReducedActivations(
    example_ids=example_ids,
    concepts=concept_labels,
    layers=reduced_layer_arrays,
)
directions = extract_emotion_vectors(
    captures,
    preset="sofroniew_neutral",
    neutral_concept="neutral",
)

fear_vectors = {
    layer: layer_vectors["fear"]
    for layer, layer_vectors in directions.vectors.items()
}
```

See [Concepts](concepts.md) for the differences between extraction presets.

## 6. Calibrate and steer

Projection-SD calibration expresses strength relative to the model's observed
activation distribution. Total-budget allocation prevents the nominal dose
from growing just because more layers are selected.

```python
from actsteer.core.interventions import (
    InterventionWindow,
    SteeringController,
    allocate_amplitudes,
    build_torch_deltas,
)
from actsteer.emotion import calibrate_layers

calibration = calibrate_layers(
    neutral_layer_arrays,
    method="projection_sd",
    vectors=fear_vectors,
)
amplitudes = allocate_amplitudes(1.0, calibration.scales)
deltas = build_torch_deltas(
    fear_vectors,
    amplitudes,
    device=device,
    dtype=next(adapter.model.parameters()).dtype,
)

controller = SteeringController(
    adapter,
    deltas,
    window=InterventionWindow.ENCODE_RELEASE,
)
with controller.active(environment_mask):
    generated_ids = adapter.model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=64,
    )
```

Encode-release applies the intervention while the selected prompt tokens are
encoded and removes it during decoding. Response-only and continuous windows
are available for comparison experiments.

## Validate local inputs

```bash
actsteer config validate examples/configs/smoke.yaml
actsteer data validate examples/data/emotions.jsonl
```

The CLI currently validates stable inputs. Standard extraction, sweep, and
cluster-scheduler commands are listed on the [roadmap](roadmap.md).
