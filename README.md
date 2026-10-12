# ActSteer

ActSteer is a Python research toolkit for extracting directions from language
model activations and applying controlled residual-stream interventions.

It is designed for experiments that need to answer questions such as:

- Where is a concept represented across model depth?
- How does a direction extracted from one dataset generalize to another?
- What changes when an intervention targets prompt encoding, response
  generation, or both?
- How can steering strength be compared across layers and model families?

ActSteer provides the instrumentation and reproducibility layer. Your project
retains control of prompts, behavioral tasks, evaluation, statistics, and
scientific interpretation.

> **Status:** early alpha. The Python APIs described below are usable, but the
> configuration and artifact schemas may change before v0.1. Pin a commit for
> experiments that need exact reproducibility.

## Features

- Hugging Face Transformers with native PyTorch hooks.
- Gemma, Llama, and Qwen decoder-only model adapters.
- Plain-text and model-native chat inputs without implicit system prompts.
- Named prompt sections that resolve to tokenizer-aware spans.
- Attention-mask-aware residual capture with on-device token reduction.
- Absolute and relative layer selection.
- Contrastive-mean and neutral-PC-removed direction extraction.
- Model- and layer-specific strength calibration.
- Single- and multi-layer steering with matched total-dose semantics.
- Encode-release, response-only, and continuous intervention windows.
- Deterministic dataset fingerprints and immutable local artifacts.
- Generic datasets plus an optional Codrai adapter for emotion research.

ActSteer intentionally does not provide built-in behavioral classifiers,
statistical tests, plots, or LLM judges.

## Installation

Clone the repository and create the included environment:

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

Development tools are available through the `dev` extra:

```bash
python -m pip install -e ".[models,data,dev]"
```

ActSteer is installed from a clone or a pinned Git revision; it is not currently
published to a package registry.

## Quick example

Create a model-independent dataset from your own records:

```python
from actsteer.core import ConceptDataset

dataset = ConceptDataset.from_tabular_records(
    rows,
    id_column="id",
    concept_column="emotion",
    text_column="story",
)
print(dataset.summary)
```

Once reduced activations have been captured, extract one unit direction per
concept and layer:

```python
from actsteer.emotion import ReducedActivations, extract_emotion_vectors

captures = ReducedActivations(
    example_ids=tuple(example.id for example in dataset),
    concepts=tuple(example.concept for example in dataset),
    layers={12: layer_12, 16: layer_16, 20: layer_20},
)
directions = extract_emotion_vectors(
    captures,
    preset="sofroniew_neutral",
    neutral_concept="neutral",
)
```

See the [getting-started guide](docs/getting-started.md) for model loading,
token-span targeting, activation capture, calibration, and steering.

## Extraction presets

| Preset | Center | Remove neutral PCs | Normalize |
|---|---|---:|---:|
| `contrastive_mean` | Neutral mean | No | L2 |
| `sofroniew_global` | Mean of selected concept means | Yes | L2 |
| `sofroniew_neutral` | Neutral mean | Yes | L2 |

The preset is part of the scientific method, not just a software option. See
[Core concepts](docs/concepts.md) for calibration, multi-layer dose, and
intervention-window semantics.

## Command line

The CLI currently covers stable validation workflows:

```bash
actsteer config validate examples/configs/smoke.yaml
actsteer data validate examples/data/emotions.jsonl
```

Extraction matrices, readouts, and scheduler integration remain on the
[public roadmap](docs/roadmap.md). Custom experiments should use the Python API.

## Documentation

- [Getting started](docs/getting-started.md)
- [Core concepts](docs/concepts.md)
- [Architecture](docs/architecture.md)
- [Artifacts and provenance](docs/artifacts.md)
- [Model support](docs/model-support.md)
- [Instrumentation validation and HPC commands](docs/validation.md)
- [Roadmap](docs/roadmap.md)

## Development

Run the lightweight test suite without downloading model weights:

```bash
python -m unittest discover -s tests -v
```

For offline tiny-model smoke checks and the separate real-checkpoint gate, see
[Instrumentation validation](docs/validation.md). A fixture pass does not validate
a production checkpoint.

## License

ActSteer is available under the [MIT License](LICENSE).
