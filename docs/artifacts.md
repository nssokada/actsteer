# Artifacts

Activation extraction can be expensive, so ActSteer treats reusable results as
immutable artifacts rather than mutable cache files. `ArtifactStore` writes:

```text
<root>/<artifact-type>/<sha256>/
├── manifest.json
└── ... payload files
```

The directory ID hashes the artifact type and complete provenance input. Every
payload has a size and SHA-256 checksum in the manifest. An identical write is
idempotent. Reusing the same provenance with different bytes fails rather than
silently overwriting data.

```python
from actsteer.core.artifacts import ArtifactStore

store = ArtifactStore("artifacts")
ref = store.write(
    "vectors",
    inputs={
        "model": "<model-id-and-revision>",
        "dataset": dataset.fingerprint,
        "method": "sofroniew_neutral",
        "layers": [12, 16, 20],
    },
    files={
        "metadata.json": {"concepts": ["fear", "neutral"]},
        "notes.txt": "Pilot extraction\n",
    },
)
store.verify(ref)
```

Callers must include every scientifically material input: model/tokenizer
revision, dataset fingerprint, resolved prompt and token selector, activation
site and layers, estimator/transforms, package or Git revision, and numerical
mode. The store cannot infer omitted provenance.

Dense tensor persistence will use safetensors and tabular indices will use
Parquet. The current store already enforces the write-once directory and
manifest contract; format-specific capture/vector serializers are the next
implementation step.

## What belongs in provenance

At minimum, record:

- model and tokenizer identifiers and revisions;
- numerical dtype;
- dataset fingerprint and selected examples;
- semantic messages, rendered prompt template, and token selector;
- activation site and resolved layer indices;
- reducer and extraction preset;
- calibration dataset and method;
- intervention window and strength policy;
- package or Git revision; and
- random seeds that affect the result.

Project-specific run outputs do not need to live in the shared artifact store.
Keep behavioral measurements, analysis tables, and figures in the downstream
experiment repository while referencing the immutable artifact ID they used.
