# Artifacts

ActSteer separates expensive reusable artifacts from project-owned experiment
outputs. `ArtifactStore` writes reusable inputs to:

```text
<root>/<artifact-type>/<sha256>/
├── manifest.json
└── ... payload files
```

The directory ID hashes the artifact type and complete provenance input. Every
payload has a size and SHA-256 checksum in the manifest. An identical write is
idempotent. Reusing the same provenance with different bytes fails rather than
silently overwriting data.

Callers must include every scientifically material input: model/tokenizer
revision, dataset fingerprint, resolved prompt and token selector, activation
site and layers, estimator/transforms, package or Git revision, and numerical
mode. The store cannot infer omitted provenance.

Dense tensor persistence will use safetensors and tabular indices will use
Parquet. The current store already enforces the write-once directory and
manifest contract; format-specific capture/vector serializers are the next
implementation step.
