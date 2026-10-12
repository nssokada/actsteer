# Roadmap

ActSteer is an early-stage research toolkit. This roadmap distinguishes usable
features from planned work so experiments do not accidentally depend on an
unfinished interface.

## Available now

- Validated plain-text and chat datasets with deterministic fingerprints.
- Generic JSONL, table-record, and Parquet ingestion.
- An optional adapter for Codrai emotion-expression stories.
- Hugging Face adapters for Gemma, Llama, and Qwen decoder-only models.
- Native chat-template rendering with named character-to-token spans.
- Residual-stream layer and token selection.
- Reduced, attention-mask-aware activation capture.
- Contrastive-mean and Sofroniew-style emotion-vector extraction.
- Projection-standard-deviation and residual-norm calibration.
- Single- and multi-layer steering with explicit dose semantics.
- Encode-release, response-only, and continuous intervention windows.
- Immutable, checksum-verified local artifact directories.
- Configuration and normalized-dataset validation commands.

## Next milestones

1. Add safetensors and Parquet serializers for capture, vector, and calibration
   artifacts.
2. Add candidate-sequence, numeric-token, and raw-generation readouts.
3. Add typed experiment matrices, immutable shards, resume logic, and
   fail-loud consolidation.
4. Add scheduler-neutral job-script rendering, with Slurm as the first
   supported backend.
5. Publish small reproducible smoke workflows for one instruction-tuned Gemma,
   Llama, and Qwen checkpoint.
6. Stabilize the configuration schema and document compatibility guarantees.

## Deliberately out of scope for v0.1

- Attention-head, MLP, and neuron-level interventions.
- Low-bit quantization.
- Distributed or multi-node model execution.
- Built-in behavioral classification, statistical analysis, or plotting.
- Synthetic emotion-dataset generation.
- Hosted experiment tracking.
- Package-registry publication.

Ideas and bug reports are welcome through GitHub issues. Because the package is
pre-release, pin a commit when using it for a long-running study.
