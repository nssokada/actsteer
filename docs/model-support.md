# Model support

ActSteer targets decoder-only Hugging Face causal language models whose
configuration declares one of these `model_type` values:

| Family | Accepted `model_type` values | Adapter contract tests |
|---|---|---|
| Gemma | `gemma`, `gemma2`, `gemma3_text` | Passing |
| Llama | `llama` | Passing |
| Qwen | `qwen2`, `qwen3` | Passing |

The adapter uses the model's native chat template when chat input is requested
and adds no system prompt. Base/plain-text input is supported separately.
Models are loaded in BF16, FP16, or FP32 without quantization.

The table records implemented adapter compatibility. Real-checkpoint evidence
is limited to the following instruction-tuned checkpoints, tested on 2026-10-12
at clean source SHA **`a960c22107d720cfc515e00b45c126c184135559`**:

| Tested checkpoint | Model revision = tokenizer revision | Instrumentation checks |
|---|---|---|
| `google/gemma-2-9b-it` | `11c9b309abf73637e4b6f9a3fa1e92e615547819` | 20/20 pass |
| `meta-llama/Llama-3.1-8B-Instruct` | `0e9e39f249a16976918f6564b8830bc894c89659` | 20/20 pass |
| `Qwen/Qwen2.5-7B-Instruct` | `a09a35458c702b33eeacc393d103063234e8bc28` | 20/20 pass |

All three ran on **NVIDIA H100 80GB HBM3**, unquantized BF16, eager attention,
deterministic algorithms, and TF32 disabled. Aggregate acceptance was
`validated`. Exact no-op tolerances were `atol=rtol=0`. The
[validation record](validation/2026-10-12-a960c221.md) contains environment
versions, memory diagnostics, evidence checksums, and reproduction commands.
Keep the tested source SHA as the experimental pin even when reading newer docs.

Six offline random-model fixtures (Gemma 2, Llama, Qwen 2, each in chat and text
modes) also passed. These local fixtures are separate from production checkpoint
validation. Other family variants, other checkpoints or revisions, base-model
weights, other precision modes, and H200 hardware were not covered by this real
checkpoint run. The harness establishes mechanical instrumentation behavior,
not scientific effects or behavioral performance.

## Expected model structure

A supported checkpoint must expose:

- a positive `num_hidden_layers` and `hidden_size` in its configuration;
- an ordered decoder-block collection supported by its family adapter;
- block inputs or outputs whose first tensor has shape `[batch, tokens, hidden]`;
  and
- a fast tokenizer when named character spans must be mapped to tokens.

Unsupported structures fail explicitly instead of silently selecting a nearby
module.

## Instruction and base models

For instruction-tuned checkpoints, use structured messages and the tokenizer's
native chat template. For base models, use literal text. ActSteer stores these
as different input modes because adding chat control tokens can materially
change both extracted activations and intervention effects.

## Hardware and precision

The intended v0.1 path is one model process on one accelerator with sufficient
memory for an approximately 8B-parameter checkpoint in BF16 or FP16. CPU and
tiny-model fixtures are useful for contract tests but are not representative
performance benchmarks. Quantized loading and multi-node execution are not yet
supported.
