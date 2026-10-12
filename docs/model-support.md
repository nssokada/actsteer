# Model support

ActSteer targets decoder-only Hugging Face causal language models whose
configuration declares one of these `model_type` values:

| Family | Accepted `model_type` values | Adapter contract tests | Published checkpoint smoke |
|---|---|---:|---:|
| Gemma | `gemma`, `gemma2`, `gemma3_text` | Passing | Pending |
| Llama | `llama` | Passing | Pending |
| Qwen | `qwen2`, `qwen3` | Passing | Pending |

The adapter uses the model's native chat template when chat input is requested
and adds no system prompt. Base/plain-text input is supported separately.
Models are loaded in BF16, FP16, or FP32 without quantization.

The table records implemented mechanical compatibility, not a claim that every
checkpoint has passed the scientific acceptance workflow. Exact checkpoint IDs
and revisions will be added as reproducible smoke workflows are completed.

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
