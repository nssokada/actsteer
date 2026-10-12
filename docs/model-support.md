# Model support

ActSteer v0.1 targets decoder-only Hugging Face causal language models whose
configuration declares one of these `model_type` values:

| Family | Accepted model types | Local contract tests | H100/H200 smoke |
|---|---|---:|---:|
| Gemma | `gemma`, `gemma2`, `gemma3_text` | yes | pending |
| Llama | `llama` | yes | pending |
| Qwen | `qwen2`, `qwen3` | yes | pending |

The adapter uses the model's native chat template when chat input is requested
and adds no system prompt. Base/plain-text input is supported separately.
Models are loaded in BF16, FP16, or FP32 without quantization. The first public
acceptance target is one instruction-tuned model around 8B parameters from each
family on a single Caltech H100 or H200.

The table records implemented mechanical compatibility, not a claim that every
checkpoint has passed the scientific acceptance workflow. Exact checkpoint IDs
and revisions will be pinned before the HPC smoke run.
