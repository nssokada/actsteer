"""Transformers-backed adapters for the v0.1 model families."""

from __future__ import annotations

from typing import Any

from actsteer.core.config import ModelConfig
from actsteer.core.models.base import ModelAdapter, ModelFamily


def _nested_attr(root: Any, path: str) -> Any:
    value = root
    for part in path.split("."):
        value = getattr(value, part)
    return value


class _DecoderOnlyAdapter(ModelAdapter):
    block_paths: tuple[str, ...] = ("model.layers",)

    def _blocks(self) -> Any:
        failures: list[str] = []
        for path in self.block_paths:
            try:
                blocks = _nested_attr(self.model, path)
            except AttributeError:
                failures.append(path)
                continue
            if len(blocks) != self.num_layers:
                raise ValueError(
                    f"{self.family.value} block path {path!r} has {len(blocks)} blocks, "
                    f"but config declares {self.num_layers}"
                )
            return blocks
        raise ValueError(
            f"could not locate {self.family.value} decoder blocks; tried {failures}"
        )

    def get_block(self, layer: int) -> Any:
        if layer < 0 or layer >= self.num_layers:
            raise IndexError(
                f"layer {layer} out of range for model with {self.num_layers} layers"
            )
        return self._blocks()[layer]


class GemmaAdapter(_DecoderOnlyAdapter):
    family = ModelFamily.GEMMA
    block_paths = ("model.layers", "model.language_model.layers")


class LlamaAdapter(_DecoderOnlyAdapter):
    family = ModelFamily.LLAMA


class QwenAdapter(_DecoderOnlyAdapter):
    family = ModelFamily.QWEN


_MODEL_TYPE_ADAPTERS: dict[str, type[ModelAdapter]] = {
    "gemma": GemmaAdapter,
    "gemma2": GemmaAdapter,
    "gemma3_text": GemmaAdapter,
    "llama": LlamaAdapter,
    "qwen2": QwenAdapter,
    "qwen3": QwenAdapter,
}


def adapter_from_objects(model: Any, tokenizer: Any) -> ModelAdapter:
    """Select an explicit family adapter from a loaded model's config."""

    model_type = getattr(model.config, "model_type", None)
    try:
        adapter_type = _MODEL_TYPE_ADAPTERS[model_type if isinstance(model_type, str) else ""]
    except KeyError as error:
        supported = ", ".join(sorted(_MODEL_TYPE_ADAPTERS))
        raise ValueError(
            f"unsupported model_type {model_type!r}; supported types: {supported}"
        ) from error
    return adapter_type(model=model, tokenizer=tokenizer)


def load_model(
    config: ModelConfig, *, device_map: str | None = "auto",
    attn_implementation: str | None = None,
) -> ModelAdapter:
    """Load a configured causal LM with lazy heavy dependencies."""

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:  # pragma: no cover - exercised in GPU environment
        raise ImportError(
            "model loading requires the ActSteer model dependencies"
        ) from error

    dtype = getattr(torch, config.dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        config.id,
        revision=config.tokenizer_revision or config.revision,
        trust_remote_code=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.id,
        revision=config.revision,
        torch_dtype=dtype,
        **({"attn_implementation": attn_implementation} if attn_implementation else {}),
        device_map=device_map,
        trust_remote_code=False,
    )
    model.eval()
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return adapter_from_objects(model, tokenizer)
