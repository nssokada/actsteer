"""Abstract model interface exposed to capture and intervention code."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Sequence

from actsteer.core.records import ChatMessage
from actsteer.core.prompts import PromptSpec, locate_embedded_prompt, resolve_token_spans


class ModelFamily(str, Enum):
    GEMMA = "gemma"
    LLAMA = "llama"
    QWEN = "qwen"


@dataclass(frozen=True)
class ModelCapabilities:
    text: bool = True
    chat: bool = False
    residual_pre: bool = True
    residual_post: bool = True


@dataclass(frozen=True)
class RenderedModelInput:
    text: str
    token_ids: tuple[int, ...]
    token_spans: dict[str, tuple[int, int]] = field(default_factory=dict)


class ModelAdapter(ABC):
    """Semantic access to one loaded causal language model."""

    family: ModelFamily

    def __init__(self, model: Any, tokenizer: Any) -> None:
        self.model = model
        self.tokenizer = tokenizer

    @property
    def num_layers(self) -> int:
        value = getattr(self.model.config, "num_hidden_layers", None)
        if not isinstance(value, int) or value <= 0:
            raise ValueError("model config does not declare a positive num_hidden_layers")
        return value

    @property
    def hidden_size(self) -> int:
        value = getattr(self.model.config, "hidden_size", None)
        if not isinstance(value, int) or value <= 0:
            raise ValueError("model config does not declare a positive hidden_size")
        return value

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(chat=bool(getattr(self.tokenizer, "chat_template", None)))

    @abstractmethod
    def get_block(self, layer: int) -> Any:
        """Return the transformer block at an already resolved layer index."""

    def render_text(self, text: str) -> RenderedModelInput:
        if not text:
            raise ValueError("text input cannot be empty")
        token_ids = self.tokenizer.encode(text, add_special_tokens=True)
        return RenderedModelInput(text=text, token_ids=tuple(token_ids))

    def render_chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        add_generation_prompt: bool = True,
    ) -> RenderedModelInput:
        if not messages:
            raise ValueError("chat input cannot be empty")
        if not self.capabilities.chat:
            raise ValueError(f"{self.family.value} tokenizer has no chat template")
        payload = [message.model_dump() for message in messages]
        text = self.tokenizer.apply_chat_template(
            payload,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        return RenderedModelInput(text=text, token_ids=tuple(token_ids))

    def render_tagged_text(self, prompt: PromptSpec) -> RenderedModelInput:
        """Render plain text and retain named section boundaries in token space."""

        rendered = prompt.render_plain()
        token_ids, token_spans = resolve_token_spans(
            self.tokenizer,
            rendered.text,
            rendered.character_spans,
            add_special_tokens=True,
        )
        return RenderedModelInput(
            text=rendered.text,
            token_ids=token_ids,
            token_spans=token_spans,
        )

    def render_tagged_chat(
        self,
        prompt: PromptSpec,
        *,
        role: str = "user",
        add_generation_prompt: bool = True,
    ) -> RenderedModelInput:
        """Wrap tagged content in the tokenizer's native chat template."""

        rendered = prompt.render_plain()
        wrapped = self.render_chat(
            (ChatMessage(role=role, content=rendered.text),),
            add_generation_prompt=add_generation_prompt,
        )
        relocated = locate_embedded_prompt(wrapped.text, rendered)
        token_ids, token_spans = resolve_token_spans(
            self.tokenizer,
            wrapped.text,
            relocated,
            add_special_tokens=False,
        )
        if token_ids != wrapped.token_ids:
            raise ValueError("offset tokenization disagrees with native chat rendering")
        return RenderedModelInput(
            text=wrapped.text,
            token_ids=token_ids,
            token_spans=token_spans,
        )
