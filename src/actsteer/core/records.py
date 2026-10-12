"""Normalized records accepted by ActSteer dataset adapters."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ChatMessage(BaseModel):
    """One semantically meaningful chat message before model-specific rendering."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    role: Literal["system", "user", "assistant", "tool"]
    content: str = Field(min_length=1)


class ConceptExample(BaseModel):
    """Model-independent example for activation capture or vector extraction."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    concept: str = Field(min_length=1)
    text: str | None = None
    messages: tuple[ChatMessage, ...] | None = None
    reference: str | None = None
    pair_id: str | None = None
    split: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _require_exactly_one_content_form(self) -> ConceptExample:
        has_text = self.text is not None and bool(self.text.strip())
        has_messages = self.messages is not None and len(self.messages) > 0
        if has_text == has_messages:
            raise ValueError("provide exactly one of non-empty text or messages")
        return self

