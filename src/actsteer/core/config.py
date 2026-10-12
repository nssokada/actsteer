"""Typed, serializable configuration shared by Python and CLI workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class SerializableConfig(BaseModel):
    """Base class with strict validation and YAML round-tripping."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Self:
        source = Path(path)
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"configuration root must be a mapping: {source}")
        return cls.model_validate(payload)

    def to_yaml(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False),
            encoding="utf-8",
        )


class ModelConfig(SerializableConfig):
    """Model identity and numerical execution mode."""

    id: str = Field(min_length=1)
    revision: str | None = None
    tokenizer_revision: str | None = None
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    mode: Literal["auto", "chat", "text"] = "auto"

    @field_validator("id")
    @classmethod
    def _strip_model_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("model id cannot be blank")
        return value


class ArtifactStoreConfig(SerializableConfig):
    """Location of reusable immutable artifacts."""

    root: Path


class ActSteerConfig(SerializableConfig):
    """Small root configuration used while the workflow schemas evolve."""

    schema_version: int = Field(default=1, ge=1)
    name: str = Field(min_length=1)
    model: ModelConfig
    artifacts: ArtifactStoreConfig
    seed: int = Field(default=0, ge=0)

