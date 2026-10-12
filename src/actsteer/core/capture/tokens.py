"""Declarative token selectors resolved after model-specific tokenization."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol


class TokenSelector(Protocol):
    @property
    def kind(self) -> str: ...


@dataclass(frozen=True)
class AllValidTokens:
    kind: str = "all_valid"


@dataclass(frozen=True)
class LastValidToken:
    kind: str = "last_valid"


@dataclass(frozen=True)
class LastKTokens:
    count: int
    kind: str = "last_k"

    def __post_init__(self) -> None:
        if self.count <= 0:
            raise ValueError("last-k token count must be positive")


@dataclass(frozen=True)
class AfterOffset:
    offset: int
    kind: str = "after_offset"

    def __post_init__(self) -> None:
        if self.offset < 0:
            raise ValueError("token offset must be non-negative")


@dataclass(frozen=True)
class ExplicitTokenSpan:
    start: int
    stop: int
    kind: str = "explicit_span"

    def __post_init__(self) -> None:
        if self.start < 0 or self.stop <= self.start:
            raise ValueError(
                f"token span must satisfy 0 <= start < stop; got ({self.start}, {self.stop})"
            )


@dataclass(frozen=True)
class TaggedSpan:
    name: str
    kind: str = "tagged_span"

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("tagged span name cannot be blank")


def resolve_token_mask(
    selector: TokenSelector,
    attention_mask: Any,
    *,
    tagged_spans: Mapping[str, Sequence[tuple[int, int]]] | None = None,
) -> Any:
    """Resolve a declarative selector to a boolean ``[batch, tokens]`` mask."""

    try:
        import torch
    except ImportError as error:  # pragma: no cover - model environment only
        raise ImportError(
            "token-mask resolution requires the ActSteer model dependencies"
        ) from error

    if attention_mask.ndim != 2:
        raise ValueError("attention_mask must have shape [batch, tokens]")
    valid = attention_mask.to(dtype=torch.bool)
    selected = torch.zeros_like(valid)
    batch_size, sequence_length = valid.shape

    if isinstance(selector, AllValidTokens):
        return valid

    if isinstance(selector, ExplicitTokenSpan):
        if selector.stop > sequence_length:
            raise ValueError(
                f"token span stop {selector.stop} exceeds sequence length {sequence_length}"
            )
        selected[:, selector.start : selector.stop] = True
        return selected & valid

    if isinstance(selector, TaggedSpan):
        if tagged_spans is None or selector.name not in tagged_spans:
            raise ValueError(f"no resolved spans supplied for tag {selector.name!r}")
        spans = tagged_spans[selector.name]
        if len(spans) != batch_size:
            raise ValueError(
                f"tag {selector.name!r} has {len(spans)} spans for batch size {batch_size}"
            )
        for row, (start, stop) in enumerate(spans):
            if start < 0 or stop <= start or stop > sequence_length:
                raise ValueError(f"invalid token span ({start}, {stop}) for row {row}")
            selected[row, start:stop] = True
        result = selected & valid
        if bool((result.sum(dim=1) == 0).any().item()):
            raise ValueError(f"tagged span {selector.name!r} selects no valid tokens")
        return result

    for row in range(batch_size):
        positions = torch.nonzero(valid[row], as_tuple=False).flatten()
        if positions.numel() == 0:
            raise ValueError(f"attention mask row {row} contains no valid tokens")
        if isinstance(selector, LastValidToken):
            positions = positions[-1:]
        elif isinstance(selector, LastKTokens):
            positions = positions[-selector.count :]
        elif isinstance(selector, AfterOffset):
            positions = positions[selector.offset :]
            if positions.numel() == 0:
                raise ValueError(
                    f"offset {selector.offset} selects no tokens in attention mask row {row}"
                )
        else:
            raise TypeError(f"unsupported token selector: {type(selector).__name__}")
        selected[row, positions] = True
    return selected
