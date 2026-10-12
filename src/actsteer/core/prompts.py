"""Semantic prompt sections used for model-specific rendering and span targeting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class PromptSection:
    name: str
    content: str

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("prompt section name cannot be blank")
        if not self.content:
            raise ValueError("prompt section content cannot be empty")


@dataclass(frozen=True)
class RenderedPrompt:
    text: str
    character_spans: dict[str, tuple[int, int]]


def char_span_to_token_span(
    offsets: Sequence[Sequence[int]],
    start: int,
    stop: int,
) -> tuple[int, int]:
    """Map a character interval to its smallest covering token interval."""

    if start < 0 or stop <= start:
        raise ValueError(f"invalid character span ({start}, {stop})")
    selected = [
        index
        for index, (token_start, token_stop) in enumerate(offsets)
        if not (token_start == token_stop == 0)
        and token_stop > start
        and token_start < stop
    ]
    if not selected:
        raise ValueError(f"character span ({start}, {stop}) does not overlap any token")
    return selected[0], selected[-1] + 1


def locate_embedded_prompt(
    rendered_text: str,
    prompt: RenderedPrompt,
) -> dict[str, tuple[int, int]]:
    """Relocate plain-prompt spans after a native chat template wraps the text."""

    first = rendered_text.find(prompt.text)
    if first < 0:
        raise ValueError("prompt payload was changed or removed by the chat template")
    if rendered_text.find(prompt.text, first + 1) >= 0:
        raise ValueError("prompt payload occurs more than once after chat templating")
    return {
        name: (first + start, first + stop)
        for name, (start, stop) in prompt.character_spans.items()
    }


def resolve_token_spans(
    tokenizer: object,
    text: str,
    character_spans: dict[str, tuple[int, int]],
    *,
    add_special_tokens: bool,
) -> tuple[tuple[int, ...], dict[str, tuple[int, int]]]:
    """Tokenize once with offsets and resolve every named character span."""

    encoded = tokenizer(  # type: ignore[operator]
        text,
        add_special_tokens=add_special_tokens,
        return_offsets_mapping=True,
    )
    token_ids = encoded["input_ids"]
    offsets = encoded.get("offset_mapping")
    if offsets is None:
        raise ValueError("tagged spans require a fast tokenizer with offset mappings")
    if token_ids and isinstance(token_ids[0], list):
        if len(token_ids) != 1 or len(offsets) != 1:
            raise ValueError("tagged prompt rendering expects one unbatched input")
        token_ids = token_ids[0]
        offsets = offsets[0]
    spans = {
        name: char_span_to_token_span(offsets, start, stop)
        for name, (start, stop) in character_spans.items()
    }
    return tuple(token_ids), spans


@dataclass(frozen=True)
class PromptSpec:
    sections: tuple[PromptSection, ...]
    separator: str = "\n\n"

    def render_plain(self) -> RenderedPrompt:
        if not self.sections:
            raise ValueError("prompt must contain at least one section")
        duplicate_names = {
            section.name
            for section in self.sections
            if sum(other.name == section.name for other in self.sections) > 1
        }
        if duplicate_names:
            raise ValueError(f"duplicate prompt section names: {sorted(duplicate_names)}")

        chunks: list[str] = []
        spans: dict[str, tuple[int, int]] = {}
        cursor = 0
        for index, section in enumerate(self.sections):
            if index:
                chunks.append(self.separator)
                cursor += len(self.separator)
            start = cursor
            chunks.append(section.content)
            cursor += len(section.content)
            spans[section.name] = (start, cursor)
        return RenderedPrompt(text="".join(chunks), character_spans=spans)
