"""Exception-safe residual-stream steering controllers."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

from actsteer.core.capture.reducers import hidden_from_output, replace_hidden
from actsteer.core.interventions.windows import InterventionWindow
from actsteer.core.models.base import ModelAdapter


class SteeringController:
    """Install one post-block residual addition per configured layer."""

    def __init__(
        self,
        adapter: ModelAdapter,
        deltas: Mapping[int, Any],
        *,
        window: InterventionWindow = InterventionWindow.ENCODE_RELEASE,
    ) -> None:
        if not deltas:
            raise ValueError("at least one layer delta is required")
        self.adapter = adapter
        self.deltas = dict(deltas)
        self.window = window
        self._prompt_mask: Any | None = None
        self._handles: list[Any] = []
        self._in_prefill = True

    @property
    def installed(self) -> bool:
        return bool(self._handles)

    def set_prompt_mask(self, mask: Any) -> None:
        if mask.ndim not in (2, 3):
            raise ValueError("prompt mask must have shape [B,T] or [B,T,1]")
        if mask.ndim == 3 and mask.shape[-1] != 1:
            raise ValueError("three-dimensional prompt mask must end in singleton dimension")
        self._prompt_mask = mask

    def clear_prompt_mask(self) -> None:
        self._prompt_mask = None

    def _mask_for(self, hidden: Any) -> Any | None:
        if self._prompt_mask is None:
            return None
        mask = self._prompt_mask
        if mask.ndim == 2:
            mask = mask[..., None]
        if self._in_prefill:
            if self.window is InterventionWindow.RESPONSE_ONLY:
                return None
            if hidden.shape[1] != mask.shape[1]:
                raise ValueError(
                    "prefill hidden sequence length does not match the configured prompt mask"
                )
            if hidden.shape[0] != mask.shape[0]:
                raise ValueError("prompt mask batch size does not match hidden state")
            return mask
        if self.window is InterventionWindow.ENCODE_RELEASE:
            return None
        return hidden.new_ones((*hidden.shape[:2], 1))

    def _hook(self, layer: int):
        delta = self.deltas[layer]

        def apply(_module: Any, _inputs: tuple[Any, ...], output: Any) -> Any:
            hidden = hidden_from_output(output)
            mask = self._mask_for(hidden)
            if mask is None:
                return output
            typed_mask = mask.to(device=hidden.device, dtype=hidden.dtype)
            typed_delta = delta.to(device=hidden.device, dtype=hidden.dtype)
            steered = hidden + typed_mask * typed_delta.reshape(1, 1, -1)
            return replace_hidden(output, steered)

        return apply

    def _track_forward_phase(
        self,
        _module: Any,
        _args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        """Distinguish the initial prompt pass from cached decode passes."""

        cache_position = kwargs.get("cache_position")
        if cache_position is not None and cache_position.numel() > 0:
            self._in_prefill = int(cache_position.reshape(-1)[0].item()) == 0
            return
        cache = kwargs.get("past_key_values")
        # Transformers may supply an allocated but empty Cache on the first
        # forward even when cache_position is not exposed at this boundary.
        if cache is not None and hasattr(cache, "get_seq_length"):
            self._in_prefill = int(cache.get_seq_length()) == 0
        else:
            self._in_prefill = cache is None

    def install(self) -> None:
        if self.installed:
            raise RuntimeError("steering controller is already installed")
        try:
            self._handles.append(
                self.adapter.model.register_forward_pre_hook(
                    self._track_forward_phase,
                    with_kwargs=True,
                )
            )
            for layer in self.deltas:
                handle = self.adapter.get_block(layer).register_forward_hook(self._hook(layer))
                self._handles.append(handle)
        except Exception:
            self.remove()
            raise

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self.clear_prompt_mask()
        self._in_prefill = True

    @contextmanager
    def active(self, prompt_mask: Any) -> Iterator[SteeringController]:
        self.set_prompt_mask(prompt_mask)
        self.install()
        try:
            yield self
        finally:
            self.remove()
