"""One-forward reduced residual capture with exception-safe hook cleanup."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from actsteer.core.capture.reducers import hidden_from_output, masked_mean
from actsteer.core.capture.sites import ActivationPoint, ActivationSite
from actsteer.core.models.base import ModelAdapter


def capture_reduced_forward(
    adapter: ModelAdapter,
    model_inputs: Mapping[str, Any],
    *,
    layers: Sequence[int],
    token_mask: Any,
    site: ActivationSite = ActivationSite(),
) -> tuple[Any, dict[int, Any]]:
    """Run one model forward and return CPU float32 ``[B,H]`` captures.

    The reduction happens inside each hook before device-to-host transfer. This
    keeps the default artifact path proportional to examples × hidden size,
    rather than examples × sequence length × hidden size.
    """

    try:
        import torch
    except ImportError as error:  # pragma: no cover - model environment only
        raise ImportError("activation capture requires the ActSteer model dependencies") from error

    resolved = tuple(layers)
    if not resolved or len(set(resolved)) != len(resolved):
        raise ValueError("capture layers must be a non-empty unique sequence")
    captures: dict[int, Any] = {}
    handles: list[Any] = []

    def save(layer: int, hidden: Any) -> None:
        reduced = masked_mean(hidden, token_mask)
        captures[layer] = reduced.detach().to(device="cpu", dtype=torch.float32)

    try:
        for layer in resolved:
            block = adapter.get_block(layer)
            if site.point is ActivationPoint.PRE:
                def pre_hook(_module: Any, inputs: tuple[Any, ...], layer: int = layer) -> None:
                    if not inputs:
                        raise ValueError("transformer block pre-hook received no positional inputs")
                    save(layer, inputs[0])

                handles.append(block.register_forward_pre_hook(pre_hook))
            else:
                def post_hook(
                    _module: Any,
                    _inputs: tuple[Any, ...],
                    output: Any,
                    layer: int = layer,
                ) -> None:
                    save(layer, hidden_from_output(output))

                handles.append(block.register_forward_hook(post_hook))

        with torch.inference_mode():
            output = adapter.model(**dict(model_inputs))
        missing = set(resolved) - set(captures)
        if missing:
            raise RuntimeError(f"capture hooks did not fire for layers {sorted(missing)}")
        return output, captures
    finally:
        for handle in handles:
            handle.remove()
