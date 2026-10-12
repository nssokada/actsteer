from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from actsteer.core.interventions import InterventionWindow, SteeringController
from actsteer.core.models import LlamaAdapter


class FakeTensor:
    def __init__(self, values) -> None:
        self.values = np.asarray(values, dtype=np.float32)
        self.device = "cpu"
        self.dtype = self.values.dtype

    @property
    def shape(self):
        return self.values.shape

    @property
    def ndim(self):
        return self.values.ndim

    def to(self, **_kwargs):
        return self

    def reshape(self, *shape):
        return FakeTensor(self.values.reshape(*shape))

    def new_ones(self, shape):
        return FakeTensor(np.ones(shape, dtype=np.float32))

    def __getitem__(self, item):
        return FakeTensor(self.values[item])

    def __mul__(self, other):
        values = other.values if isinstance(other, FakeTensor) else other
        return FakeTensor(self.values * values)

    def __add__(self, other):
        values = other.values if isinstance(other, FakeTensor) else other
        return FakeTensor(self.values + values)


class FakeHandle:
    def __init__(self, hooks: list, hook) -> None:
        self.hooks = hooks
        self.hook = hook

    def remove(self) -> None:
        if self.hook in self.hooks:
            self.hooks.remove(self.hook)


class FakeBlock:
    def __init__(self) -> None:
        self.hooks = []

    def register_forward_hook(self, hook):
        self.hooks.append(hook)
        return FakeHandle(self.hooks, hook)

    def forward(self, hidden):
        output = (hidden, "cache")
        for hook in tuple(self.hooks):
            replacement = hook(self, (hidden,), output)
            if replacement is not None:
                output = replacement
        return output


class FakeModel:
    def __init__(self) -> None:
        self.config = SimpleNamespace(
            model_type="llama",
            num_hidden_layers=1,
            hidden_size=2,
        )
        self.model = SimpleNamespace(layers=[FakeBlock()])
        self.pre_hooks = []

    def register_forward_pre_hook(self, hook, *, with_kwargs):
        self.pre_hooks.append(hook)
        return FakeHandle(self.pre_hooks, hook)

    def forward(self, hidden, **kwargs):
        for hook in tuple(self.pre_hooks):
            hook(self, (hidden,), kwargs)
        return self.model.layers[0].forward(hidden)


class FakeTokenizer:
    chat_template = None


class SteeringControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = FakeModel()
        self.adapter = LlamaAdapter(self.model, FakeTokenizer())
        self.delta = FakeTensor([2.0, 0.0])
        self.mask = FakeTensor([[1.0, 0.0]])

    def test_encode_release_changes_only_selected_prefill_token(self) -> None:
        controller = SteeringController(
            self.adapter,
            {0: self.delta},
            window=InterventionWindow.ENCODE_RELEASE,
        )
        prompt = FakeTensor(np.zeros((1, 2, 2)))
        with controller.active(self.mask):
            output = self.model.forward(prompt)[0]
            np.testing.assert_allclose(output.values[0, 0], [2.0, 0.0])
            np.testing.assert_allclose(output.values[0, 1], [0.0, 0.0])

            decode = FakeTensor(np.zeros((1, 1, 2)))
            output = self.model.forward(decode, past_key_values=object())[0]
            np.testing.assert_allclose(output.values, 0.0)

        self.assertEqual(self.model.pre_hooks, [])
        self.assertEqual(self.model.model.layers[0].hooks, [])

    def test_response_only_skips_prefill_and_changes_decode(self) -> None:
        controller = SteeringController(
            self.adapter,
            {0: self.delta},
            window=InterventionWindow.RESPONSE_ONLY,
        )
        with controller.active(self.mask):
            prompt = self.model.forward(FakeTensor(np.zeros((1, 2, 2))))[0]
            np.testing.assert_allclose(prompt.values, 0.0)
            decode = self.model.forward(
                FakeTensor(np.zeros((1, 1, 2))),
                past_key_values=object(),
            )[0]
            np.testing.assert_allclose(decode.values[0, 0], [2.0, 0.0])

    def test_cleanup_occurs_after_exception(self) -> None:
        controller = SteeringController(self.adapter, {0: self.delta})
        with self.assertRaisesRegex(RuntimeError, "boom"):
            with controller.active(self.mask):
                raise RuntimeError("boom")
        self.assertFalse(controller.installed)
        self.assertEqual(self.model.model.layers[0].hooks, [])


if __name__ == "__main__":
    unittest.main()
