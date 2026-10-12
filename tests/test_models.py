from __future__ import annotations

import unittest
from types import SimpleNamespace

from actsteer.core.models import GemmaAdapter, LlamaAdapter, QwenAdapter, adapter_from_objects
from actsteer.core.prompts import PromptSection, PromptSpec
from actsteer.core.records import ChatMessage


class FakeTokenizer:
    chat_template = "fake"

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        prefix = [0] if add_special_tokens else []
        return prefix + [ord(character) for character in text]

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        return_offsets_mapping: bool,
    ) -> dict[str, list]:
        prefix = [0] if add_special_tokens else []
        special_offsets = [(0, 0)] if add_special_tokens else []
        return {
            "input_ids": prefix + [ord(character) for character in text],
            "offset_mapping": special_offsets
            + [(index, index + 1) for index in range(len(text))],
        }

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert not tokenize
        rendered = "\n".join(f"{message['role']}: {message['content']}" for message in messages)
        return rendered + ("\nassistant:" if add_generation_prompt else "")


def fake_model(model_type: str, layers: int = 4) -> SimpleNamespace:
    blocks = [object() for _ in range(layers)]
    return SimpleNamespace(
        config=SimpleNamespace(
            model_type=model_type,
            num_hidden_layers=layers,
            hidden_size=16,
        ),
        model=SimpleNamespace(layers=blocks),
    )


class ModelAdapterTests(unittest.TestCase):
    def test_family_dispatch(self) -> None:
        tokenizer = FakeTokenizer()
        self.assertIsInstance(adapter_from_objects(fake_model("gemma2"), tokenizer), GemmaAdapter)
        self.assertIsInstance(adapter_from_objects(fake_model("llama"), tokenizer), LlamaAdapter)
        self.assertIsInstance(adapter_from_objects(fake_model("qwen3"), tokenizer), QwenAdapter)

    def test_block_access_is_checked(self) -> None:
        adapter = adapter_from_objects(fake_model("llama"), FakeTokenizer())
        self.assertIs(adapter.get_block(2), adapter.model.model.layers[2])
        with self.assertRaises(IndexError):
            adapter.get_block(4)

    def test_native_chat_template_is_used(self) -> None:
        adapter = adapter_from_objects(fake_model("qwen3"), FakeTokenizer())
        rendered = adapter.render_chat((ChatMessage(role="user", content="Hello"),))
        self.assertEqual(rendered.text, "user: Hello\nassistant:")
        self.assertGreater(len(rendered.token_ids), 0)

    def test_tagged_chat_spans_survive_native_template(self) -> None:
        adapter = adapter_from_objects(fake_model("llama"), FakeTokenizer())
        prompt = PromptSpec(
            (
                PromptSection("environment", "A dark forest."),
                PromptSection("question", "What now?"),
            )
        )

        rendered = adapter.render_tagged_chat(prompt)

        start, stop = rendered.token_spans["environment"]
        recovered = "".join(chr(token) for token in rendered.token_ids[start:stop])
        self.assertEqual(recovered, "A dark forest.")

    def test_unsupported_model_fails_loudly(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported model_type"):
            adapter_from_objects(fake_model("unknown"), FakeTokenizer())


if __name__ == "__main__":
    unittest.main()
