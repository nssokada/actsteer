"""Offline, randomly initialized Transformers fixtures (never checkpoint evidence)."""

from pathlib import Path


def write_fixture(family: str, destination: Path) -> None:
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import (
        AutoModelForCausalLM,
        Gemma2Config,
        LlamaConfig,
        PreTrainedTokenizerFast,
        Qwen2Config,
    )

    torch.manual_seed(1729)
    words = (
        "[PAD] [UNK] [BOS] [EOS] user assistant : . "
        "Read these words alpha beta gamma delta one two three four "
        "Repeat the last word Write first item Count items List in order"
    ).split()
    vocab = {word: i for i, word in enumerate(dict.fromkeys(words))}
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="[PAD]",
        unk_token="[UNK]",
        bos_token="[BOS]",
        eos_token="[EOS]",
    )
    tokenizer.chat_template = (
        "{% for message in messages %}{{ message['role'] + ': ' + message['content'] "
        "+ '\\n' }}{% endfor %}{% if add_generation_prompt %}{{ 'assistant: ' }}{% endif %}"
    )
    config_type = {"gemma": Gemma2Config, "llama": LlamaConfig, "qwen": Qwen2Config}[family]
    config = config_type(
        vocab_size=len(vocab),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=256,
        bos_token_id=2,
        eos_token_id=3,
        pad_token_id=0,
    )
    model = AutoModelForCausalLM.from_config(config, attn_implementation="eager")
    model.save_pretrained(destination)
    tokenizer.save_pretrained(destination)
