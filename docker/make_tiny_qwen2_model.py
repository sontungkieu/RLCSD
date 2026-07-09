#!/usr/bin/env python3
"""Create a small random Qwen2 causal LM for pipeline smoke tests."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="Output model directory")
    parser.add_argument(
        "--tokenizer-source",
        default="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
        help="Small public Qwen2 tokenizer source",
    )
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_source, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if not tokenizer.chat_template:
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{{ message['role'] + ': ' + message['content'] + '\\n' }}"
            "{% endfor %}"
            "{% if add_generation_prompt %}assistant: {% endif %}"
        )
    tokenizer.save_pretrained(out)

    config = Qwen2Config(
        vocab_size=len(tokenizer),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
        rms_norm_eps=1e-6,
        rope_theta=1_000_000.0,
        attention_dropout=0.0,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        tie_word_embeddings=True,
        torch_dtype="bfloat16",
    )
    model = Qwen2ForCausalLM(config).to(dtype=torch.bfloat16)
    model.save_pretrained(out, safe_serialization=True)
    print(f"saved tiny qwen2 model to {out}")


if __name__ == "__main__":
    main()
