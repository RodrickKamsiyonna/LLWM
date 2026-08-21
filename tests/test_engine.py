"""
Test Engine class. Example run:

python -m pytest tests/test_engine.py -v
"""

import torch
from nanochat.engine import Engine
from dataclasses import dataclass


# -----------------------------------------------------------------------------
# Mock classes for testing Engine without loading a real model

@dataclass
class MockConfig:
    """Minimal config for Engine tests."""
    sequence_len: int = 128


class MockModel:
    """
    Mock model that simulates plan_and_generate for testing Engine.
    """
    def __init__(self, vocab_size=262):
        self.vocab_size = vocab_size
        self.config = MockConfig()
        self._device = torch.device("cpu")

    def get_device(self):
        return self._device

    def plan_and_generate(self, prompt_ids, goal_ids, num_steps):
        # Generate dummy token ids sequentially
        return [(i % (self.vocab_size - 10)) + 1 for i in range(num_steps)]


class ByteTokenizer:
    """
    Simple byte-level tokenizer for testing.
    Tokens 0-255 are raw bytes, 256+ are special tokens.
    """
    def __init__(self):
        # Special tokens start at 256
        self._special_tokens = {
            "<|python_start|>": 256,
            "<|python_end|>": 257,
            "<|output_start|>": 258,
            "<|output_end|>": 259,
            "<|assistant_end|>": 260,
            "<|bos|>": 261,
        }
        self._bos = 261

    def encode_special(self, s):
        return self._special_tokens[s]

    def get_bos_token_id(self):
        return self._bos

    def encode(self, s, prepend=None):
        tokens = list(s.encode("utf-8"))  # bytes 0-255
        if prepend is not None:
            tokens = [prepend] + tokens
        return tokens

    def decode(self, tokens):
        # Filter out special tokens before decoding
        byte_tokens = [t for t in tokens if t < 256]
        return bytes(byte_tokens).decode("utf-8", errors="replace")

def test_max_tokens_respected():
    """Generation stops at max_tokens limit."""
    model = MockModel()
    engine = Engine(model, ByteTokenizer())
    prompt = [261, 72, 101, 108, 108, 111]

    for max_tokens in [1, 4, 16, 64]:
        results, _ = engine.generate_batch(prompt, max_tokens=max_tokens)
        num_generated_tokens = len(results[0]) - len(prompt)
        assert num_generated_tokens <= max_tokens, f"Generated {num_generated_tokens} tokens, expected max_tokens={max_tokens} or less."


def test_num_samples_count():
    """num_samples=N produces exactly N sequences."""
    model = MockModel()
    engine = Engine(model, ByteTokenizer())
    prompt = [261, 72, 101, 108, 108, 111]

    for num_samples in [1, 4, 16, 64]:
        results, _ = engine.generate_batch(prompt, num_samples=num_samples, max_tokens=3)
        assert len(results) == num_samples, f"Expected {num_samples} sequences from {num_samples} samples, got {len(results)}"
