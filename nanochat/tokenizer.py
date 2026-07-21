"""
Tokenizer (rewrite): now a thin nanochat-shaped wrapper around Qwen1.5's own
HF tokenizer, instead of training our own rustbpe/tiktoken vocab from scratch.

Why: nanochat.gpt.GPT now wraps a frozen Qwen1.5 backbone (see nanochat/gpt.py). Its
input embedding table is indexed by Qwen1.5's vocabulary, so token ids produced here
have to be Qwen1.5's token ids, not our old custom 32768-token BPE vocab.

How the public API stays the same: RustBPETokenizer's methods (encode, decode,
encode_special, render_conversation, visualize_tokenization, render_for_completion,
get_bos_token_id, get_vocab_size, get_special_tokens, id_to_token,
decode_single_token_bytes, __call__, save) are UNCHANGED below - they were all written
against tiktoken.Encoding's interface, and self.enc is now a small adapter
(_HFTiktokenAdapter) that presents that same interface on top of a real HF tokenizer.
So the only things that actually changed are: what self.enc IS, and the four
classmethods that construct it (train_from_iterator / from_directory / from_pretrained,
plus a new from_qwen).
"""

import os
import re
import json
import copy
import pickle
from functools import lru_cache

import tiktoken
from transformers import AutoTokenizer, AutoConfig
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode

# Keep this in sync with nanochat.gpt.GPTConfig.qwen_model_name's default. Imported
# rather than hardcoded twice where possible, with a literal fallback so this file
# doesn't hard-fail to import if gpt.py is ever missing/renamed.
try:
    from nanochat.gpt import GPTConfig as _GPTConfig
    QWEN_MODEL_NAME = _GPTConfig.__dataclass_fields__["qwen_model_name"].default
except Exception:
    QWEN_MODEL_NAME = "Qwen/Qwen1.5-1.8B"

SPECIAL_TOKENS = [
    # every document begins with the Beginning of Sequence (BOS) token that delimits documents
    "<|bos|>",
    # tokens below are only used during finetuning to render Conversations into token ids
    "<|user_start|>", # user messages
    "<|user_end|>",
    "<|assistant_start|>", # assistant messages
    "<|assistant_end|>",
    "<|python_start|>", # assistant invokes python REPL tool
    "<|python_end|>",
    "<|output_start|>", # python REPL outputs back to assistant
    "<|output_end|>",
]

# NOTE: no longer used for training (Qwen1.5's own pretrained tokenizer is used as-is),
# kept only in case anything still imports SPLIT_PATTERN from this module.
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

# -----------------------------------------------------------------------------
# Byte-level decoding helpers (Qwen1.5, like GPT-2, maps raw bytes to a printable
# unicode alphabet before BPE merging - this inverts that mapping so
# decode_single_token_bytes can return exact original bytes, same guarantee tiktoken's
# version gave us).
_BYTE_DECODER = {v: k for k, v in bytes_to_unicode().items()}  # unicode char -> byte value

def _token_str_to_bytes(token_str):
    try:
        return bytes(_BYTE_DECODER[c] for c in token_str)
    except KeyError:
        # Not a pure byte-level-alphabet token (e.g. a special token's literal string) -
        # fall back to plain utf-8 bytes of the token text.
        return token_str.encode("utf-8")


class _HFTiktokenAdapter:
    """Presents a tiktoken.Encoding-like interface (encode_ordinary,
    encode_ordinary_batch, decode, decode_single_token_bytes, encode_single_token,
    special_tokens_set, n_vocab) on top of a real HF tokenizer (Qwen1.5's), so
    RustBPETokenizer's own methods below don't need to know or care that the backing
    tokenizer changed."""
    def __init__(self, hf_tokenizer, special_map):
        self.hf_tokenizer = hf_tokenizer
        self.special_map = dict(special_map)  # nanochat name -> token id, e.g. "<|bos|>" -> 151646
        self._id_to_name = {v: k for k, v in self.special_map.items()}

    @property
    def n_vocab(self):
        return len(self.hf_tokenizer)

    @property
    def special_tokens_set(self):
        return set(self.special_map.keys())

    def encode_single_token(self, text):
        if text in self.special_map:
            return self.special_map[text]
        ids = self.hf_tokenizer.encode(text, add_special_tokens=False)
        assert len(ids) == 1, f"'{text}' is not a single token in the Qwen1.5 tokenizer"
        return ids[0]

    def encode_ordinary(self, text):
        return self.hf_tokenizer.encode(text, add_special_tokens=False)

    def encode_ordinary_batch(self, texts, num_threads=8):
        # HF's fast tokenizers are already internally multi-threaded (Rust); num_threads
        # is accepted only for call-site compatibility with the old tiktoken signature.
        enc = self.hf_tokenizer(texts, add_special_tokens=False)
        return enc["input_ids"]

    def decode(self, ids):
        # clean_up_tokenization_spaces=False is essential here: HF's default behavior
        # (True, for legacy reasons) rewrites things like " ." -> "." and " n't" -> "n't"
        # on decode, which silently breaks the exact encode/decode roundtrip tiktoken
        # always guaranteed (and that tok_eval.py's `assert decoded == text` relies on).
        return self.hf_tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)

    def decode_single_token_bytes(self, token_id):
        if token_id in self._id_to_name:
            return self._id_to_name[token_id].encode("utf-8")
        token_str = self.hf_tokenizer.convert_ids_to_tokens([token_id])[0]
        return _token_str_to_bytes(token_str)


# -----------------------------------------------------------------------------
# Tokenizer based on Qwen1.5's HF tokenizer

class RustBPETokenizer:
    """Light nanochat-shaped wrapper, now backed by Qwen1.5's own HF tokenizer.
    Class name kept as-is so `from nanochat.tokenizer import RustBPETokenizer`
    elsewhere in the codebase doesn't need to change."""

    def __init__(self, enc, bos_token):
        self.enc = enc
        self.bos_token_id = self.encode_special(bos_token)

    @classmethod
    def from_qwen(cls, qwen_model_name=None):
        """Load Qwen1.5's own tokenizer and splice in nanochat's own special tokens
        (BOS + chat/tool delimiters). We try hard to avoid growing the vocabulary: HF's
        add_special_tokens() appends new ids starting at len(tokenizer), and Qwen1.5's
        embedding table is usually padded a bit larger than the tokenizer's raw
        vocabulary (for tensor-core-friendly shapes), so the new ids frequently land in
        already-unused-but-allocated embedding rows for free. If they don't fit, this
        prints an explicit warning telling you to resize the (frozen) backbone's
        embedding table in nanochat/gpt.py."""
        qwen_model_name = qwen_model_name or QWEN_MODEL_NAME
        hf_tokenizer = AutoTokenizer.from_pretrained(qwen_model_name)
        vocab_size_before = len(hf_tokenizer)

        num_added = hf_tokenizer.add_special_tokens({"additional_special_tokens": list(SPECIAL_TOKENS)})
        special_map = {name: hf_tokenizer.convert_tokens_to_ids(name) for name in SPECIAL_TOKENS}

        if num_added:
            try:
                embedding_rows = AutoConfig.from_pretrained(qwen_model_name).vocab_size
            except Exception:
                embedding_rows = vocab_size_before
            if len(hf_tokenizer) > embedding_rows:
                print(
                    f"WARNING: added {num_added} new special tokens to the Qwen1.5 tokenizer, "
                    f"pushing its size to {len(hf_tokenizer)}, PAST the backbone's embedding "
                    f"table size ({embedding_rows}). Before training/inference you need to call "
                    f"`backbone.resize_token_embeddings(len(hf_tokenizer))` inside "
                    f"GPT.init_weights() in nanochat/gpt.py, or these new token ids will index "
                    f"out of range in the frozen embedding table."
                )
            else:
                print(
                    f"Added {num_added} nanochat special tokens into unused padding rows already "
                    f"present in the Qwen1.5 embedding table ({vocab_size_before} -> {len(hf_tokenizer)} "
                    f"of {embedding_rows} available) - no backbone resize needed."
                )

        enc = _HFTiktokenAdapter(hf_tokenizer, special_map)
        return cls(enc, "<|bos|>")

    @classmethod
    def train_from_iterator(cls, text_iterator, vocab_size):
        """No longer trains a custom BPE vocab - nanochat now standardizes on Qwen1.5's
        own tokenizer (to match the frozen Qwen1.5 backbone in gpt.py). Kept with the
        same signature purely so tok_train.py (and anything else that calls this) still
        runs unmodified; `text_iterator` and `vocab_size` are accepted but ignored."""
        print(
            "NOTE: train_from_iterator no longer trains a tokenizer - nanochat now uses "
            "Qwen1.5's own pretrained tokenizer (see nanochat/gpt.py). Loading it instead "
            "of training; `text_iterator` and `vocab_size` are ignored."
        )
        return cls.from_qwen()

    @classmethod
    def from_directory(cls, tokenizer_dir):
        meta_path = os.path.join(tokenizer_dir, "nanochat_special_map.json")
        pickle_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
        if os.path.exists(pickle_path) and not os.path.exists(meta_path):
            # Legacy directory saved by the old pickle-based save() (e.g. a
            # from_pretrained() tiktoken baseline that got saved for some reason).
            with open(pickle_path, "rb") as f:
                enc = pickle.load(f)
            return cls(enc, "<|bos|>" if "<|bos|>" in getattr(enc, "special_tokens_set", set()) else "<|endoftext|>")

        # Normal case: an HF-tokenizer directory (the Qwen1.5-backed nanochat tokenizer).
        hf_tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                special_map = json.load(f)
        else:
            # Directory wasn't saved by RustBPETokenizer.save() (e.g. a raw HF tokenizer
            # dir) - nanochat's special tokens may or may not already be present.
            special_map = {}
            for name in SPECIAL_TOKENS:
                tid = hf_tokenizer.convert_tokens_to_ids(name)
                if tid is not None and tid != hf_tokenizer.unk_token_id:
                    special_map[name] = tid
            missing = [name for name in SPECIAL_TOKENS if name not in special_map]
            if missing:
                raise ValueError(
                    f"{tokenizer_dir} has no nanochat_special_map.json and is missing "
                    f"nanochat special tokens {missing}. Load it via "
                    f"RustBPETokenizer.from_qwen() and .save() it once first."
                )
        enc = _HFTiktokenAdapter(hf_tokenizer, special_map)
        return cls(enc, "<|bos|>")

    @classmethod
    def from_pretrained(cls, tiktoken_name):
        """Unchanged from before: loads a real tiktoken PUBLIC encoding by name (e.g.
        'gpt2', 'cl100k_base'). This has nothing to do with the Qwen1.5 backbone - it's
        used by benchmarking/eval scripts (like tok_eval.py) purely to compare
        nanochat's tokenizer's compression ratio against GPT-2's/GPT-4's. For the actual
        model-facing tokenizer (the one gpt.py's frozen backbone indexes into), use
        from_qwen() or get_tokenizer() instead - don't conflate the two."""
        # https://github.com/openai/tiktoken/blob/eedc8563/tiktoken_ext/openai_public.py
        enc = tiktoken.get_encoding(tiktoken_name)
        # tiktoken calls the special document delimiter token "<|endoftext|>"
        return cls(enc, "<|endoftext|>")

    def get_vocab_size(self):
        return self.enc.n_vocab

    def get_special_tokens(self):
        return self.enc.special_tokens_set

    def id_to_token(self, id):
        return self.enc.decode([id])

    @lru_cache(maxsize=32)
    def encode_special(self, text):
        return self.enc.encode_single_token(text)

    def get_bos_token_id(self):
        return self.bos_token_id

    def encode(self, text, prepend=None, append=None, num_threads=8):
        # text can be either a string or a list of strings

        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.encode_special(prepend)
        if append is not None:
            append_id = append if isinstance(append, int) else self.encode_special(append)

        if isinstance(text, str):
            ids = self.enc.encode_ordinary(text)
            if prepend is not None:
                ids.insert(0, prepend_id) # TODO: slightly inefficient here? :( hmm
            if append is not None:
                ids.append(append_id)
        elif isinstance(text, list):
            ids = self.enc.encode_ordinary_batch(text, num_threads=num_threads)
            if prepend is not None:
                for ids_row in ids:
                    ids_row.insert(0, prepend_id) # TODO: same
            if append is not None:
                for ids_row in ids:
                    ids_row.append(append_id)
        else:
            raise ValueError(f"Invalid input type: {type(text)}")

        return ids

    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)

    def decode(self, ids):
        return self.enc.decode(ids)

    def decode_single_token_bytes(self, token_id):
        return self.enc.decode_single_token_bytes(token_id)

    def save(self, tokenizer_dir):
        os.makedirs(tokenizer_dir, exist_ok=True)
        if isinstance(self.enc, _HFTiktokenAdapter):
            # normal case: the Qwen1.5-backed nanochat tokenizer
            self.enc.hf_tokenizer.save_pretrained(tokenizer_dir)
            meta_path = os.path.join(tokenizer_dir, "nanochat_special_map.json")
            with open(meta_path, "w") as f:
                json.dump(self.enc.special_map, f)
            print(f"Saved Qwen1.5 tokenizer (+ nanochat special-token map) to {tokenizer_dir}")
        else:
            # a raw tiktoken.Encoding (e.g. one of from_pretrained's gpt2/cl100k_base
            # baselines) - pickle it, same as the original implementation did.
            pickle_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
            with open(pickle_path, "wb") as f:
                pickle.dump(self.enc, f)
            print(f"Saved tokenizer encoding to {pickle_path}")

    def render_conversation(self, conversation, max_tokens=2048):
        """
        Tokenize a single Chat conversation (which we call a "doc" or "document" here).
        Returns:
        - ids: list[int] is a list of token ids of this rendered conversation
        - mask: list[int] of same length, mask = 1 for tokens that the Assistant is expected to train on.
        """
        # ids, masks that we will return and a helper function to help build them up.
        ids, mask = [], []
        def add_tokens(token_ids, mask_val):
            if isinstance(token_ids, int):
                token_ids = [token_ids]
            ids.extend(token_ids)
            mask.extend([mask_val] * len(token_ids))

        # sometimes the first message is a system message...
        # => just merge it with the second (user) message
        if conversation["messages"][0]["role"] == "system":
            # some conversation surgery is necessary here for now...
            conversation = copy.deepcopy(conversation) # avoid mutating the original
            messages = conversation["messages"]
            assert messages[1]["role"] == "user", "System message must be followed by a user message"
            messages[1]["content"] = messages[0]["content"] + "\n\n" + messages[1]["content"]
            messages = messages[1:]
        else:
            messages = conversation["messages"]
        assert len(messages) >= 1, f"Conversation has less than 1 message: {messages}"

        # fetch all the special tokens we need
        bos = self.get_bos_token_id()
        user_start, user_end = self.encode_special("<|user_start|>"), self.encode_special("<|user_end|>")
        assistant_start, assistant_end = self.encode_special("<|assistant_start|>"), self.encode_special("<|assistant_end|>")
        python_start, python_end = self.encode_special("<|python_start|>"), self.encode_special("<|python_end|>")
        output_start, output_end = self.encode_special("<|output_start|>"), self.encode_special("<|output_end|>")

        # now we can tokenize the conversation
        add_tokens(bos, 0)
        for i, message in enumerate(messages):

            # some sanity checking here around assumptions, to prevent footguns
            must_be_from = "user" if i % 2 == 0 else "assistant"
            assert message["role"] == must_be_from, f"Message {i} is from {message['role']} but should be from {must_be_from}"

            # content can be either a simple string or a list of parts (e.g. containing tool calls)
            content = message["content"]

            if message["role"] == "user":
                assert isinstance(content, str), "User messages are simply expected to be strings"
                value_ids = self.encode(content)
                add_tokens(user_start, 0)
                add_tokens(value_ids, 0)
                add_tokens(user_end, 0)
            elif message["role"] == "assistant":
                add_tokens(assistant_start, 0)
                if isinstance(content, str):
                    # simple string => simply add the tokens
                    value_ids = self.encode(content)
                    add_tokens(value_ids, 1)
                elif isinstance(content, list):
                    for part in content:
                        value_ids = self.encode(part["text"])
                        if part["type"] == "text":
                            # string part => simply add the tokens
                            add_tokens(value_ids, 1)
                        elif part["type"] == "python":
                            # python tool call => add the tokens inside <|python_start|> and <|python_end|>
                            add_tokens(python_start, 1)
                            add_tokens(value_ids, 1)
                            add_tokens(python_end, 1)
                        elif part["type"] == "python_output":
                            # python output => add the tokens inside <|output_start|> and <|output_end|>
                            # none of these tokens are supervised because the tokens come from Python at test time
                            add_tokens(output_start, 0)
                            add_tokens(value_ids, 0)
                            add_tokens(output_end, 0)
                        else:
                            raise ValueError(f"Unknown part type: {part['type']}")
                else:
                    raise ValueError(f"Unknown content type: {type(content)}")
                add_tokens(assistant_end, 1)

        # truncate to max_tokens tokens MAX (helps prevent OOMs)
        ids = ids[:max_tokens]
        mask = mask[:max_tokens]
        return ids, mask

    def visualize_tokenization(self, ids, mask, with_token_id=False):
        """Small helper function useful in debugging: visualize the tokenization of render_conversation"""
        RED = '\033[91m'
        GREEN = '\033[92m'
        RESET = '\033[0m'
        GRAY = '\033[90m'
        tokens = []
        for i, (token_id, mask_val) in enumerate(zip(ids, mask)):
            token_str = self.decode([token_id])
            color = GREEN if mask_val == 1 else RED
            tokens.append(f"{color}{token_str}{RESET}")
            if with_token_id:
                tokens.append(f"{GRAY}({token_id}){RESET}")
        return '|'.join(tokens)

    def render_for_completion(self, conversation):
        """
        Used during Reinforcement Learning. In that setting, we want to
        render the conversation priming the Assistant for a completion.
        Unlike the Chat SFT case, we don't need to return the mask.
        """
        # We have some surgery to do: we need to pop the last message (of the Assistant)
        conversation = copy.deepcopy(conversation) # avoid mutating the original
        messages = conversation["messages"]
        assert messages[-1]["role"] == "assistant", "Last message must be from the Assistant"
        messages.pop() # remove the last message (of the Assistant) inplace

        # Now tokenize the conversation
        ids, mask = self.render_conversation(conversation)

        # Finally, to prime the Assistant for a completion, append the Assistant start token
        assistant_start = self.encode_special("<|assistant_start|>")
        ids.append(assistant_start)
        return ids

# -----------------------------------------------------------------------------
# nanochat-specific convenience functions

def get_tokenizer():
    from nanochat.common import get_base_dir
    base_dir = get_base_dir()
    tokenizer_dir = os.path.join(base_dir, "tokenizer")
    return RustBPETokenizer.from_directory(tokenizer_dir)

def get_token_bytes(device="cpu"):
    import torch
    from nanochat.common import get_base_dir
    base_dir = get_base_dir()
    tokenizer_dir = os.path.join(base_dir, "tokenizer")
    token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.pt")
    assert os.path.exists(token_bytes_path), f"Token bytes not found at {token_bytes_path}? It gets written by tok_train.py"
    with open(token_bytes_path, "rb") as f:
        token_bytes = torch.load(f, map_location=device)
    return token_bytes
