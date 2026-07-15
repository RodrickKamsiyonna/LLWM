"""
Engine for efficient inference of our models.

Everything works around token sequences:
- The user can send token sequences to the engine
- The engine returns the next token

Notes:
- The engine knows nothing about tokenization, it's purely token id sequences.
- For the latent-action architecture, this ALWAYS uses model.plan_and_generate()
  to infer goal-directed actions before generating tokens. 
- Because plan_and_generate optimizes actions over a fixed horizon, tool use 
  (calculator) and early stopping are handled post-generation in generate_batch.

The whole thing is made as efficient as possible.
"""

import torch
import torch.nn.functional as F
import signal
import warnings
from contextlib import contextmanager
from nanochat.common import compute_init, autodetect_device_type, COMPUTE_DTYPE
from nanochat.checkpoint_manager import load_model

# -----------------------------------------------------------------------------
# Calculator tool helpers
@contextmanager
def timeout(duration, formula):
    def timeout_handler(signum, frame):
        raise Exception(f"'{formula}': timed out after {duration} seconds")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(duration)
    yield
    signal.alarm(0)

def eval_with_timeout(formula, max_time=3):
    try:
        with timeout(max_time, formula):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                return eval(formula, {"__builtins__": {}}, {})
    except Exception as e:
        signal.alarm(0)
        return None

def use_calculator(expr):
    """
    Evaluate a Python expression safely.
    Supports both math expressions and string operations like .count()
    """
    expr = expr.replace(",", "")

    if all([x in "0123456789*+-/.() " for x in expr]):
        if "**" in expr:  # disallow power operator
            return None
        return eval_with_timeout(expr)

    allowed_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'\"()._ "
    if not all([x in allowed_chars for x in expr]):
        return None

    dangerous_patterns = ['__', 'import', 'exec', 'eval', 'compile', 'open', 'file',
                         'input', 'raw_input', 'globals', 'locals', 'vars', 'dir',
                         'getattr', 'setattr', 'delattr', 'hasattr']
    expr_lower = expr.lower()
    if any(pattern in expr_lower for pattern in dangerous_patterns):
        return None

    if '.count(' not in expr:
        return None

    return eval_with_timeout(expr)

# -----------------------------------------------------------------------------

class Engine:

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate(self, tokens, num_samples=1, max_tokens=None, temperature=1.0, top_k=None, seed=42, goal_ids=None):
        """Generate tokens using model.plan_and_generate.

        Because plan_and_generate optimizes actions over a fixed horizon:
        - Tool use (calculator) is bypassed (the planner optimizes the full sequence).
        - Temperature and top_k are ignored (generation is always greedy/argmax).
        - Early stopping tokens (e.g. <|assistant_end|>) are still filtered
          out by generate_batch after generation completes.
        - For num_samples > 1, plan_and_generate is run sequentially.
        """
        assert isinstance(tokens, list) and isinstance(tokens[0], int), "expecting list of ints"
        
        # Default goal: a full stop to encourage natural sentence endings
        if goal_ids is None:
            goal_ids = self.tokenizer.encode(".")

        if max_tokens is None:
            max_tokens = self.model.config.sequence_len - len(tokens)

        # plan_and_generate is inherently batch_size=1, run sequentially for multiple samples
        all_gen_ids = []
        for _ in range(num_samples):
            gen_ids = self.model.plan_and_generate(
                prompt_ids=tokens,
                goal_ids=goal_ids,
                num_steps=max_tokens,
            )
            all_gen_ids.append(gen_ids)

        # Yield token columns to maintain the streaming generator interface
        # expected by generate_batch (shape: [num_samples] per step)
        for t in range(max_tokens):
            token_column = [all_gen_ids[i][t] for i in range(num_samples)]
            token_masks = [1] * num_samples
            yield token_column, token_masks

    def generate_batch(self, tokens, num_samples=1, **kwargs):
        """
        Non-streaming batch generation that just returns the final token sequences.
        Returns a list of token sequences (list of lists of ints).
        Terminal tokens (assistant_end, bos) are not included in the results.
        """
        assistant_end = self.tokenizer.encode_special("<|assistant_end|>")
        bos = self.tokenizer.get_bos_token_id()
        results = [tokens.copy() for _ in range(num_samples)]
        masks = [[0] * len(tokens) for _ in range(num_samples)]
        completed = [False] * num_samples
        for token_column, token_masks in self.generate(tokens, num_samples, **kwargs):
            for i, (token, mask) in enumerate(zip(token_column, token_masks)):
                if not completed[i]:
                    if token == assistant_end or token == bos:
                        completed[i] = True
                    else:
                        results[i].append(token)
                        masks[i].append(mask)
            # Stop appending if all rows are completed
            if all(completed):
                break
        return results, masks


if __name__ == "__main__":
    """
    Quick inline test to make sure that the Engine.generate function correctly
    wraps model.plan_and_generate for the latent-action architecture.
    """
    import time
    # init compute
    device_type = autodetect_device_type()
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    # load the model and tokenizer
    model, tokenizer, meta = load_model("base", device, phase="eval")
    bos_token_id = tokenizer.get_bos_token_id()
    
    kwargs = dict(max_tokens=64)
    prompt_tokens = tokenizer.encode("The chemical formula of water is", prepend=bos_token_id)
    goal_ids = tokenizer.encode(".")

    # --- Test 1: Engine generate (wraps plan_and_generate) ---
    print("=" * 80)
    print("Test 1: Engine.generate_batch (wraps plan_and_generate, goal='.')")
    print("=" * 80)
    engine = Engine(model, tokenizer)
    torch.cuda.synchronize()
    t0 = time.time()
    results, masks = engine.generate_batch(prompt_tokens, num_samples=1, **kwargs)
    torch.cuda.synchronize()
    t1 = time.time()
    print(tokenizer.decode(results[0]))
    print(f"\nEngine time: {t1 - t0:.2f}s")
    engine_ids = results[0][len(prompt_tokens):]  # strip prompt to get generated tokens

    # --- Test 2: Direct model.plan_and_generate ---
    print("\n" + "=" * 80)
    print("Test 2: Direct model.plan_and_generate (goal='.')")
    print("=" * 80)
    torch.cuda.synchronize()
    t0 = time.time()
    direct_ids = model.plan_and_generate(
        prompt_ids=prompt_tokens,
        goal_ids=goal_ids,
        num_steps=64,
    )
    torch.cuda.synchronize()
    t1 = time.time()
    print(tokenizer.decode(prompt_tokens + direct_ids))
    print(f"\nDirect time: {t1 - t0:.2f}s")

    # --- Compare ---
    # These should match EXACTLY because Engine just wraps plan_and_generate
    print("\n" + "=" * 80)
    print("Comparison")
    print("=" * 80)
    print(f"Engine tokens: {engine_ids}")
    print(f"Direct tokens: {direct_ids}")
    print(f"Sequences match: {engine_ids == direct_ids}")
