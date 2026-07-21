"""
A number of functions that help with evaluating a base model.
"""
import math
import torch
import torch.nn.functional as F
import torch.distributed as dist

@torch.no_grad()
def evaluate_bpb(model, batches, steps, token_bytes):
    """
    Instead of the naive 'mean loss', this function returns the bits per byte (bpb),
    which is a tokenization vocab size-independent metric, meaning you are still comparing
    apples:apples if you change the vocab size. The way this works is that instead of just
    calculating the average loss as usual, you calculate the sum loss, and independently
    also the sum bytes (of all the target tokens), and divide. This normalizes the loss by
    the number of bytes that the target tokens represent.

    The added complexity is so that:
    1) All "normal" tokens are normalized by the length of the token in bytes
    2) No special tokens (e.g. <|bos|>) are included in the metric - they are masked out.
    3) No actively masked tokens (using ignore_index of e.g. -1) are included in the metric.

    In addition to evaluate_loss, we need the token_bytes tensor:
    It is a 1D tensor of shape (vocab_size,), indicating the number of bytes for
    each token id, or 0 if the token is to not be counted (e.g. special tokens).

    NOTE: This computes per-token CE loss directly from the model's components
    (encode -> action_encoder -> predictor), using the mean action (no sampling
    noise) for deterministic evaluation.
    """
    total_nats = torch.tensor(0.0, dtype=torch.float32, device=model.get_device())
    total_bytes = torch.tensor(0, dtype=torch.int64, device=model.get_device())
    batch_iter = iter(batches)
    vocab_size = model.config.vocab_size

    for _ in range(steps):
        x, y = next(batch_iter)
        B, T = x.size()

        # Compute per-token CE loss deterministically using the model's components.
        # Use the mean action (no sampling noise) for stable, reproducible eval —
        # same idea as using the mean latent in a VAE at test time.
        h = model.encode(x)
        y_emb = model.wte(y.clamp(min=0)).to(h.dtype)
        mean, _ = model.action_encoder(h, y_emb)
        logits = model.predictor(h, mean)
        logits = logits[..., :vocab_size].float()
        logits = 15 * torch.tanh(logits / 15)  # same logit squishing as training forward

        loss2d = F.cross_entropy(
            logits.view(-1, vocab_size), y.view(-1),
            ignore_index=-1, reduction='none',
        ).view(B, T)

        loss2d = loss2d.view(-1)  # flatten
        y_flat = y.view(-1)       # flatten
        if (y_flat.int() < 0).any():  # mps does not currently have kernel for < 0 for int64, only int32
            # slightly more complex code path if some target tokens are ignore_index (e.g. -1)
            # any target token < 0 is to be ignored: do NOT index token_bytes with negatives
            valid = y_flat >= 0
            y_safe = torch.where(valid, y_flat, torch.zeros_like(y_flat))
            # map valid targets to their byte length; ignored targets contribute 0 bytes
            num_bytes2d = torch.where(
                valid,
                token_bytes[y_safe],
                torch.zeros_like(y_flat, dtype=token_bytes.dtype)
            )
            total_nats += (loss2d * (num_bytes2d > 0)).sum()
            total_bytes += num_bytes2d.sum()
        else:
            # fast path: no ignored targets, safe to index directly
            num_bytes2d = token_bytes[y_flat]
            total_nats += (loss2d * (num_bytes2d > 0)).sum()
            total_bytes += num_bytes2d.sum()

    # sum reduce across all ranks
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if world_size > 1:
        dist.all_reduce(total_nats, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_bytes, op=dist.ReduceOp.SUM)

    # move both to cpu, calculate bpb and return
    total_nats = total_nats.item()
    total_bytes = total_bytes.item()
    if total_bytes == 0:
        return float('inf')
    bpb = total_nats / (math.log(2) * total_bytes)
    return bpb
