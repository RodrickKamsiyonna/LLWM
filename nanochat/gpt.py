"""
GPT model (rewrite #2): frozen Qwen1.5 backbone + our own world-model heads
Notable features:
- The trunk (token embedding + attention/MLP stack that used to be hand-rolled here,
  with rotary embeddings, QK-norm, GQA, value-embeddings, smear, backout, etc.) is now
  a frozen Qwen1.5 model loaded via `transformers`. We never train it.
- Qwen1.5's own lm_head is never called - we take its `last_hidden_state` and feed it
  into our own ActionEncoder / Predictor, which is what actually gets trained. The
  Predictor's own output head (`Predictor.head`) is the "projection head" now.
- ActionEncoder / Predictor / AdaLN / equilibrium-matching-loss logic is UNCHANGED from
  before - only where `h` (the per-token hidden state) comes from has changed.
- Public interface (GPTConfig, GPT.__init__, .init_weights(), .forward(), .encode(),
  .encode_embeddings(), .setup_optimizer(), .plan_and_generate(), .get_device(),
  .num_scaling_params(), the various *_flops/*_bytes estimators, ._new_kv_cache()) is
  kept intact so calling code (train scripts, engine.py, eval scripts, ...) does not
  need to change. See the big caveat about `_new_kv_cache()` below though - please read
  it, it's the one place where "no other script needs to change" is an assumption I
  can't verify without seeing engine.py.
"""

from functools import partial
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import AutoConfig, AutoModel
except ImportError as e:
    raise ImportError(
        "This version of nanochat.gpt needs `transformers` to load the Qwen1.5 backbone. "
        "Install it with `pip install transformers`."
    ) from e

from nanochat.common import get_dist_info, print0, COMPUTE_DTYPE
from nanochat.optim import MuonAdamW


@dataclass
class GPTConfig:
    # --- the one real input now: which frozen backbone to wrap ---
    qwen_model_name: str = "Qwen/Qwen1.5-0.5B"

    # --- world-model head config (still real inputs) ---
    action_dim: int = None
    action_encoder_depth_ratio: int = 12 # ActionEncoder gets ceil(n_layer / this); Predictor is fixed at ceil(n_layer/2)
    eqm_lambda: float = 1.0

    # --- derived from the backbone's HF config in __post_init__, kept here purely so
    # existing code that reads config.n_embd / config.n_layer / config.vocab_size / etc.
    # (checkpoint naming, scaling-law logging, dataloader chunking, ...) keeps working
    # without modification. Any value you pass in for these is overwritten. ---
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    window_pattern: str = "SSSL"  # no longer used (Qwen1.5 base does plain full causal attention); kept only so old call sites that pass it don't break

    def __post_init__(self):
        hf_config = AutoConfig.from_pretrained(self.qwen_model_name)
        self.hf_config = hf_config
        self.n_embd = hf_config.hidden_size
        self.n_layer = hf_config.num_hidden_layers
        self.n_head = hf_config.num_attention_heads
        self.n_kv_head = getattr(hf_config, "num_key_value_heads", self.n_head)
        self.vocab_size = hf_config.vocab_size
        self.sequence_len = min(self.sequence_len, getattr(hf_config, "max_position_embeddings", self.sequence_len))
        if self.action_dim is None:
            self.action_dim = max(1, self.n_embd // 32)


def norm(x):
    return F.rms_norm(x, (x.size(-1),)) # note that this will run in bf16, seems ok


class Linear(nn.Linear):
    """nn.Linear that casts weights to match input dtype in forward.
    Replaces autocast: master weights stay fp32 for optimizer precision,
    but matmuls run in the activation dtype (typically bf16 from embeddings).
    NOTE: this is a distinct class from torch.nn.Linear on purpose - the Qwen backbone's
    internal projections are plain nn.Linear, so `isinstance(m, Linear)` scans below
    (num_matmul_params) naturally only pick up our own trainable ActionEncoder/Predictor
    weights and never the frozen backbone's."""
    def forward(self, x):
        return F.linear(x, self.weight.to(dtype=x.dtype))


class AdaLN(nn.Module):
    """DiT-style adaptive-norm modulation: projects the latent action into per-channel
    (shift, scale, gate). Zero-init so at init gate=0 -> conditioned branch is a no-op
    residual; training gradually turns on the action's influence. This is what "condition
    with layer norm" means once norm() has no learnable affine params of its own."""
    def __init__(self, action_dim, n_embd):
        super().__init__()
        self.proj = Linear(action_dim, 3 * n_embd, bias=False)

    def forward(self, action):
        shift, scale, gate = self.proj(action).chunk(3, dim=-1)
        return shift, scale, gate


class MLPBlock(nn.Module):
    """Pre-norm residual MLP, same relu^2 / no-bias pattern as the trunk's old MLP, at a
    configurable width. Used by the ActionEncoder."""
    def __init__(self, dim, expansion=4):
        super().__init__()
        self.c_fc = Linear(dim, expansion * dim, bias=False)
        self.c_proj = Linear(expansion * dim, dim, bias=False)

    def forward(self, x):
        h = F.relu(self.c_fc(norm(x))).square()
        return x + self.c_proj(h)


class ActionEncoder(nn.Module):
    """Infers a ~ N(mean, std) explaining the transition h_t -> y_t. Scaled relative to
    the backbone on both axes: half the width (n_embd // 2) and a fraction of the depth
    (ceil(n_layer / action_encoder_depth_ratio), default quarter-depth) - it only needs
    to compress down to action_dim (n_embd // 32), not model vocab-scale distributions,
    so it tracks the backbone's size without needing to match it. n_layer here is the
    Qwen backbone's depth (see GPTConfig.__post_init__)."""
    def __init__(self, config):
        super().__init__()
        hidden = config.n_embd // 2
        depth = max(1, -(-config.n_layer // config.action_encoder_depth_ratio))  # ceil division
        self.in_proj = Linear(2 * config.n_embd, hidden, bias=False)
        self.blocks = nn.ModuleList([MLPBlock(hidden) for _ in range(depth)])
        self.mean_head = Linear(hidden, config.action_dim, bias=False)
        self.log_std_head = Linear(hidden, config.action_dim, bias=False)

    def forward(self, h, y_emb):
        x = self.in_proj(torch.cat([h.detach(), y_emb.detach()], dim=-1))
        for block in self.blocks:
            x = block(x)
        x = norm(x)
        mean = self.mean_head(x)
        log_std = torch.clamp(self.log_std_head(x), min=-5.0, max=2.0)
        return mean, log_std


class MLP(nn.Module):
    """Plain relu^2 / no-bias MLP with NO internal residual add (unlike MLPBlock) -
    PredictorBlock applies its own gated residual on top of this, via AdaLN's gate."""
    def __init__(self, config):
        super().__init__()
        self.c_fc = Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class PredictorBlock(nn.Module):
    """Pre-norm relu^2 MLP, modulated by the action via AdaLN. Purely pointwise - h_t
    already carries full causal context from the (frozen) backbone's attention, so no
    attention is needed here, just a nonlinear combination of (context, action) at each
    position."""
    def __init__(self, config):
        super().__init__()
        self.mlp = MLP(config)
        self.ada = AdaLN(config.action_dim, config.n_embd)

    def forward(self, x, action):
        shift, scale, gate = self.ada(action)
        h = norm(x) * (1 + scale) + shift
        x = x + gate * self.mlp(h)
        return x


class Predictor(nn.Module):
    """Stack of AdaLN-MLP blocks operating pointwise on (h_t, action_t). Depth =
    ceil(n_layer / 2) of the Qwen backbone: it only resolves 'what does this action mean
    here', not re-derive context, so it needs less depth than the backbone. Width =
    n_embd, matching the encoder and the backbone's hidden size exactly. `self.head` is
    the model's actual output projection now (Qwen's own lm_head is never used)."""
    def __init__(self, config, pad_vocab_size_to=64):
        super().__init__()
        self.n_layer = -(-config.n_layer // 12)  # ceil(n_layer / 2)
        self.blocks = nn.ModuleList([PredictorBlock(config) for _ in range(self.n_layer)])
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        self.head = Linear(config.n_embd, padded_vocab_size, bias=False)

    def forward(self, h, action):
        x = h
        for block in self.blocks:
            x = block(x, action)
        return self.head(norm(x))


class GPT(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        """
        NOTE this still runs in a meta device context in some call sites (fast schema
        init). Everything built directly here (action_encoder, predictor) is fine under
        meta - it's freshly initialized from scratch either way, and init_weights()
        fills in real values. The Qwen backbone is different: it needs real, downloaded,
        pretrained weights, not meta/random ones, so it is deliberately NOT constructed
        here. `self.backbone` stays None until init_weights() actually loads it. Anything
        that calls encode()/forward()/plan_and_generate() before init_weights() has run
        will hit an assertion telling you so.
        """
        super().__init__()
        self.config = config
        self.backbone = None  # frozen Qwen1.5 backbone; populated in init_weights()
        self.action_encoder = ActionEncoder(config)
        self.predictor = Predictor(config, pad_vocab_size_to=pad_vocab_size_to)

    @torch.no_grad()
    def init_weights(self):
        """
        Initialize the action_encoder / predictor heads (random init, same scheme as
        before) and load the real frozen Qwen1.5 backbone (real pretrained weights,
        frozen, eval mode).
        """
        device = self.action_encoder.mean_head.weight.device

        # ActionEncoder
        a_in = self.action_encoder.in_proj.weight.shape[1]
        torch.nn.init.uniform_(self.action_encoder.in_proj.weight, -a_in**-0.5, a_in**-0.5)
        for block in self.action_encoder.blocks:
            fan_in = block.c_fc.weight.shape[1]
            torch.nn.init.uniform_(block.c_fc.weight, -fan_in**-0.5, fan_in**-0.5)
            torch.nn.init.zeros_(block.c_proj.weight)
        torch.nn.init.normal_(self.action_encoder.mean_head.weight)
        torch.nn.init.normal_(self.action_encoder.log_std_head.weight)

        # Predictor
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5  # sqrt(3) multiplier makes Uniform match the Normal std used before
        for block in self.predictor.blocks:
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s * 0.4, s * 0.4)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
            torch.nn.init.normal_(block.ada.proj.weight)
        torch.nn.init.normal_(self.predictor.head.weight, mean=0.0, std=0.001)

        # Frozen Qwen1.5 backbone - loaded here (not __init__) specifically to get real
        # weights even when the rest of the model was constructed under `torch.device("meta")`.
        dtype = COMPUTE_DTYPE if COMPUTE_DTYPE != torch.float16 else torch.float32
        print0(f"Loading frozen Qwen1.5 backbone: {self.config.qwen_model_name}")
        backbone = AutoModel.from_pretrained(self.config.qwen_model_name, torch_dtype=dtype)
        backbone = backbone.to(device)
        backbone.eval()
        for p in backbone.parameters():
            p.requires_grad_(False)
        assert backbone.config.vocab_size == self.config.vocab_size, "backbone vocab_size drifted from GPTConfig - reconstruct GPTConfig after changing qwen_model_name"
        assert backbone.config.hidden_size == self.config.n_embd, "backbone hidden_size drifted from GPTConfig - reconstruct GPTConfig after changing qwen_model_name"
        self.backbone = backbone

    def get_device(self):
        if self.backbone is not None:
            return next(self.backbone.parameters()).device
        return self.action_encoder.mean_head.weight.device

    @property
    def wte(self):
        """Back-compat alias: several nanochat scripts (loss_eval.py's evaluate_bpb, in
        particular) call `model.wte(ids)` directly to get token embeddings, from back
        when the trunk had its own dedicated embedding table. That table is now inside
        the frozen Qwen backbone, so this just forwards to it - `model.wte(ids)` still
        works exactly as before without those call sites needing to change."""
        assert self.backbone is not None, "call init_weights() first to load the Qwen backbone"
        return self.backbone.get_input_embeddings()

    def _embed(self, idx):
        """Token ids -> Qwen's own input embeddings (used for building inputs_embeds
        outside of a plain forward pass, e.g. in plan_and_generate). Equivalent to
        self.wte(idx) - kept as a separate method for internal readability.""" 
        return self.wte(idx)

    # ---- FLOPs / param accounting -------------------------------------------------

    def num_matmul_params(self):
        """Trainable matmul params (ActionEncoder + Predictor only). The Qwen backbone
        is frozen and intentionally excluded here - see num_backbone_params() for its
        contribution to FLOPs accounting."""
        return sum(m.weight.numel() for m in self.modules() if isinstance(m, Linear))

    def num_backbone_params(self):
        if self.backbone is not None:
            return sum(p.numel() for p in self.backbone.parameters())
        return self._estimate_backbone_params_from_config()

    def _estimate_backbone_params_from_config(self):
        """Fallback param-count estimate from the HF config alone, for use before
        init_weights() has actually downloaded/materialized the backbone."""
        c = self.config.hf_config
        h, L, inter, V = c.hidden_size, c.num_hidden_layers, c.intermediate_size, c.vocab_size
        n_head = c.num_attention_heads
        n_kv = getattr(c, "num_key_value_heads", n_head)
        head_dim = h // n_head
        attn = h * h + 2 * h * n_kv * head_dim + h * h  # q_proj, k_proj+v_proj (GQA), o_proj
        mlp = 3 * h * inter  # SwiGLU: gate_proj, up_proj, down_proj
        embed = h * V  # input embedding table (Qwen1.5 does not tie embeddings)
        return L * (attn + mlp) + embed

    def estimate_flops(self):
        """FLOPs per token (forward + backward). Backbone is frozen so it only pays
        forward cost (2 FLOPs/param); ActionEncoder+Predictor pay full forward+backward
        (6 FLOPs/param). Attention FLOPs are backbone-only now (Predictor is pointwise,
        Qwen1.5 base uses plain full causal attention, no sliding window)."""
        h = self.config.n_head
        q = self.config.n_embd // self.config.n_head
        t = self.config.sequence_len
        backbone_attn_flops = 12 * h * q * t * self.config.n_layer
        trainable_flops = 6 * self.num_matmul_params()
        backbone_flops = 2 * self.num_backbone_params() + backbone_attn_flops
        return trainable_flops + backbone_flops

    def estimate_decode_flops(self, context_len):
        """Forward FLOPs to decode one token at a given context length during inference."""
        h = self.config.n_head
        q = self.config.n_embd // self.config.n_head
        backbone_attn = self.config.n_layer * 4 * h * q * min(context_len, self.config.sequence_len)
        decode_flops = 2 * (self.num_backbone_params() + self.num_matmul_params()) + backbone_attn
        return decode_flops

    def estimate_prefill_flops(self, num_tokens):
        """Forward FLOPs to prefill a prompt: causal, so token t attends to min(t, ctx)."""
        h = self.config.n_head
        q = self.config.n_embd // self.config.n_head
        w = min(self.config.sequence_len, num_tokens)
        attended_tokens = w * (w + 1) // 2 + (num_tokens - w) * w  # ramp up to w, then flat
        backbone_attn = self.config.n_layer * 4 * h * q * attended_tokens
        matmul_flops = 2 * (self.num_backbone_params() + self.num_matmul_params()) * num_tokens
        return matmul_flops + backbone_attn

    def kv_bytes_per_token(self):
        """Bytes to *store* one token of KV cache during inference, across all backbone
        layers (the Predictor has no attention, so no KV cache of its own)."""
        c = self.config.hf_config
        n_kv = getattr(c, "num_key_value_heads", c.num_attention_heads)
        head_dim = c.hidden_size // c.num_attention_heads
        kv_dtype_bytes = COMPUTE_DTYPE.itemsize
        return c.num_hidden_layers * 2 * n_kv * head_dim * kv_dtype_bytes

    def kv_read_bytes(self, context_len):
        """Bytes of KV cache *read* by one decode step at a given context length. No
        sliding window in the Qwen1.5 base backbone, so this is just every cached token."""
        return self.kv_bytes_per_token() * min(context_len, self.config.sequence_len)

    def num_scaling_params(self):
        """
        Return detailed parameter counts for scaling law analysis.

        IMPORTANT: this keeps the OLD key names ('wte', 'value_embeds',
        'transformer_matrices', 'lm_head', 'action_encoder', 'predictor', 'scalars',
        'total') alive too, purely so call sites like base_train.py's
        `params_counts['transformer_matrices'] + params_counts['lm_head']` don't
        KeyError. 'transformer_matrices' is mapped to the frozen backbone's param count
        (closest old-key analogue - it used to be "the trunk's big matrix stack", which
        is now Qwen1.5 instead of our own attention/MLP blocks). 'wte', 'value_embeds',
        and 'scalars' are 0 - there's no separate trainable embedding table or per-layer
        scalar params anymore, they're gone along with the old trunk.

        CAVEAT for anything doing Chinchilla-style compute-optimal scaling-law math with
        these numbers: 'transformer_matrices' (the backbone) is FROZEN - it costs forward
        FLOPs but receives no gradient and isn't "trained" in the way the old trunk was.
        If your scaling-law code assumes `transformer_matrices + lm_head` params are all
        being trained, that assumption no longer holds; you likely want 'trainable_total'
        (action_encoder + predictor + lm_head only) for that kind of analysis instead.
        """
        backbone = self.num_backbone_params()
        lm_head = sum(p.numel() for p in self.predictor.head.parameters())
        action_encoder = sum(p.numel() for p in self.action_encoder.parameters())
        predictor = sum(p.numel() for block in self.predictor.blocks for p in block.parameters())
        trainable_total = lm_head + action_encoder + predictor
        total = backbone + trainable_total
        if self.backbone is not None:
            assert total == sum(p.numel() for p in self.parameters()), "Parameter count mismatch"
        return {
            # new, clearer names
            'backbone_frozen': backbone, 'trainable_total': trainable_total,
            # old names, kept alive for backward compatibility (see docstring caveat above)
            'wte': 0, 'value_embeds': 0, 'scalars': 0,
            'transformer_matrices': backbone,
            'lm_head': lm_head, 'action_encoder': action_encoder, 'predictor': predictor,
            'total': total,
        }

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, scalar_lr=0.5):
        """Only ActionEncoder + Predictor are trained now - the backbone is frozen, so
        there is no more embedding/value-embed/per-layer-scalar/matrix param group for
        it. `embedding_lr`/`scalar_lr` are accepted for call-site compatibility but
        unused (nothing left to apply them to)."""
        model_dim = self.config.n_embd

        predictor_mlp_params = [p for block in self.predictor.blocks for p in block.mlp.parameters()]
        predictor_ada_params = [p for block in self.predictor.blocks for p in block.ada.parameters()]
        lm_head_params = list(self.predictor.head.parameters())
        action_encoder_params = list(self.action_encoder.parameters())

        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters \u221d1/\u221a({model_dim}/768) = {dmodel_lr_scale:.6f}")

        param_groups = [
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', params=action_encoder_params, lr=matrix_lr * dmodel_lr_scale, betas=(0.8, 0.95), eps=1e-10, weight_decay=weight_decay),
            dict(kind='adamw', params=predictor_ada_params, lr=0.2, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0),
        ]

        for shape in sorted({p.shape for p in predictor_mlp_params}):
            group_params = [p for p in predictor_mlp_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon',
                params=group_params,
                lr=matrix_lr,
                momentum=0.95,
                ns_steps=5,
                beta2=0.9,
                weight_decay=weight_decay,
            ))

        assigned = set()
        for group in param_groups:
            for p in group["params"]:
                if id(p) in assigned:
                    raise RuntimeError(f"Duplicate parameter assigned to optimizer: {tuple(p.shape)}")
                assigned.add(id(p))

        # Only trainable params should be assigned - the frozen backbone (requires_grad=False)
        # is deliberately excluded from this check.
        all_trainable_params = {id(p) for p in self.parameters() if p.requires_grad}
        missing = all_trainable_params - assigned
        extra = assigned - all_trainable_params

        assert not missing, f"{len(missing)} trainable model parameters are missing from optimizer groups."
        assert not extra, f"{len(extra)} optimizer parameters are not part of the model's trainable parameters."

        optimizer = MuonAdamW(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    # ---- forward / encode -----------------------------------------------------------

    @torch._dynamo.disable
    def encode(self, idx, kv_cache=None):
        """Runs token ids through the frozen Qwen1.5 backbone and returns h (post-norm),
        exactly like the old hand-rolled trunk's encode() did - just with Qwen managing
        its own attention/rotary/KV-cache internally instead of our flash-attn code.

        @torch._dynamo.disable: base_train.py wraps the whole GPT model in
        torch.compile(). The old trunk was written in plain, compile-friendly torch ops
        specifically for that; HF's Qwen2Model implementation is not, and tracing/
        compiling through it directly is known to be extremely slow (and buys nothing,
        since the backbone is frozen - there's no gradient-fusion benefit to compiling
        through it). This tells dynamo to treat the backbone call as an opaque boundary
        (one graph break here) and only compile the parts we actually wrote below
        (ActionEncoder/Predictor/loss math), which are the parts that benefit from it."""
        assert self.backbone is not None, "call init_weights() first to load the Qwen backbone"
        B, T = idx.size()
        assert T <= self.config.sequence_len, f"sequence length {T} exceeds backbone's max_position_embeddings-derived sequence_len {self.config.sequence_len}"
        if kv_cache is None:
            out = self.backbone(input_ids=idx, use_cache=False)
        else:
            out = self.backbone(input_ids=idx, past_key_values=kv_cache, use_cache=True)
        h = out.last_hidden_state.to(COMPUTE_DTYPE)
        return norm(h)

    @torch._dynamo.disable
    def encode_embeddings(self, emb, kv_cache=None):
        """Same as encode(), but input is already-embedded (used during planning, where
        'thinking' tokens are soft/probability-weighted rather than hard ids). See
        encode()'s docstring for why this is also excluded from torch.compile tracing."""
        assert self.backbone is not None, "call init_weights() first to load the Qwen backbone"
        emb = emb.to(COMPUTE_DTYPE)
        if kv_cache is None:
            out = self.backbone(inputs_embeds=emb, use_cache=False)
        else:
            out = self.backbone(inputs_embeds=emb, past_key_values=kv_cache, use_cache=True)
        h = out.last_hidden_state.to(COMPUTE_DTYPE)
        return norm(h)

    def forward(self, idx, targets, kv_cache=None):
        """Latent-action objective: infer action from (h, target) via ActionEncoder,
        predict target from (h, action) via the AdaLN-conditioned Predictor.
        Returns CE + KL + EQM, same convention/keys as before."""
        h = self.encode(idx, kv_cache=kv_cache)
        y_emb = self._embed(targets.clamp(min=0)).to(h.dtype)  # clamp guards ignore_index=-1
        mean, log_std = self.action_encoder(h, y_emb)
        std = log_std.exp()
        action = mean + std * torch.randn_like(mean)

        logits = self.predictor(h, action)
        logits = logits[..., :self.config.vocab_size].float()
        logits = 15 * torch.tanh(logits / 15)

        ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        kl_loss = -0.5 * torch.mean(1 + 2 * log_std - mean.pow(2) - std.pow(2))
        eqm_loss = self.equilibrium_matching_loss(h, action, targets)
        return {
            "ce_loss": ce_loss,
            "kl_loss": kl_loss,
            "eqm_loss": eqm_loss}

    import torch._dynamo

    @torch._dynamo.disable
    def equilibrium_matching_loss(self, h, action, targets, eqm_lambda=None):
        """Equilibrium Matching (EQM): trains the predictor's cross-entropy 'energy'
        landscape over actions to have a specific gradient field - pointing from a random
        noise action `eps` back toward the actual sampled `action`, scaled by how far along
        the noise->action interpolation gamma we are. This is what makes gradient descent
        over actions at planning time (plan_and_generate) actually work: without it, nothing
        guarantees the predictor's loss surface w.r.t. action has a useful slope away from
        the training point, only exactly at it. Unchanged from before."""
        if eqm_lambda is None:
            eqm_lambda = self.config.eqm_lambda
        B, T, _ = action.shape
        gamma = torch.rand(B, T, 1, device=action.device, dtype=action.dtype)
        eps = torch.randn_like(action)
        act_gamma = (gamma * action.detach() + (1 - gamma) * eps).requires_grad_(True)

        logits_noisy = self.predictor(h.detach(), act_gamma)
        logits_noisy = logits_noisy[..., :self.config.vocab_size].float()
        energy = F.cross_entropy(
            logits_noisy.reshape(-1, self.config.vocab_size), targets.reshape(-1),
            ignore_index=-1, reduction='sum',
        )

        grad_energy = torch.autograd.grad(energy, act_gamma, create_graph=True)[0]
        target_grad = (eps - action.detach()) * eqm_lambda * (1 - gamma)
        return (grad_energy - target_grad).pow(2).mean()

    # ---- inference --------------------------------------------------------------

    def _new_kv_cache(self, batch_size=1):
        """
        Returns an empty KV cache for the backbone.

        IMPORTANT / PLEASE VERIFY: the old trunk used a custom nanochat.engine.KVCache
        tied to our hand-rolled flash-attention code (with methods like .get_pos(),
        .advance(), .get_layer_cache(), .cache_seqlens). Now that the trunk is Qwen1.5,
        those calls no longer make sense - `transformers` manages its own KV cache via a
        `DynamicCache` object, which is what this now returns. `encode()` /
        `encode_embeddings()` pass it straight through to the backbone as
        `past_key_values` and it mutates itself in place, which covers the *common*
        engine.py pattern of "create a cache, then repeatedly pass the same object back
        into forward calls during decoding."

        Where this could break something: if engine.py calls any nanochat-KVCache-
        specific methods directly on the object returned here (e.g. `kv_cache.get_pos()`
        to compute position ids itself, or `kv_cache.advance(T)`), that will now fail,
        since DynamicCache doesn't have those methods. `batch_size` is also currently
        unused - DynamicCache infers its batch dim from the first tensor it caches. If
        engine.py does per-row/batched cache manipulation (e.g. for speculative decoding
        or beam search) beyond "hold it and pass it back", that code will need small
        updates too. I did not have engine.py in front of me, so please grep it for
        `kv_cache.` and sanity check the calls it makes against DynamicCache's API
        (https://huggingface.co/docs/transformers/main/en/kv_cache) before assuming this
        is a drop-in swap.
        """
        from transformers import DynamicCache
        return DynamicCache()

    @torch.enable_grad()
    def plan_and_generate(self, prompt_ids, goal_ids, num_steps, lr=0.05, num_iters=150):
        """Optimize `num_steps` latent thinking-actions so the model's own
        ActionEncoder/Predictor can explain the known goal continuation, then decode
        tokens using the planned actions.

        KV-cache accelerated: each rollout inside compute_goal_loss reuses a single,
        growing `DynamicCache` (built fresh at the start of every rollout, since the
        actions it depends on change every Adam step) instead of recomputing the whole
        prefix through the backbone at every one of the num_steps positions. This turns
        the backbone-attention cost of one rollout from O(num_steps^2) into O(num_steps),
        same as the previous full-recompute version numerically (gradients still flow
        end-to-end through the cached activations back into `actions`), just much
        cheaper. Final decoding (after planning) is also cache-accelerated the same way.
        """
        from transformers import DynamicCache
        self.eval()
        device = self.get_device()
        prompt = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        goal = torch.tensor([goal_ids], dtype=torch.long, device=device)
        actions = torch.randn(1, num_steps, self.config.action_dim, device=device, requires_grad=True)
        planner = torch.optim.Adam([actions], lr=lr)
        embed_weight = self.backbone.get_input_embeddings().weight

        def compute_goal_loss(act):
            cache = DynamicCache()
            prompt_emb = self._embed(prompt).to(COMPUTE_DTYPE)
            h_last = self.encode_embeddings(prompt_emb, kv_cache=cache)[:, -1:, :]
            for step in range(num_steps):
                logits = self.predictor(h_last, act[:, step:step+1, :])
                probs = F.softmax(logits[..., :self.config.vocab_size].float(), dim=-1).to(COMPUTE_DTYPE)
                soft_emb = probs @ embed_weight[:self.config.vocab_size]
                h_last = self.encode_embeddings(soft_emb, kv_cache=cache)[:, -1:, :]

            goal_emb = self._embed(goal).to(COMPUTE_DTYPE)
            losses = []
            for j in range(goal.shape[1]):
                mean, _ = self.action_encoder(h_last, goal_emb[:, j:j+1, :])  # deterministic goal action
                logits = self.predictor(h_last, mean)
                losses.append(F.cross_entropy(logits[:, -1, :self.config.vocab_size].float(), goal[:, j]))
                h_last = self.encode_embeddings(goal_emb[:, j:j+1, :], kv_cache=cache)[:, -1:, :]  # teacher force
            return torch.stack(losses).mean()

        for _ in range(num_iters):
            planner.zero_grad()
            compute_goal_loss(actions).backward()
            planner.step()

        with torch.no_grad():
            cache = DynamicCache()
            prompt_emb = self._embed(prompt).to(COMPUTE_DTYPE)
            h_last = self.encode_embeddings(prompt_emb, kv_cache=cache)[:, -1:, :]
            gen_ids = []
            for step in range(num_steps):
                logits = self.predictor(h_last, actions[:, step:step+1, :])
                tok = logits[..., :self.config.vocab_size].argmax(dim=-1)
                gen_ids.append(tok.item())
                new_emb = self._embed(tok).to(COMPUTE_DTYPE)
                h_last = self.encode_embeddings(new_emb, kv_cache=cache)[:, -1:, :]
        return gen_ids
