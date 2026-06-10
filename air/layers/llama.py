"""Low-rank SVD replacement modules for Llama attention / MLP.

Each original weight W (m x n) is replaced by two linears: a down-projection V (n -> k)
and an up-projection U (k -> m), so y = U(V x) with k(m+n) << mn parameters. These are
standard (non-LRP) modules — the influence/LRP analysis runs on the *original* model
beforehand; here we only need a correct, fast forward for the compressed model.

Attention mirrors the HF Llama eager path (RoPE + GQA + causal softmax), reusing
transformers' apply_rotary_pos_emb / repeat_kv. A standard KV cache supports generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb, repeat_kv, LlamaRotaryEmbedding,
)


class SVD_LlamaMLP(nn.Module):
    """gate/up/down each factorized: down_u(down_v( act(gate_u(gate_v x)) * up_u(up_v x) ))."""

    def __init__(self, config, ranks, dtype):
        super().__init__()
        self.gate_rank, self.up_rank, self.down_rank = ranks
        h, inter = config.hidden_size, config.intermediate_size
        bias = config.mlp_bias
        self.act_fn = ACT2FN[config.hidden_act]
        # gate and up share input x -> fuse their down-projections
        self.gate_up_v_proj = nn.Linear(h, self.gate_rank + self.up_rank, bias=bias, dtype=dtype)
        self.gate_u_proj = nn.Linear(self.gate_rank, inter, bias=bias, dtype=dtype)
        self.up_u_proj = nn.Linear(self.up_rank, inter, bias=bias, dtype=dtype)
        self.down_v_proj = nn.Linear(inter, self.down_rank, bias=bias, dtype=dtype)
        self.down_u_proj = nn.Linear(self.down_rank, h, bias=bias, dtype=dtype)

    def forward(self, x):
        gate_v, up_v = torch.split(self.gate_up_v_proj(x), [self.gate_rank, self.up_rank], dim=-1)
        inter = self.act_fn(self.gate_u_proj(gate_v)) * self.up_u_proj(up_v)
        return self.down_u_proj(self.down_v_proj(inter))


class SVD_LlamaAttention(nn.Module):
    """Llama attention with q/k/v/o low-rank factorizations (q,k,v down-projections fused)."""

    def __init__(self, config, ranks, layer_idx, dtype):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.is_causal = True
        bias = config.attention_bias

        self.q_rank, self.k_rank, self.v_rank, self.o_rank = ranks
        q_out = self.num_heads * self.head_dim
        kv_out = self.num_kv_heads * self.head_dim
        self.qkv_v_proj = nn.Linear(self.hidden_size, self.q_rank + self.k_rank + self.v_rank, bias=bias, dtype=dtype)
        self.q_u_proj = nn.Linear(self.q_rank, q_out, bias=bias, dtype=dtype)
        self.k_u_proj = nn.Linear(self.k_rank, kv_out, bias=bias, dtype=dtype)
        self.v_u_proj = nn.Linear(self.v_rank, kv_out, bias=bias, dtype=dtype)
        self.o_v_proj = nn.Linear(q_out, self.o_rank, bias=bias, dtype=dtype)
        self.o_u_proj = nn.Linear(self.o_rank, self.hidden_size, bias=bias, dtype=dtype)
        # RoPE fallback when the model does not pass position_embeddings
        self.rotary_emb = LlamaRotaryEmbedding(config=config)

        # Optimized low-rank KV-cache path (O1+O2+O3); state below is filled per generation and
        # reset by clear_kv_cache(). Active only when config.optimize_kv_cache is set (throughput).
        self._kv_pos = 0
        self._k_cache = None      # full-dim keys, RoPE pre-applied   [b, n_kv, max_seq, head_dim]
        self._v_cache = None      # low-rank value latents            [b, max_seq, v_rank]
        self._W_v_u = None        # per-head, GQA-expanded v up-proj  [n_heads, head_dim, v_rank]
        self._v_u_bias = None     # per-head v up-proj bias           [1, n_heads, 1, head_dim] or None
        self._pos_t = None        # O5 graph mode: write position as a tensor [1]   (decoding.py)
        self._graph_mask = None   # O5 graph mode: additive mask [1, 1, 1, max_seq] (decoding.py)

    def forward(self, hidden_states, attention_mask=None, position_ids=None,
                past_key_value=None, output_attentions=False, use_cache=False,
                cache_position=None, position_embeddings=None, **kwargs):
        bsz, q_len, _ = hidden_states.size()

        q_v, k_v, v_v = torch.split(self.qkv_v_proj(hidden_states),
                                    [self.q_rank, self.k_rank, self.v_rank], dim=-1)
        q = self.q_u_proj(q_v).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_u_proj(k_v).view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Optimized path (O1+O2+O3): generation only (use_cache), opt-in via config flag.
        # It computes its own RoPE (see below), so it must branch before the standard RoPE.
        if use_cache and getattr(self.config, "optimize_kv_cache", False):
            return self._forward_opt_kv(q, k, v_v, bsz, q_len, past_key_value)

        if position_embeddings is None:
            cos, sin = self.rotary_emb(q, position_ids)
        else:
            cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # ---- standard path: full-dim K/V through the HF KV cache (perplexity; throughput w/o opt) ----
        v = self.v_u_proj(v_v).view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        if past_key_value is not None:  # standard KV cache (generation)
            k, v = past_key_value.update(k, v, self.layer_idx,
                                         {"sin": sin, "cos": cos, "cache_position": cache_position})

        k = repeat_kv(k, self.num_kv_groups)
        v = repeat_kv(v, self.num_kv_groups)

        # same softmax(QKᵀ/√d + mask)·V as the eager matmul/softmax/matmul, via the fused SDPA
        # kernel — avoids materializing the [b, heads, q, kv] attention matrix.
        mask = attention_mask[:, :, :, : k.shape[-2]] if attention_mask is not None else None
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        out = out.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
        out = self.o_u_proj(self.o_v_proj(out))
        return out, None, past_key_value

    def _init_fused_v_weights(self):
        """Precompute the value up-projection reshaped per attention head and GQA-expanded so the
        fused-value attention can up-project in head space without per-step reshape/repeat (O3)."""
        w = self.v_u_proj.weight.detach().view(self.num_kv_heads, self.head_dim, self.v_rank)
        self._W_v_u = w.repeat_interleave(self.num_kv_groups, dim=0).contiguous()   # [n_heads, hd, r_v]
        b = self.v_u_proj.bias
        if b is not None:
            b = b.detach().view(self.num_kv_heads, self.head_dim).repeat_interleave(self.num_kv_groups, dim=0)
            self._v_u_bias = b.view(1, self.num_heads, 1, self.head_dim)
        else:
            self._v_u_bias = None

    def _forward_opt_kv(self, q, k, v_v, bsz, q_len, past_key_value):
        """O1 (RoPE pre-applied so cached keys are final) + O2 (values cached as the r_v latent; the
        v up-projection is fused into the attention output) + O3 (cache buffers pre-allocated once at
        the max sequence length, filled in place) + O4 (grouped-query scores against the unexpanded
        K cache; SDPA for the prefill chunk). Bypasses the HF KV cache; the module keeps its own
        state across decode steps. Assumes causal attention, no padding (the throughput harness feeds
        an all-ones mask). With _pos_t set (O5 graph mode, decode/q_len==1 only) the per-step state
        lives in tensors so the whole step is CUDA-graph capturable [decoding.py]."""
        graphed = self._pos_t is not None
        if graphed:
            positions = self._pos_t.view(1, 1)
        else:
            pos0 = self._kv_pos
            total = pos0 + q_len
            positions = torch.arange(pos0, total, device=q.device).unsqueeze(0)

        # O1: recompute RoPE for exactly the q_len fed tokens from the module's own position state.
        # (The HF-passed position_embeddings span the whole attention mask — wrong length here, since
        # we keep our own cache and leave HF's empty — so we cannot reuse them.)
        cos, sin = self.rotary_emb(q, positions)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if self._k_cache is None:                       # O3: allocate once per generation
            max_seq = max(int(getattr(self.config, "kv_opt_max_seq", total)), total)
            self._k_cache = torch.zeros(bsz, self.num_kv_heads, max_seq, self.head_dim,
                                        dtype=k.dtype, device=k.device)
            self._v_cache = torch.zeros(bsz, max_seq, self.v_rank,
                                        dtype=v_v.dtype, device=v_v.device)
            self._init_fused_v_weights()
        if graphed:                                     # tensor-indexed in-place writes (capturable)
            # _kv_pos is NOT advanced here: graph mode must be exited via clear_kv_cache()
            # (decoding.py guarantees this in its finally block).
            self._k_cache.index_copy_(2, self._pos_t, k)
            self._v_cache.index_copy_(1, self._pos_t, v_v)
            k_att, v_att = self._k_cache, self._v_cache  # full buffer; _graph_mask hides unwritten slots
        else:
            if total > self._k_cache.shape[2]:          # grow in pages — never silently drop writes
                ext = max(total, self._k_cache.shape[2] + 256) - self._k_cache.shape[2]
                self._k_cache = torch.cat([self._k_cache, self._k_cache.new_zeros(
                    bsz, self.num_kv_heads, ext, self.head_dim)], dim=2)
                self._v_cache = torch.cat([self._v_cache, self._v_cache.new_zeros(
                    bsz, ext, self.v_rank)], dim=1)
            self._k_cache[:, :, pos0:total, :] = k
            self._v_cache[:, pos0:total, :] = v_v       # O2: store the low-rank value latent
            self._kv_pos = total
            k_att = self._k_cache[:, :, :total, :]
            v_att = self._v_cache[:, :total, :]

        if q_len > 1:
            # Prefill chunk: fused SDPA over the (transiently) expanded K/V — peak scales with
            # q_len·head_dim instead of q_len·total. Decode keeps the O2 latent value path below.
            k_full = repeat_kv(k_att, self.num_kv_groups)                       # [b, n_heads, total, hd]
            v_full = torch.einsum("btr,hdr->bhtd", v_att, self._W_v_u)
            if self._v_u_bias is not None:
                v_full = v_full + self._v_u_bias        # softmax weights sum to 1 -> same as post-bias
            qpos = pos0 + torch.arange(q_len, device=q.device)
            kpos = torch.arange(total, device=q.device)
            mask = torch.zeros(q_len, total, dtype=q.dtype, device=q.device)
            mask.masked_fill_(kpos[None, :] > qpos[:, None], torch.finfo(q.dtype).min)
            out = F.scaled_dot_product_attention(q, k_full, v_full, attn_mask=mask)
        else:
            # O4: grouped-query scores straight against the unexpanded K cache (no per-step
            # repeat_kv copy of the cache); the 1/sqrt(d) scale is folded into q (cheaper there).
            qg = (q * self.head_dim ** -0.5).view(bsz, self.num_kv_heads, self.num_kv_groups,
                                                  q_len, self.head_dim)
            attn = torch.einsum("bkgqd,bktd->bkgqt", qg, k_att).reshape(bsz, self.num_heads, q_len, -1)
            if graphed:
                attn = attn + self._graph_mask          # -inf beyond the current position
            attn = torch.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
            # O2: weighted sum stays in r_v space, then a single fused per-head up-projection.
            weighted = torch.einsum("bhqk,bkr->bhqr", attn, v_att)
            out = torch.einsum("bhqr,hdr->bhqd", weighted, self._W_v_u)         # [b, n_heads, q, hd]
            if self._v_u_bias is not None:
                out = out + self._v_u_bias
        out = out.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
        out = self.o_u_proj(self.o_v_proj(out))
        return out, None, past_key_value               # return the (untouched) HF cache, so generate keeps threading it

    def clear_kv_cache(self):
        """Reset the optimized KV-cache state. Call before each generation."""
        self._kv_pos = 0
        self._k_cache = None
        self._v_cache = None
        self._W_v_u = None
        self._v_u_bias = None
        self._pos_t = None
        self._graph_mask = None


def clear_all_kv_caches(model):
    """Reset every layer's optimized KV-cache state (no-op for plain / non-SVD attention)."""
    for layer in getattr(model.model, "layers", []):
        attn = getattr(layer, "self_attn", None)
        if attn is not None and hasattr(attn, "clear_kv_cache"):
            attn.clear_kv_cache()
