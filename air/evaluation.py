"""Measurement primitives: WikiText-2 perplexity (quality) and generation
throughput / peak GPU memory (dynamic efficiency).

These return raw numbers; the baseline-vs-compressed reporting lives in efficiency.py.
"""

import gc
import time
import numpy as np
import torch


def _reset_peak():
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(i)


def _peak_gb():
    """Peak allocated GPU memory summed over all visible devices (model may be split)."""
    if not torch.cuda.is_available():
        return 0.0
    return sum(torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())) / 1e9


def _pad_id(model):
    """A single valid pad token id (modern Llama/Qwen store eos_token_id as a list)."""
    eos = getattr(model.config, "eos_token_id", None)
    if isinstance(eos, (list, tuple)):
        eos = eos[0] if eos else None
    return eos if isinstance(eos, int) else 0


@torch.no_grad()
def perplexity(model, eval_batches, device):
    model.eval()
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    nlls = []
    for batch in eval_batches:
        batch = batch.to(device)
        logits = model(batch, use_cache=False).logits
        shift_logits = logits[:, :-1, :].reshape(-1, logits.size(-1))
        shift_labels = batch[:, 1:].reshape(-1).to(shift_logits.device)
        nlls.append(loss_fct(shift_logits, shift_labels).float().cpu())   # per-token NLLs
        del logits, batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    # Filter non-finite tokens (not whole batches) so baseline and compressed are scored over the
    # same token set; report how many were excluded. All non-finite -> perplexity is infinite.
    all_nll = torch.cat(nlls)
    finite = torch.isfinite(all_nll)
    dropped = int((~finite).sum())
    if dropped:
        print(f"[perplexity] excluded {dropped}/{all_nll.numel()} non-finite token losses")
    if not bool(finite.any()):
        return float("inf")
    return float(np.exp(all_nll[finite].mean().item()))


def _empty_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _kv_bytes_per_token(model):
    """KV-cache bytes per generated token per sequence: 2(K,V) · layers · kv_heads · head_dim · 2 (bf16)."""
    c = model.config
    n_kv = getattr(c, "num_key_value_heads", None) or c.num_attention_heads
    head_dim = getattr(c, "head_dim", None) or (c.hidden_size // c.num_attention_heads)
    return 2 * c.num_hidden_layers * n_kv * head_dim * 2


def _fit_batch(model, args):
    """Largest throughput batch that should fit the free VRAM on the model's device (proactive).
    Caps the configured batch using a KV-cache estimate; the reactive halving below is the net."""
    bs = args.throughput_batch_size
    dev = next(model.parameters()).device
    if dev.type != "cuda":
        return bs
    free, _ = torch.cuda.mem_get_info(dev.index or 0)
    total_seq = args.throughput_prompt_len + args.throughput_n_generate
    kv_per_seq = max(_kv_bytes_per_token(model) * total_seq, 1)
    cap = max(1, int(free * 0.8 / kv_per_seq))           # 80% of free for KV; rest for activations
    if cap < bs:
        print(f"[throughput] batch capped {bs} -> {cap} for {free / 1e9:.1f} GB free")
    return min(bs, cap)


@torch.no_grad()
def measure_throughput(model, args, batch=None):
    """Greedy-generate and return (micro-seconds/token, peak GPU memory GB, batch actually used).

    A warmup generation precedes the timed run so both the baseline (measured first, cold) and the
    compressed model are timed at steady state. The generation batch is auto-halved on CUDA OOM
    until it fits (down to 1) — a batch-N generation's KV cache can exceed VRAM even when the model
    itself fits. Pass `batch` to force the same size on baseline and compressed (fair comparison)."""
    dev = next(model.parameters()).device
    vocab = model.config.vocab_size
    n = args.throughput_n_generate
    pad = _pad_id(model)
    model.eval()

    # Optimized low-rank KV-cache path (O1+O2+O3). The flag is read by the SVD attention modules
    # (inert for the plain baseline model); buffers are pre-allocated to the full generation length.
    opt_kv = bool(getattr(args, "OPTIMIZE_KV_CACHE", False))
    model.config.optimize_kv_cache = opt_kv
    model.config.kv_opt_max_seq = args.throughput_prompt_len + args.throughput_n_generate
    clear_caches = None
    if opt_kv:
        from layers.llama import clear_all_kv_caches as clear_caches

    # O5 (--CUDA_GRAPH): replay the decode step as one captured graph. Only the compressed model
    # carries the opt modules, so the flag is inert for the plain baseline.
    graphed = False
    if opt_kv and bool(getattr(args, "CUDA_GRAPH", False)) and dev.type == "cuda":
        from decoding import graph_generate, opt_attns
        graphed = len(opt_attns(model)) > 0

    bs = batch if batch is not None else _fit_batch(model, args)   # proactive cap (None=baseline)
    start_bs = bs
    while bs >= 1:
        try:
            prompt = torch.randint(0, vocab, (bs, args.throughput_prompt_len), device=dev)
            attn = torch.ones_like(prompt)               # no padding -> explicit mask (silences warning)

            def gen(k):
                # min_new_tokens == max_new_tokens: always emit exactly k tokens (no early EOS),
                # so the token count is exact for the timing.
                if clear_caches is not None:             # reset the internal cache before each run
                    clear_caches(model)
                if graphed and k >= 4:                   # O5: same greedy exact-k contract
                    return graph_generate(model, prompt, k)
                return model.generate(prompt, attention_mask=attn, max_new_tokens=k, min_new_tokens=k,
                                      do_sample=False, use_cache=True, pad_token_id=pad)

            gen(min(8, n))                               # warmup (discarded)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            _reset_peak()
            t0 = time.time()
            out = gen(n)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            dt = time.time() - t0
            new_tokens = (out.shape[1] - prompt.shape[1]) * bs       # == n * bs
            if bs != start_bs:
                print(f"[throughput] batch reduced to {bs} after OOM")
            return 1e6 * dt / max(new_tokens, 1), _peak_gb(), bs
        except torch.cuda.OutOfMemoryError:
            prompt = attn = out = None
            _empty_cache()
            if bs == 1:
                print("[throughput] OOM even at batch 1 — skipping throughput")
                return float("nan"), 0.0, 0
            bs //= 2


def dynamic_metrics(model, tokenizer, args, want_ppl, want_thr, thr_batch=None):
    """{perplexity?, latency_us?, peak_mem_gb?, thr_batch?} for `model` (whichever are requested).
    `thr_batch` forces the throughput batch size (to match baseline and compressed)."""
    from data import get_eval_batches
    out = {}
    if want_ppl:
        dev = next(model.parameters()).device
        out["perplexity"] = perplexity(model, get_eval_batches(args, tokenizer), dev)
        _empty_cache()                                   # free perplexity logits before throughput
    if want_thr:
        us, peak, bs = measure_throughput(model, args, batch=thr_batch)
        out["latency_us"] = us
        out["peak_mem_gb"] = peak
        out["thr_batch"] = bs
    return out
