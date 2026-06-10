"""CUDA-graph greedy decoding for the optimized KV-cache path (O5, --CUDA_GRAPH).

At small batch the decode step is dominated by kernel-launch + Python overhead (~10 GEMV
launches per layer per token, plus the HF generate loop). The O1-O3 path makes the step
static: the O3 buffers are pre-allocated, and with the per-step attention state held in
tensors (layers/llama.py graph mode) one full step — embeddings -> all layers -> lm_head ->
argmax -> position/mask advance -> token feedback — is captured once in a CUDA graph and
replayed per token with a single launch.

Function-preserving: the captured step runs exactly the optimized-path kernels; slots beyond
the current position are masked to -inf, so they contribute exact zeros after softmax.
"""

import torch


def opt_attns(model):
    """The SVD attention modules carrying the optimized KV-cache state (empty for plain models)."""
    layers = getattr(getattr(model, "model", None), "layers", [])
    return [l.self_attn for l in layers if hasattr(l.self_attn, "clear_kv_cache")]


@torch.no_grad()
def graph_generate(model, prompt, n_new):
    """Greedy-generate exactly `n_new` tokens after `prompt` [b, p]; returns [b, p + n_new]
    (the contract of model.generate(min_new_tokens=max_new_tokens=n_new, do_sample=False)).
    Requires config.optimize_kv_cache, CUDA, no padding, and n_new >= 4 (2 warmup + 1 capture)."""
    from layers.llama import clear_all_kv_caches

    attns = opt_attns(model)
    dev = prompt.device
    b, p = prompt.shape
    dtype = next(model.parameters()).dtype
    model.config.kv_opt_max_seq = p + n_new
    clear_all_kv_caches(model)

    tokens = torch.empty(b, n_new, dtype=prompt.dtype, device=dev)
    logits = model(prompt, use_cache=True).logits          # prefill (allocates the O3 buffers)
    tokens[:, 0] = logits[:, -1].argmax(-1)

    # Switch the attention modules to tensor-held per-step state (graph mode).
    pos = torch.tensor([p], device=dev)                    # cache slot the next forward writes to
    step = torch.tensor([1], device=dev)                   # write index into `tokens`
    mask = torch.full((1, 1, 1, p + n_new), torch.finfo(dtype).min, dtype=dtype, device=dev)
    mask[..., :p] = 0
    for a in attns:
        a._pos_t, a._graph_mask = pos, mask
    inp = tokens[:, :1].clone()                            # static input buffer [b, 1]

    def decode_step():
        mask.index_fill_(3, pos, 0)                        # unmask the slot written this step
        nxt = model(inp, use_cache=True).logits[:, -1].argmax(-1)
        tokens.index_copy_(1, step, nxt.unsqueeze(1))
        inp.copy_(nxt.unsqueeze(1))
        pos.add_(1)
        step.add_(1)

    try:
        s = torch.cuda.Stream()                            # canonical capture recipe: warm up on a
        s.wait_stream(torch.cuda.current_stream())         # side stream, then record one step
        with torch.cuda.stream(s):
            decode_step()
            decode_step()                                  # 2 real (eager) steps -> tokens 1, 2
        torch.cuda.current_stream().wait_stream(s)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            decode_step()                                  # recorded, not executed
        for _ in range(n_new - 3):                         # tokens 3 .. n_new-1
            graph.replay()
        torch.cuda.synchronize()
    finally:
        for a in attns:
            a._pos_t = a._graph_mask = None
        clear_all_kv_caches(model)
    return torch.cat([prompt, tokens], dim=1)
