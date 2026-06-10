"""Merged forward-backward analysis (the elegant single-pass core of this codebase).

For each calibration sample we do ONE forward + ONE backward, capturing simultaneously:
  * forward  -> the profiling statistic  S[layer,name] = sum_x X X^T  (X = input to the linear),
    via forward hooks;
  * backward -> the influence  I[layer,name]. The paper is agnostic to this backward signal
    (--importance_metrics):
      - 'relevances'  : AttnLRP ε-rule on the lxt model (stores weight.relevance);
      - 'weightxgrad' : standard LM-loss backward, I = | ∂L/∂W · W |  (Weight×Gradient);
      - 'fisher'      : standard LM-loss backward, I = (∂L/∂W)²       (diagonal empirical Fisher).
    Only 'relevances' needs the lxt model; the gradient signals use a plain model.
After the pass we iteratively rescale each per-layer influence matrix (the under-specified but
crucial normalization) and cache S and I per layer to disk.

Memory: per-sample graph (one sample at a time) + immediate CPU offload of the influence + cache
emptying; the same per-sample-backward choreography the original code uses to fit large models.
"""

import gc
import os
import math
import torch
from tqdm import tqdm

from lxt.rules import EpsilonRule
from modeling import get_attnlrp, get_decoder_layers

WEIGHT_SUFFIXES = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}

# q/k/v (and gate/up) read the same layer input, so their XXᵀ are bit-identical: accumulate
# one Gram matrix per input group and alias the others at save time (aliases share storage,
# which torch.save stores once — smaller caches, ~40% fewer Gram GEMMs + CPU transfers).
SHARED_XTX = {"k_proj": "q_proj", "v_proj": "q_proj", "up_proj": "gate_proj"}


def normalize_influence(mat, scale=100.0, tol=2e-2, max_iter=10, stabilizer=1e-12):
    """Iteratively rescale an influence matrix toward [mean=1, max=scale]. Crucial for ALS
    stability; barely described in the paper but reproduced from the reference implementation."""
    m = mat.abs().double()
    dev_mean = dev_max = 1.0
    it = 0
    while (dev_mean > tol or dev_max > tol) and it < max_iter:
        m = m - m.min()
        m = m / m.max()
        m = m + stabilizer
        m = m / m.mean()
        x = math.log(scale) / math.log(m.max().item())
        m = m ** x
        m = m / m.mean()
        dev_mean = abs(1.0 - m.mean().item())
        dev_max = abs(scale - m.max().item()) / scale
        it += 1
    return m.float()


def _targets(model, use_lrp):
    """{full_name: module} for the q/k/v/o/gate/up/down projections in the decoder layers.
    use_lrp -> the lxt EpsilonRule wrappers (relevance read from .module.weight);
    else    -> the plain Linears (gradient read from .weight.grad)."""
    out = {}
    for name, mod in model.named_modules():
        if name.split(".")[-1] in WEIGHT_SUFFIXES and ".layers." in name:
            if use_lrp and isinstance(mod, EpsilonRule):
                out[name] = mod
            elif not use_lrp and isinstance(mod, torch.nn.Linear):
                out[name] = mod
    return out


def _accumulate(Inf, name, contrib):
    Inf[name] = contrib if Inf[name] is None else Inf[name] + contrib


def _lrp_backward(model, input_ids, targets, Inf):
    """AttnLRP ε-rule backward: lxt stores relevance = weight_grad · weight on each linear."""
    embeds = model.get_input_embeddings()(input_ids).detach().requires_grad_(True)
    with torch.enable_grad():
        logits = model(inputs_embeds=embeds, use_cache=False).logits
        target = logits[0, -1, :].max()                  # top logit of last token (AttnLRP init)
        torch.autograd.grad(target, embeds, grad_outputs=target.detach(), retain_graph=False)
    for n, w in targets.items():
        rel = getattr(w.module.weight, "relevance", None)
        if rel is not None:
            _accumulate(Inf, n, rel.detach().abs().float().cpu())
            del w.module.weight.relevance
    del embeds, logits, target


def _grad_backward(model, input_ids, targets, Inf, metric):
    """Standard LM cross-entropy backward: weightxgrad I = |∂L/∂W · W|, fisher I = (∂L/∂W)²."""
    with torch.enable_grad():
        logits = model(input_ids, use_cache=False).logits
        shift = logits[:, :-1, :].reshape(-1, logits.size(-1))
        labels = input_ids[:, 1:].reshape(-1).to(shift.device)
        loss = torch.nn.functional.cross_entropy(shift, labels)
        model.zero_grad(set_to_none=True)
        loss.backward()
    for n, lin in targets.items():
        g = lin.weight.grad
        if g is None:
            continue
        c = g.detach().pow(2) if metric == "fisher" else (g.detach() * lin.weight.detach()).abs()
        _accumulate(Inf, n, c.float().cpu())
    del logits, loss


def _layer_index(name):
    parts = name.split(".")
    return int(parts[parts.index("layers") + 1])


def analysis_cached(args, n_layers):
    if args.overwrite_analysis:
        return False
    return all(os.path.exists(os.path.join(args.analysis_dir, f"S_layer_{i}.pt")) and
               (not args.use_influence or os.path.exists(os.path.join(args.analysis_dir, f"I_layer_{i}.pt")))
               for i in range(n_layers))


def compute_analysis(model, calib_data, args):
    """Run the merged pass and cache per-layer profiling S_pre (+ influence I) to disk."""
    n_layers = len(get_decoder_layers(model))
    if analysis_cached(args, n_layers):
        print("Analysis cache present — skipping merged pass.")
        return

    metric = args.importance_metrics
    use_lrp = args.use_lrp                # = use_influence and metric == "relevances"
    if args.use_influence:
        names = {"relevances": "AttnLRP ε-rule", "weightxgrad": "Weight×Gradient",
                 "fisher": "diagonal empirical Fisher"}
        infl = f" + influence I ({names[metric]} backward)"
    else:
        infl = ""
    print(f"\nStarting analysis: merged forward-backward over {len(calib_data)} calibration samples "
          f"(seqlen {args.relevance_subset_seq_len}) -> profiling matrix S (forward XXᵀ){infl}, "
          f"cached per layer ...")

    if use_lrp:
        attnlrp = get_attnlrp(args.model_name)
        attnlrp.register(model)
    try:
        targets = _targets(model, use_lrp)
        embed_dev = model.get_input_embeddings().weight.device

        XtX = {n: None for n in targets if n.split(".")[-1] not in SHARED_XTX}  # sum X X^T (CPU, fp64)
        Inf = {n: None for n in targets} if args.use_influence else None  # sum influence (CPU, fp32)

        def xtx_hook(name):
            def hook(_m, inp, _out):
                x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
                contrib = (x.t() @ x).cpu().double()     # fp32 over PCIe, fp64 accumulation (exact upcast)
                XtX[name] = contrib if XtX[name] is None else XtX[name] + contrib
            return hook

        handles = [w.register_forward_hook(xtx_hook(n)) for n, w in targets.items() if n in XtX]

        for input_ids in tqdm(calib_data, desc="  analysis", unit="sample"):
            input_ids = input_ids.to(embed_dev)
            if not args.use_influence:
                with torch.no_grad():
                    model(input_ids, use_cache=False)            # profiling S only (svd_llm)
            elif use_lrp:
                _lrp_backward(model, input_ids, targets, Inf)
            else:
                _grad_backward(model, input_ids, targets, Inf, metric)
            model.zero_grad(set_to_none=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        for h in handles:
            h.remove()
    finally:
        if use_lrp:
            attnlrp.remove()   # restore plain linears for compression

    # ---- normalization + per-layer caching (S_pre always; I normalized if influence) ----
    n = len(calib_data)
    if args.use_influence and args.RELEVANCE_NORMALIZATION:
        print(f"\nForward-backward pass complete — normalizing influence per weight "
              f"(iterative rescale toward mean 1 / max {args.relevance_normalization_max:g}), "
              f"then caching S + I per layer ...")
    elif args.use_influence:
        print("\nForward-backward pass complete — caching S + raw influence per layer "
              "(normalization off) ...")
    else:
        print("\nForward-backward pass complete — caching profiling matrix S per layer ...")
    by_layer_S, by_layer_I = {}, {}
    for name in targets:
        i, suf = _layer_index(name), name.split(".")[-1]
        src = name if suf not in SHARED_XTX else name.rsplit(".", 1)[0] + "." + SHARED_XTX[suf]
        by_layer_S.setdefault(i, {})[suf] = XtX[src]
        if args.use_influence:
            norm = normalize_influence(Inf[name] / n, scale=args.relevance_normalization_max) \
                   if args.RELEVANCE_NORMALIZATION else (Inf[name] / n)
            by_layer_I.setdefault(i, {})[suf] = norm
    for i in by_layer_S:
        torch.save(by_layer_S[i], os.path.join(args.analysis_dir, f"S_layer_{i}.pt"))
        if args.use_influence:
            torch.save(by_layer_I[i], os.path.join(args.analysis_dir, f"I_layer_{i}.pt"))
    print(f"\nAnalysis done: cached S{' + I' if args.use_influence else ''} for {len(by_layer_S)} layers.")
