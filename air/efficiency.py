"""Efficiency + quality reporting: baseline vs compressed, absolute values and relative %.

Two blocks are printed at the end of a run:
  * Efficiency — parameters, footprint (storage), forward FLOPs/token (static counts) plus
    peak GPU memory and latency/token (measured by generation).
  * Quality — WikiText-2 perplexity.

For every metric we report the uncompressed BASELINE value, the COMPRESSED value, and
rel % = compressed / baseline.

Baseline metrics are measured once and cached per model at <model_dir>/baseline_metrics.json:
  * the static counts (params/footprint/FLOPs) depend only on the model and come from a
    zero-memory meta-device copy (even a 30B baseline costs nothing);
  * the dynamic ones (perplexity, latency, peak memory) depend on the eval/throughput config,
    so they are stored under a config sub-key and measured by loading the uncompressed model.
Everything is reused on later runs (including compressed-cache hits) unless --OVERWRITE_CACHE.
"""

import gc
import json
import math
import os

import torch
from torch import nn

from compression import _human


def _dtype_bytes(args):
    return 4 if args.tiny else 2          # tiny build is fp32; everything else loads bf16


def static_metrics(model, args):
    """{params, footprint_bytes, flops_per_token} for `model` as it currently is."""
    n_params = sum(p.numel() for p in model.parameters())
    flops_tok = sum(2 * m.weight.numel() for m in model.modules() if isinstance(m, nn.Linear))
    return {"params": int(n_params),
            "footprint_bytes": int(n_params * _dtype_bytes(args)),
            "flops_per_token": int(flops_tok)}


def _ppl_sig(args):
    return f"{args.evaluation_perplexity_data}-seq{args.eval_seqlen}-bs{args.batch_size}"


def _thr_sig(args):
    # latency + peak memory depend on how the model is sharded, i.e. the visible GPU count.
    n_gpu = torch.cuda.device_count() if (torch.cuda.is_available() and args.device != "cpu") else 0
    return (f"bs{args.throughput_batch_size}-gen{args.throughput_n_generate}"
            f"-p{args.throughput_prompt_len}-gpu{n_gpu}")


def get_baseline_metrics(args, want_ppl, want_thr):
    """Baseline (uncompressed) metrics for the current config, measured-or-cached. See module doc."""
    path = os.path.join(args.model_dir, "baseline_metrics.json")
    data = {}
    if os.path.exists(path):                              # always load, so --OVERWRITE_CACHE only
        with open(path) as f:                            # refreshes THIS config's keys, not others'
            data = json.load(f)

    ppl_sig, thr_sig = _ppl_sig(args), _thr_sig(args)
    ow = args.OVERWRITE_CACHE
    need_static = ow or not all(k in data for k in ("params", "footprint_bytes", "flops_per_token"))
    need_ppl = want_ppl and (ow or ppl_sig not in data.get("perplexity", {}))
    need_thr = want_thr and (ow or thr_sig not in data.get("throughput", {}))

    measured = need_ppl or need_thr
    if measured:                                             # dynamic -> need the real model
        from modeling import load_model_and_tokenizer
        from evaluation import dynamic_metrics
        print("\nMeasuring baseline (uncompressed) metrics ...")
        model, tokenizer = load_model_and_tokenizer(args, lxt=False)
        if need_static:
            data.update(static_metrics(model, args))
            need_static = False
        dyn = dynamic_metrics(model, tokenizer, args, need_ppl, need_thr)
        if need_ppl:
            data.setdefault("perplexity", {})[ppl_sig] = dyn["perplexity"]
        if need_thr:
            data.setdefault("throughput", {})[thr_sig] = {
                "latency_us": dyn["latency_us"], "peak_mem_gb": dyn["peak_mem_gb"],
                "thr_batch": dyn["thr_batch"]}
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if need_static:                                          # static only -> free meta-device copy
        m = _baseline_arch(args)
        data.update(static_metrics(m, args))
        del m

    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"{'' if measured else chr(10)}Baseline metrics cached -> {path}")

    out = {k: data[k] for k in ("params", "footprint_bytes", "flops_per_token")}
    if want_ppl and ppl_sig in data.get("perplexity", {}):
        out["perplexity"] = data["perplexity"][ppl_sig]
    if want_thr and thr_sig in data.get("throughput", {}):
        out.update(data["throughput"][thr_sig])
    return out


def compressed_metrics(model, tokenizer, args, want_ppl, want_thr, thr_batch=None):
    """Static + dynamic metrics for the compressed model. `thr_batch` (the batch the baseline
    throughput actually fit at) is reused so baseline and compressed are timed at the same batch."""
    from evaluation import dynamic_metrics
    out = static_metrics(model, args)
    out.update(dynamic_metrics(model, tokenizer, args, want_ppl, want_thr, thr_batch=thr_batch))
    return out


def _baseline_arch(args):
    """An uninstantiated (meta-device) baseline model — for counting params/FLOPs at ~no cost."""
    from accelerate import init_empty_weights
    if args.tiny:
        from transformers import LlamaForCausalLM
        from modeling import tiny_config
        with init_empty_weights():
            return LlamaForCausalLM(tiny_config())
    from transformers import AutoModelForCausalLM
    from modeling import _config_for
    cfg, _fam, _cache = _config_for(args)
    with init_empty_weights():
        return AutoModelForCausalLM.from_config(cfg)


def _fmt_bytes(b):
    for unit, div in (("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if b >= div:
            return f"{b / div:.2f} {unit}"
    return f"{b} B"


def _fmt_flops(f):
    for unit, div in (("TFLOP", 1e12), ("GFLOP", 1e9), ("MFLOP", 1e6), ("KFLOP", 1e3)):
        if f >= div:
            return f"{f / div:.2f} {unit}"
    return f"{f} FLOP"


def _rel(b, c):
    return 100.0 * c / b if b else float("nan")


def _ok(x):
    return x is not None and not (isinstance(x, float) and math.isnan(x))


def _print_table(title, base, comp, rows):
    print(f"\n=== {title} (baseline -> compressed | rel % = compressed/baseline) ===")
    print(f"  {'metric':24s} {'baseline':>14s} {'compressed':>14s} {'rel %':>8s}")
    for label, key, fmt in rows:
        b, c = base.get(key), comp.get(key)
        if not (_ok(b) and _ok(c)) or b == 0:           # skip missing / nan (not measured) / 0-baseline
            continue
        print(f"  {label:24s} {fmt(b):>14s} {fmt(c):>14s} {_rel(b, c):7.1f}%")


def report_efficiency(base, comp):
    _print_table("Efficiency", base, comp, [
        ("Parameters", "params", _human),
        ("Footprint (storage)", "footprint_bytes", _fmt_bytes),
        ("Forward FLOPs / token", "flops_per_token", _fmt_flops),
        ("Peak GPU memory", "peak_mem_gb", lambda x: f"{x:.2f} GB"),
        ("Latency / token", "latency_us", lambda x: f"{x:.2f} us"),
    ])
    bs = comp.get("thr_batch")
    if bs:
        print(f"  (throughput measured at batch {bs}; baseline and compressed use the same batch)")


def report_quality(base, comp, args):
    _print_table("Quality", base, comp, [
        (f"Perplexity ({args.evaluation_perplexity_data})", "perplexity", lambda x: f"{x:.4f}"),
    ])
