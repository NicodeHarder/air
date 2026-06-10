"""AIR entry point.

Pipeline: load model -> merged forward-backward analysis (profiling S + influence I in ONE
pass) -> layer-by-layer compression (SVD-LLM(W) init + influence-aware ALS) -> evaluation.
"""

import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # less fragmentation

import sys
import gc
import datetime


class _Tee:
    """Mirror everything written to stdout/stderr into a log file as well."""

    def __init__(self, path, stream):
        self._f = open(path, "a", buffering=1)
        self._stream = stream

    def write(self, s):
        self._stream.write(s)
        self._f.write(s)

    def flush(self):
        self._stream.flush()
        self._f.flush()

    def isatty(self):
        return self._stream.isatty()

    def fileno(self):
        return self._stream.fileno()


def _log_gpu_status(device):
    """Heads-up: list the visible GPU(s) and their CURRENT memory usage (driver-level, so it
    also reflects other processes already on the card) + utilization."""
    import torch
    if device == "cpu" or not torch.cuda.is_available():
        print("GPU status: CUDA not available — running on CPU.")
        return
    n = torch.cuda.device_count()
    gb = 1024 ** 3
    print(f"GPU status ({n} visible):")
    for i in range(n):
        free, total = torch.cuda.mem_get_info(i)
        used = total - free
        try:
            util = f"{torch.cuda.utilization(i):>3d}%"
        except Exception:
            util = "  ?"
        print(f"  [{i}] {torch.cuda.get_device_name(i)} | {total / gb:4.1f} GB total | "
              f"{used / gb:5.1f} used / {free / gb:5.1f} free | util {util}")


def _start_logging(args):
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(args.model_dir, f"run_{ts}.log")
    sys.stdout = _Tee(log_path, sys.__stdout__)
    sys.stderr = _Tee(log_path, sys.__stderr__)
    print(f"=== AIR run {ts} | logging to {log_path} ===")
    for k, v in sorted(vars(args).items()):
        print(f"  {k}: {v}")
    print("=" * 60)
    _log_gpu_status(args.device)
    print("=" * 60)

# air/ on sys.path (flat intra-package imports) + repo root (vendored lxt/ one level up).
_air_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(_air_dir)
sys.path.insert(0, _repo_root)
sys.path.insert(0, _air_dir)
os.chdir(_repo_root)

import torch

from config import get_config
from modeling import load_model_and_tokenizer, num_layers
from data import get_calibration_data
from analysis import compute_analysis, analysis_cached
from compression import compress_model, build_compressed_skeleton
from efficiency import (get_baseline_metrics, compressed_metrics,
                        report_efficiency, report_quality)

import transformers
transformers.logging.set_verbosity_error()   # silence benign HF tokenizer/generation warnings


def main():
    args = get_config()
    _start_logging(args)
    torch.manual_seed(args.seed)
    print(f"\nAIR | model={args.model_name} method={args.compression_method} "
          f"rate={args.parameter_rate:.2f} influence={args.use_influence} als={args.run_als}")

    want_ppl = args.EVALUATE
    want_thr = (args.MEASURE_THROUGHPUT or args.MEASURE_TIME) and not args.PERPLEXITY_ONLY

    # Baseline efficiency + quality metrics first (before any lxt wrapping), cached per model.
    base = get_baseline_metrics(args, want_ppl, want_thr)

    if args.SVD:
        compressed_path = os.path.join(args.compressed_dir, "model_state_dict.pt")
        if (not args.overwrite_svd) and os.path.exists(compressed_path):
            # --- Cache hit: load the compressed model; skip analysis + compression entirely ---
            model, tokenizer = load_model_and_tokenizer(args, lxt=False)
            build_compressed_skeleton(model, args)       # rebuild low-rank architecture (right ranks)
            model.load_state_dict(torch.load(compressed_path, map_location="cpu"))
            print(f"\nLoaded cached compressed model <- {compressed_path}")
        else:
            # --- Phase 1: merged forward-backward analysis (multi-GPU). The lxt model is needed
            #     only for the 'relevances' signal; weightxgrad/fisher use a plain model. ---
            if not analysis_cached(args, num_layers(args)):
                a_model, tokenizer = load_model_and_tokenizer(args, lxt=args.use_lrp, multi_gpu=True)
                calib = get_calibration_data(args, tokenizer)
                compute_analysis(a_model, calib, args)        # one pass: profiling S (+ influence I) -> disk
                del a_model, calib                            # free the analysis model + graph from GPU
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                print("\nAnalysis cache present — skipping analysis phase.")

            # --- Phase 2: freshly load a plain model for compression + evaluation ---
            model, tokenizer = load_model_and_tokenizer(args, lxt=False)
            compress_model(model, args)                  # SVD-LLM(W) init + influence-aware ALS

            if args.SAVE_COMPRESSED_MODEL:
                torch.save(model.state_dict(), compressed_path)
                print(f"\nSaved compressed model -> {compressed_path}")
    else:
        model, tokenizer = load_model_and_tokenizer(args, lxt=False)

    comp = compressed_metrics(model, tokenizer, args, want_ppl, want_thr,
                              thr_batch=base.get("thr_batch"))   # same throughput batch as baseline
    report_efficiency(base, comp)        # block 1: params, footprint, FLOPs, peak GPU, latency
    report_quality(base, comp, args)     # block 2: perplexity

    print("\n... AIR run complete.")


if __name__ == "__main__":
    main()
