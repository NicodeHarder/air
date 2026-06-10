"""Command-line configuration for AIR.

Only the knobs needed to run the launch.json pipeline (compress a Llama-family
model, evaluate WikiText-2 perplexity, optional throughput). No experiment
tracking / no extensive logging.
"""

import argparse
import os

import torch

# Load .vscode/.env (repo root) if present, so CLI runs see HF_TOKEN / RESULTS_DIR.
try:
    from dotenv import load_dotenv
    _repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _env = os.path.join(_repo_root, ".vscode", ".env")
    if os.path.exists(_env):
        load_dotenv(_env)
except ImportError:
    pass


def get_config(argv=None):
    p = argparse.ArgumentParser(
        "AIR: Activation- and Influence-Aware Ranks (SVD compression for LLMs)"
    )

    # ---- model ----
    p.add_argument("--model_name", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                   help="HF model id (Llama / Mistral / Qwen family)")
    p.add_argument("--tiny", action="store_true",
                   help="Use a tiny randomly-initialized Llama (smoke test; no download)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--hf_token", default=os.environ.get("HF_TOKEN"))
    p.add_argument("--results_dir", default=os.environ.get("RESULTS_DIR", "./results"))
    p.add_argument("--cache_dir", default=os.environ.get("HF_HOME"),
                   help="HF download cache for BOTH model weights and datasets (shared, "
                        "model-agnostic). Defaults to $HF_HOME, else <results_dir>/hf_cache.")
    p.add_argument("--seed", type=int, default=3)

    # ---- compression ----
    p.add_argument("--compression_method", default="air",
                   help="'air' = SVD-LLM(W) init + influence-aware ALS; "
                        "'svd_llm' = activation-aware init only (no ALS). 'rr' is a legacy alias for 'air'.")
    p.add_argument("--parameter_rate_idx_list", type=float, default=60.0,
                   help="Target parameter rate in PERCENT (e.g. 60 = keep 60%% of params).")
    p.add_argument("--parameter_rate", type=float, default=None,
                   help="Target rate in [0,1]; overrides --parameter_rate_idx_list if given.")
    p.add_argument("--SVD", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--SAVE_COMPRESSED_MODEL", action="store_true")

    # ---- AIR hyper-parameters (paper defaults) ----
    p.add_argument("--relevance_impact", type=float, default=2.0,
                   help="Influence weighting strength delta in the (1 + delta*I) ALS loss.")
    p.add_argument("--activation_impact", type=float, default=1.0,
                   help="The all-ones activation anchor in (activation_impact + delta*I).")
    p.add_argument("--n_sweeps", type=int, default=1, help="Number of ALS sweeps over the ranks.")
    p.add_argument("--importance_metrics", default="relevances",
                   choices=["relevances", "weightxgrad", "fisher"],
                   help="Backward influence signal I (the paper is agnostic to it): 'relevances' "
                        "= AttnLRP ε-rule (lxt); 'weightxgrad' = Weight×Gradient |∂L/∂W·W|; "
                        "'fisher' = diagonal empirical Fisher (∂L/∂W)². The latter two need no lxt.")
    p.add_argument("--RELEVANCE_NORMALIZATION", action=argparse.BooleanOptionalAction, default=True,
                   help="Iteratively rescale each per-layer influence matrix toward [mean=1, max=scale].")
    p.add_argument("--relevance_normalization_max", type=float, default=100.0)
    p.add_argument("--lrp_epsilon", type=float, default=1e-6)

    # ---- calibration ----
    p.add_argument("--calibration_data", default="wikitext2", help="wikitext2 | c4 | ptb")
    p.add_argument("--n_samples_calibration", type=int, default=256)
    p.add_argument("--relevance_subset_seq_len", type=int, default=2048,
                   help="Calibration sequence length (forward+backward analysis).")

    # ---- cache control ----
    # By default every artifact (profiling S, influence I, compressed model) is loaded from a
    # config-specific cache when present and only computed when missing. The OVERWRITE_* flags
    # force recomputation; --OVERWRITE_CACHE forces recomputation of everything.
    p.add_argument("--OVERWRITE_CACHE", action="store_true",
                   help="Ignore all caches and recompute everything (analysis S/I + compressed model).")
    p.add_argument("--COMPUTE_RELEVANCES", action="store_true",
                   help="(Accepted for compatibility; the merged analysis computes S and I as needed.)")
    p.add_argument("--OVERWRITE_ANALYSIS", action="store_true", help="Recompute the cached profiling S + influence I.")
    p.add_argument("--OVERWRITE_RELEVANCES", action="store_true", help="Alias for --OVERWRITE_ANALYSIS.")
    p.add_argument("--OVERWRITE_PROFILING_MATRIX", action="store_true", help="Alias for --OVERWRITE_ANALYSIS.")
    p.add_argument("--OVERWRITE_SVD", action="store_true", help="Recompute (and re-cache) the compressed model.")

    # ---- evaluation ----
    p.add_argument("--EVALUATE", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--evaluation_perplexity_data", default="wikitext2", help="wikitext2 | c4")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--eval_seqlen", type=int, default=2048)
    p.add_argument("--PERPLEXITY_ONLY", action="store_true",
                   help="Skip the throughput / latency / peak-memory pass.")
    p.add_argument("--MEASURE_THROUGHPUT", action="store_true")
    p.add_argument("--MEASURE_TIME", action="store_true")
    p.add_argument("--OPTIMIZE_KV_CACHE", action="store_true",
                   help="Use the low-rank KV-cache attention path (throughput).")
    p.add_argument("--CUDA_GRAPH", action="store_true",
                   help="O5: capture the optimized decode step in a CUDA graph and replay it per "
                        "token (greedy; needs --OPTIMIZE_KV_CACHE on CUDA; biggest win at small batch).")
    p.add_argument("--throughput_batch_size", type=int, default=64)
    p.add_argument("--throughput_n_generate", type=int, default=512)
    p.add_argument("--throughput_prompt_len", type=int, default=16)

    # ---- accepted-and-ignored (legacy launch.json) ----
    p.add_argument("--wandb_group_name", default=None, help="(ignored — no experiment tracking)")

    args = p.parse_args(argv)

    # --- derive ---
    if args.compression_method == "rr":
        args.compression_method = "air"
    if args.parameter_rate is None:
        args.parameter_rate = args.parameter_rate_idx_list / 100.0
    args.use_influence = (args.compression_method == "air" and args.relevance_impact > 0)
    args.use_lrp = args.use_influence and args.importance_metrics == "relevances"
    args.run_als = (args.compression_method == "air")
    args.overwrite_analysis = (args.OVERWRITE_ANALYSIS or args.OVERWRITE_RELEVANCES
                               or args.OVERWRITE_PROFILING_MATRIX or args.OVERWRITE_CACHE)
    args.overwrite_svd = (args.OVERWRITE_SVD or args.OVERWRITE_CACHE)

    # --- paths ---
    # Shared HF download cache (model weights + datasets), model-agnostic so a base model is
    # downloaded once regardless of model_tag / rate. Under results_dir unless $HF_HOME is set.
    if not args.cache_dir:
        args.cache_dir = os.path.join(args.results_dir, "hf_cache")
    os.makedirs(args.cache_dir, exist_ok=True)

    # Per-model outputs. Caches are keyed by a CONFIG SIGNATURE so different settings never
    # collide: S/I live under the analysis signature (what defines them — calibration set +
    # influence params); the compressed model lives under analysis-sig / compression-sig (it is
    # built from S/I AND the compression params). Run logs sit at the model level.
    model_tag = ("tiny_llama_synthetic" if args.tiny else args.model_name.replace("/", "_"))
    args.analysis_sig, args.compress_sig = _cache_signatures(args)
    args.model_dir = os.path.join(args.results_dir, model_tag)
    args.analysis_dir = os.path.join(args.model_dir, "analysis", args.analysis_sig)
    args.compressed_dir = os.path.join(args.model_dir, "compressed", args.analysis_sig, args.compress_sig)
    for d in (args.model_dir, args.analysis_dir, args.compressed_dir):
        os.makedirs(d, exist_ok=True)

    return args


def _cache_signatures(args):
    """Build the cache keys.

    analysis_sig — everything that determines the profiling S and influence I: the calibration
    set (dataset, #samples, seqlen, seed) and the influence signal (metric, lrp epsilon, the
    iterative normalization). Shared by `air` and `svd_llm` at matching settings, so svd_llm
    reuses air's S. (use_influence is NOT part of it: svd_llm just omits the I files.)

    compress_sig — what additionally determines the compressed model: method, parameter rate,
    and (air only) the ALS influence weighting / sweeps.
    """
    g = lambda x: format(x, "g")
    infl = f"-{args.importance_metrics}" + (f"-eps{g(args.lrp_epsilon)}"     # eps only affects LRP
                                            if args.importance_metrics == "relevances" else "")
    infl += f"-norm{int(args.RELEVANCE_NORMALIZATION)}x{g(args.relevance_normalization_max)}"
    analysis_sig = (f"{args.calibration_data}-n{args.n_samples_calibration}"
                    f"-seq{args.relevance_subset_seq_len}-seed{args.seed}{infl}")
    if args.compression_method == "air":
        compress_sig = (f"air-rate{g(args.parameter_rate)}-d{g(args.relevance_impact)}"
                        f"-a{g(args.activation_impact)}-sw{args.n_sweeps}")
    else:
        compress_sig = f"{args.compression_method}-rate{g(args.parameter_rate)}"
    return analysis_sig, compress_sig
