# AIR — Activation- and Influence-Aware Ranks

**Function-preserving SVD compression for LLMs.** A clean, paper-driven implementation
(see the NeurIPS write-up) of AIR: compress each weight matrix `W ≈ U_k V_kᵀ` so the layer's
*function* is preserved, using a forward-pass **activation** signal and a backward-pass
**influence** signal.

```
W'  = W · S,        S Sᵀ = Σ_cal X Xᵀ (S = cholesky)    # activation-aware whitening (profiling)
I   = | (∂L/∂W) · W |,  AttnLRP ε-rule, per-layer rescaled   # backward influence
U_k Σ_k V_kᵀ = SVD(W', k)                                # SVD-LLM(W) activation-aware init
minimize ‖ √(1+δI) ⊙ (W' − U_k Σ_k V_kᵀ) ‖_F²  via ALS  # influence-aware refinement (Eqs 5-6)
```
The whitening `S` is the **Cholesky** factor of `Σ XXᵀ` (SVD-LLM v1). Its lower-triangular,
diagonal-dominant structure (`W'[:,j] ≈ S_jj·W[:,j]`) keeps the element-wise influence `I` (defined
on `W`) aligned with `W'`, which the influence-weighted ALS (Eq.4) relies on. (A full-eigenbasis
square-root such as `V√Λ` would be equivalent for the activation-aware truncation but rotates the
columns and decorrelates `I` from the residual, so it is not used.) Non-PD `XXᵀ` (too few
calibration tokens) is handled by an eigenvalue shift + eigh fallback; for stable whitening keep
`n_samples × seqlen ≫ hidden`.

## What's distinctive here
- **Merged forward-backward analysis** (`air/analysis.py`): the profiling matrix `S` (forward
  hooks accumulating `XXᵀ`) and the influence matrix `I` (AttnLRP backward) are computed in a
  **single pass** over the calibration data, rather than two separate pipelines.
- **Two-phase model lifecycle** (mirrors the proven design): the AttnLRP-instrumented model is
  used **only** for analysis, then deleted from the GPU; a **fresh plain model** is loaded for
  compression + evaluation.
- **Memory-frugal** so large models compress on a single small GPU (see below).
- **Minimal**: only what the `air` launch config needs — no experiment tracking, no dead code.

## Layout
```
air/
├── run.py            # entry point — the two-phase pipeline
├── config.py         # CLI flags
├── modeling.py       # load lxt model (analysis) / plain model (compress+eval); device_map; tiny synthetic
├── data.py           # calibration windows + wikitext2/c4/ptb perplexity chunks (+ synthetic --tiny)
├── analysis.py       # ★ merged forward-backward: profiling S + influence I + iterative scaling
├── compression.py    # SVD-LLM(W) init, rank selection, Cholesky-shift, layer-by-layer swap
├── compression_als.py# influence-aware ALS sweep (paper Eqs 5-6)
├── efficiency.py     # static footprint + FLOPs (baseline vs compressed, cached baseline)
├── evaluation.py     # perplexity (+ optional throughput / peak memory)
└── layers/           # low-rank SVD modules: llama.py (+ mistral.py, qwen.py reuse it)
lxt/                  # vendored AttnLRP backend (the influence signal)
```
The method core is [`air/compression.py`](air/compression.py) + [`air/compression_als.py`](air/compression_als.py);
the analysis core is [`air/analysis.py`](air/analysis.py).

## Memory-reduction strategies (kept — enables e.g. 30B on a 40GB A100)
- bf16 weights + HF `device_map`. **Multi-GPU only for the analysis backward** (it needs the
  combined VRAM for the lxt 2048-token LRP backward); compression + evaluation run on a **single
  GPU** (faster inference, representative single-device latency/peak memory).
- Analysis: **per-sample** forward+backward (one sample's graph at a time); relevances detached
  to CPU immediately; `zero_grad(set_to_none=True)` + `empty_cache` + `gc` each sample.
- Compression: strictly **layer-by-layer**, weights→fp32 for SVD / covariance→fp64 for Cholesky,
  factors back in model dtype on CPU, all temporaries deleted + cache emptied per layer.
- Per-layer **disk cache** of `S` and `I` (load on demand; never all layers resident).
- The lxt analysis model is freed before the plain compression/eval model is loaded.

## Install
Requires Python 3.10/3.11 + a CUDA GPU.
```bash
chmod +x create_venv.sh && ./create_venv.sh && source env/bin/activate
cp .vscode/.env.example .vscode/.env   # set HF_TOKEN / RESULTS_DIR
```

## Run
VS Code: launch the **`air`** config. Or CLI:
```bash
python air/run.py --model_name TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --compression_method air --parameter_rate_idx_list 60 \
    --n_samples_calibration 256 --relevance_subset_seq_len 2048 \
    --evaluation_perplexity_data wikitext2 --batch_size 4
```
Key flags: `--compression_method {air,svd_llm}` (air = influence-aware ALS; svd_llm =
activation-aware init only, the baseline), `--parameter_rate_idx_list` (percent), `--model_name`
(Llama / Mistral / Qwen), `--PERPLEXITY_ONLY` (skip throughput), `--relevance_impact` (δ, default 2),
`--no-SVD` / `--no-EVALUATE`.

**Backward influence signal** — `--importance_metrics` (the paper is agnostic to the choice):
`relevances` (default, AttnLRP ε-rule; needs the lxt model), `weightxgrad` (Weight×Gradient
`|∂L/∂W·W|`) or `fisher` (diagonal empirical Fisher `(∂L/∂W)²`). The gradient signals need no lxt —
they run a plain model with a standard LM-loss backward — and are cached separately per signal.

## Caching (config-keyed)
Every expensive artifact is cached and reused automatically; nothing is recomputed unless its
inputs change. Caches are keyed by a **config signature** so different settings never collide:
```
RESULTS_DIR/<model_tag>/
├── run_<ts>.log
├── baseline_metrics.json           # uncompressed params / footprint / FLOPs (per model)
├── analysis/<analysis-sig>/        S_layer_*.pt, I_layer_*.pt   (defined by: calib set + influence params)
│   └─ e.g. wikitext2-n256-seq2048-seed3-relevances-eps1e-06-norm1x100
└── compressed/<analysis-sig>/<compress-sig>/model_state_dict.pt (adds: method, rate, ALS δ/α/sweeps)
    └─ e.g. .../air-rate0.6-d2-a1-sw1
```
On a cache hit the compressed model is loaded directly (analysis **and** compression skipped). `air`
and `svd_llm` at matching calibration share the same `analysis-sig`, so svd_llm reuses air's `S`.
Control: `--OVERWRITE_CACHE` (recompute everything), `--OVERWRITE_ANALYSIS` (just S/I),
`--OVERWRITE_SVD` (just the compressed model). The compressed model is cached only when
`--SAVE_COMPRESSED_MODEL` is set (the analysis cache — the costly LRP backward — is always written).

## Efficiency + quality report
Every run prints two tables — **baseline vs compressed absolute values and rel % (= compressed /
baseline)** — an efficiency block and a quality block:
```
=== Efficiency (baseline -> compressed | rel % = compressed/baseline) ===
  metric                         baseline     compressed    rel %
  Parameters                        1.10B         712.3M    64.8%
  Footprint (storage)             2.20 GB        1.42 GB    64.8%
  Forward FLOPs / token         2.07 GFLOP     1.29 GFLOP   62.5%
  Peak GPU memory                 4.21 GB        2.71 GB    64.4%
  Latency / token                250.1 us       190.3 us    76.1%
=== Quality (baseline -> compressed | rel % = compressed/baseline) ===
  Perplexity (wikitext2)           7.96          39.47     495.9%
```
Static metrics (params/footprint/FLOPs) come from a zero-memory meta-device model — even a 30B
baseline costs nothing. The dynamic ones (peak GPU memory, latency via a warmed-up generation)
and the quality metric (perplexity) require running the uncompressed model; they appear only when
`--MEASURE_THROUGHPUT`/`--MEASURE_TIME` (efficiency) or `--EVALUATE` (perplexity) are on.
All baseline numbers are measured once — before any lxt wrapping — and cached in
`baseline_metrics.json` (static keys model-wide; perplexity/throughput keyed by eval config), so
they're reused on later runs (including compressed-cache hits); `--OVERWRITE_CACHE` recomputes them.

## Throughput & KV-cache optimization (`--OPTIMIZE_KV_CACHE`)
The **static** metrics (params / footprint / FLOPs) reflect the compression itself and improve at
any scale. The **dynamic** metrics (peak GPU memory, per-token latency) are measured during an
actual generation and are **memory-bandwidth- and KV-cache-bound**, so they depend on the
generation regime — batch size and sequence length — not just the parameter rate.

`--OPTIMIZE_KV_CACHE` enables the three forward-pass optimizations the paper's peak-GPU/latency
claims rely on (decode-time only; perplexity is unaffected):
- **O1 — RoPE pre-application:** keys are cached full-dimension with RoPE already applied, via an
  internal cache path that bypasses HuggingFace's `DynamicCache` (prerequisite for O2/O3).
- **O2 — fused low-rank value caching:** the value cache stores the rank-`r_v` latent instead of
  full `n_kv·d_h` values; the up-projection is fused into the attention output. This is the main
  **peak-memory** driver.
- **O3 — pre-allocated cache buffers:** cache tensors are allocated once at the max sequence length
  and filled in place (with the value up-projection weight precomputed per head; buffers grow in
  pages if a generation exceeds the pre-allocated length). This is the main **latency** driver.
- **O4 — grouped-query cache attention + SDPA prefill** (automatic with the flag): decode scores
  are computed against the *unexpanded* K cache (no per-step `repeat_kv` copy of the whole cache —
  up to ~9× faster attention and ~100× less transient memory on GQA models such as Mistral /
  TinyLlama), and the prefill chunk runs through the fused SDPA kernel (prefill peak scales with
  `prompt·d_h` instead of `prompt²`).
- **O5 — `--CUDA_GRAPH`:** the whole greedy decode step (embeddings → all layers → lm_head →
  argmax → token feedback) is captured once as a CUDA graph and replayed per token, collapsing the
  per-step kernel-launch + Python overhead to a single launch. Biggest gain at small batch, where
  that overhead dominates; requires `--OPTIMIZE_KV_CACHE` on CUDA.

Without the flag the compressed model caches full K **and** V through the standard HF path (no
cache-memory reduction, and often *slower* than the base model — two smaller GEMMs do not beat one
in the bandwidth-bound decode regime).

**Reproducing the paper's peak-GPU / latency numbers requires matching the generation regime.** The
KV cache only dominates — and the optimizations only pay off — at a large batch size and a long
sequence; at small batch / short sequences the cache is negligible and the fused path's overhead can
make it marginally *slower*. Set `--throughput_batch_size`, `--throughput_prompt_len`, and
`--throughput_n_generate` to the paper's regime (e.g. batch 64, ~512 generated tokens) on comparable
hardware. The reported latency/peak-memory rates shift with these knobs, so always state them
alongside the numbers.

Smoke test (instant, no download): `python air/run.py --tiny --device cpu --n_samples_calibration 4
--relevance_subset_seq_len 32 --eval_seqlen 32 --batch_size 2 --OVERWRITE_CACHE`.

## License
MIT (see [LICENSE](LICENSE)). The vendored [`lxt/`](lxt/) AttnLRP backend is third-party code under The Clear BSD License (see [lxt/LICENSE](lxt/LICENSE)).
