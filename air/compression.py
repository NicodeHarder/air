"""AIR compression: per-weight low-rank factorization, layer-by-layer.

For each weight W (m x n):
  1. whiten:   W' = W @ S         (S = cholesky(sum XX^T) from the analysis; None -> plain SVD)
  2. init:     U_k Sigma_k V_k^T = SVD(W', k)              (SVD-LLM(W) activation-aware optimum)
  3. (AIR)     refine the factors with the influence-aware ALS sweep on W'   [compression_als.py]
  4. unwhiten + absorb sigma:  svd_u = U_k sqrt(Sigma_k),  svd_v = sqrt(Sigma_k) V_k^T S^{-1}
  -> two linears  V (n->k)  and  U (k->m), so U(V x) ~= W x.

Memory: strictly one layer at a time; weights cast to fp32 for SVD, covariance to fp64 for
Cholesky; factors stored back in model dtype; temporaries deleted + cache emptied per layer.
"""

import gc
import math
import os
import torch

def _svd_classes(family):
    if family == "qwen":
        from layers.qwen import SVD_QwenAttention, SVD_QwenMLP
        return SVD_QwenAttention, SVD_QwenMLP
    if family == "mistral":
        from layers.mistral import SVD_MistralAttention, SVD_MistralMLP
        return SVD_MistralAttention, SVD_MistralMLP
    from layers.llama import SVD_LlamaAttention, SVD_LlamaMLP
    return SVD_LlamaAttention, SVD_LlamaMLP


# weight name (suffix) -> which SVD sub-module receives U / V
ATTN_NAMES = ["q_proj", "k_proj", "v_proj", "o_proj"]
MLP_NAMES = ["gate_proj", "up_proj", "down_proj"]


def find_linears(module, prefix=""):
    out = {}
    for name, child in module.named_modules():
        if isinstance(child, torch.nn.Linear):
            out[name] = child
    return out


def compute_rank(m, n, rate):
    """Uniform per-layer rank for a target parameter `rate` in (0,1]: k(m+n) = rate*m*n."""
    k = int(m * n * rate / (m + n))
    return max(1, min(k, min(m, n)))


def _human(n):
    """Compact parameter count, e.g. 8.39M / 537.0K / 196."""
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    if n >= 1e6:
        return f"{n / 1e6:.2f}M"
    if n >= 1e3:
        return f"{n / 1e3:.1f}K"
    return str(int(n))


def _mse_db(sq_loss, n_elem):
    """Mean-squared reconstruction error in dB: 10·log10(‖·‖²_F / N) (as in the original code)."""
    mse = sq_loss / max(n_elem, 1)
    return 10.0 * math.log10(mse) if mse > 0 else float("-inf")


def compute_whitening(xtx, device):
    """Return (S, S_inv) for the cholesky whitening W' = W·S  (S Sᵀ = XXᵀ), S_inv = inv(S).
    Cholesky's lower-triangular, diagonal-dominant S keeps the influence aligned with the weight
    columns (required for AIR). Robust to a non-PD XXᵀ via an eigenvalue-relative shift, then an
    eigh square-root fallback (SVD-LLM trick)."""
    A = xtx.double().to(device)
    A = (A + A.transpose(-1, -2)) / 2
    try:
        S = torch.linalg.cholesky(A)
    except Exception:
        eig = torch.linalg.eigvalsh(A)
        eps = max(abs(eig[0].item()) * 1.1, eig[-1].item() * 1e-6, 1e-4)
        A = A + eps * torch.eye(A.shape[0], dtype=A.dtype, device=A.device)
        try:
            S = torch.linalg.cholesky(A)
        except Exception:
            w, V = torch.linalg.eigh(A)
            w = torch.clamp(w, min=1e-7)
            S = V @ torch.diag(w.sqrt()) @ V.transpose(-1, -2)
    return S, torch.linalg.inv(S)


def factorize_weight(W, k, S=None, S_inv=None, influence=None, args=None, run_als=False):
    """Return (svd_u [m,k], svd_v [k,n], errs) approximating W [m,n]. `errs` are the squared
    Frobenius losses on the whitened W': (L_act, L_actinfl_init, L_actinfl_als) — Eq.3 of the
    SVD-LLM(W) init and Eq.4 before/after ALS (the latter two None without influence-aware ALS)."""
    dev = W.device
    Wf = W.float()
    if S is not None:
        S = S.float().to(dev)
        S_inv = S_inv.float().to(dev)
        W2 = Wf @ S
    else:
        S_inv = None
        W2 = Wf

    U, sig, Vh = torch.linalg.svd(W2, full_matrices=False)
    U_k, sig_k, Vh_k = U[:, :k].contiguous(), sig[:k].contiguous(), Vh[:k, :].contiguous()

    l_act = float((sig[k:] ** 2).sum())                  # Eq.3: ‖W'-U_kΣ_kV_kᵀ‖_F² at SVD-LLM(W) init
    l_ai_init = l_ai_als = None
    if run_als:
        from compression_als import als_refine          # influence-aware ALS (paper Eqs 5-6)
        U_k, sig_k, Vh_k, l_ai_init, l_ai_als = als_refine(
            W2, U_k, sig_k, Vh_k, influence, args, return_loss=True)   # Eq.4 before/after

    truc_v = Vh_k if S_inv is None else (Vh_k @ S_inv)   # undo whitening on V
    sqrt_s = sig_k.clamp(min=0).sqrt()
    svd_u = U_k * sqrt_s.unsqueeze(0)                     # [m,k]
    svd_v = sqrt_s.unsqueeze(1) * truc_v                 # [k,n]
    return svd_u.to(W.dtype), svd_v.to(W.dtype), (l_act, l_ai_init, l_ai_als)


def _set_uv(v_module, u_module, svd_u, svd_v, orig_bias, v_row_slice=None):
    """Write factors into a (V, U) linear pair; original bias goes on U, V bias is zeroed."""
    if v_row_slice is None:
        v_module.weight.data.copy_(svd_v)
    else:
        v_module.weight.data[v_row_slice].copy_(svd_v)
        if v_module.bias is not None:
            v_module.bias.data[v_row_slice].zero_()
    if v_row_slice is None and v_module.bias is not None:
        v_module.bias.data.zero_()
    u_module.weight.data.copy_(svd_u)
    if u_module.bias is not None:
        u_module.bias.data.zero_()
        if orig_bias is not None:
            u_module.bias.data.copy_(orig_bias.to(u_module.bias.dtype))


def compress_layer(layer, layer_idx, config, dtype, rate, S_dict, I_dict, args, device,
                   attn_cls, mlp_cls, n_layers=None):
    """Replace layer.self_attn and layer.mlp with low-rank SVD modules (in place).

    Logs, per weight: native dim [m x n] -> truncation rank k -> factor dims and the resulting
    local parameter rate; then a per-layer aggregate (params before/after, layer rate)."""
    lin = {n.split(".")[-1]: m for n, m in find_linears(layer).items()
           if n.split(".")[-1] in ATTN_NAMES + MLP_NAMES}
    tag = f"{layer_idx + 1}/{n_layers}" if n_layers else f"{layer_idx + 1}"
    print(f"\nlayer {tag}:")
    stats = []  # (orig_params, factor_params) per weight, for the layer aggregate
    wh = [None, None, None]  # (XXᵀ data_ptr, S, S_inv) — q/k/v and gate/up share XXᵀ, reuse its whitening

    def fac(name):
        W = lin[name].weight.data.to(device)
        m, n = W.shape
        k = compute_rank(m, n, rate)
        if S_dict and name in S_dict:
            if wh[0] != S_dict[name].data_ptr():
                wh[:] = [None, None, None]               # free the previous group's S/S_inv first
                wh[:] = [S_dict[name].data_ptr(), *compute_whitening(S_dict[name], device)]
            S, S_inv = wh[1], wh[2]
        else:
            S, S_inv = None, None
        I = I_dict[name].to(device) if (I_dict and name in I_dict) else None
        u, v, (l_act, l0, l1) = factorize_weight(W, k, S=S, S_inv=S_inv, influence=I, args=args,
                                                 run_als=args.run_als and I is not None)
        bias = lin[name].bias.data if lin[name].bias is not None else None
        orig, new = m * n, u.numel() + v.numel()
        stats.append((orig, new))
        # Per-weight reconstruction error on the whitened W' in dB (10·log10 of the MSE):
        #   L_act (Eq.3) = activation-aware (SVD-LLM(W) init); L_act,infl (Eq.4) = influence-aware,
        #   shown init -> after ALS (the monotone descent).
        err = f"L_act {_mse_db(l_act, m * n):.2f} dB"
        if l0 is not None:
            err += f" | L_act,infl {_mse_db(l0, m * n):.2f} -> {_mse_db(l1, m * n):.2f} dB"
        print(f"  {name:9s} W[{m} x {n}] -> decomposed -> truncated to rank {k} -> "
              f"[{m} x {k}]+[{k} x {n}]  {_human(orig)}->{_human(new)}  local rate {100 * new / orig:.1f}%"
              f"  |  err(dB): {err}")
        del W, S, S_inv, I
        return u.cpu(), v.cpu(), k, bias.cpu() if bias is not None else None

    # ---- attention ----
    uq, vq, kq, bq = fac("q_proj"); uk, vk, kk, bk = fac("k_proj")
    uv, vv, kv, bv = fac("v_proj"); uo, vo, ko, bo = fac("o_proj")
    attn = attn_cls(config, [kq, kk, kv, ko], layer_idx, dtype)
    _set_uv(attn.qkv_v_proj, attn.q_u_proj, uq, vq, bq, slice(0, kq))
    _set_uv(attn.qkv_v_proj, attn.k_u_proj, uk, vk, bk, slice(kq, kq + kk))
    _set_uv(attn.qkv_v_proj, attn.v_u_proj, uv, vv, bv, slice(kq + kk, kq + kk + kv))
    _set_uv(attn.o_v_proj, attn.o_u_proj, uo, vo, bo)

    # ---- mlp ----
    ug, vg, kg, bg = fac("gate_proj"); uu, vu, ku, bu = fac("up_proj"); ud, vd, kd, bd = fac("down_proj")
    mlp = mlp_cls(config, [kg, ku, kd], dtype)
    _set_uv(mlp.gate_up_v_proj, mlp.gate_u_proj, ug, vg, bg, slice(0, kg))
    _set_uv(mlp.gate_up_v_proj, mlp.up_u_proj, uu, vu, bu, slice(kg, kg + ku))
    _set_uv(mlp.down_v_proj, mlp.down_u_proj, ud, vd, bd)

    orig_device = next(layer.parameters()).device
    layer.self_attn = attn.to(orig_device)
    layer.mlp = mlp.to(orig_device)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    lo = sum(o for o, _ in stats)
    ln = sum(n for _, n in stats)
    print(f"  -> layer {tag} total: {_human(lo)} -> {_human(ln)} params  (layer rate {100 * ln / lo:.1f}%)")
    return lo, ln


def build_compressed_skeleton(model, args):
    """Swap each decoder layer's attn/mlp for low-rank SVD modules with the CORRECT per-weight
    ranks but uninitialized factors, so a cached compressed `state_dict` can be loaded back in
    (no SVD / ALS). Ranks are deterministic from the rate, so they match what compress produced."""
    from modeling import get_decoder_layers, model_family
    layers = get_decoder_layers(model)
    config = model.config
    dtype = next(model.parameters()).dtype
    family = "llama" if args.tiny else model_family(args.model_name)
    attn_cls, mlp_cls = _svd_classes(family)
    for i, layer in enumerate(layers):
        lin = {n.split(".")[-1]: m for n, m in find_linears(layer).items()
               if n.split(".")[-1] in ATTN_NAMES + MLP_NAMES}
        rk = {nm: compute_rank(m.weight.shape[0], m.weight.shape[1], args.parameter_rate)
              for nm, m in lin.items()}
        attn = attn_cls(config, [rk["q_proj"], rk["k_proj"], rk["v_proj"], rk["o_proj"]], i, dtype)
        mlp = mlp_cls(config, [rk["gate_proj"], rk["up_proj"], rk["down_proj"]], dtype)
        orig_device = next(layer.parameters()).device
        layer.self_attn = attn.to(orig_device)
        layer.mlp = mlp.to(orig_device)
    return model


def compress_model(model, args):
    """Compress every decoder layer in place, loading cached per-layer S / I from disk."""
    from modeling import get_decoder_layers, model_family
    layers = get_decoder_layers(model)
    config = model.config
    dtype = next(model.parameters()).dtype
    device = args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu"
    family = "llama" if args.tiny else model_family(args.model_name)
    attn_cls, mlp_cls = _svd_classes(family)

    n = len(layers)
    refine = "influence-aware ALS" if args.run_als else "SVD-LLM(W) init only"
    print(f"\nStarting compression: {n} decoder layers at parameter rate {args.parameter_rate:.2f} "
          f"-> {refine} ...")
    tot_o = tot_n = 0
    for i in range(n):
        S_dict = _load(os.path.join(args.analysis_dir, f"S_layer_{i}.pt"))
        I_dict = _load(os.path.join(args.analysis_dir, f"I_layer_{i}.pt")) if args.use_influence else None
        lo, ln = compress_layer(layers[i], i, config, dtype, args.parameter_rate, S_dict, I_dict,
                                args, device, attn_cls, mlp_cls, n_layers=n)
        tot_o += lo
        tot_n += ln
        del S_dict, I_dict                       # free this layer's profiling/influence before the next
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print(f"\nCompressed {n} layers: {_human(tot_o)} -> {_human(tot_n)} decomposed params "
          f"(overall rate {100 * tot_n / tot_o:.1f}% | target {100 * args.parameter_rate:.0f}%)")
    return model


def _load(path):
    return torch.load(path, map_location="cpu") if os.path.exists(path) else None
