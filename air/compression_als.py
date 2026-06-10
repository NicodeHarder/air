"""Influence-aware ALS refinement (AIR, paper Eqs 4-6).

Starting from the SVD-LLM(W) optimum (U_k, sigma_k, V_k of the whitened W'), refine each rank
in closed form to minimize the hybrid activation- and influence-aware loss

    L = || sqrt(1 + delta*I) ⊙ (W' - U_k Sigma_k V_k^T) ||_F^2 ,

sweeping ranks r = k-1 .. 0 (largest residuals first). Each step is the exact 1-rank weighted
least-squares minimizer, so the loss is non-increasing (paper Prop. 3.1).
"""

import torch

EPS = 1e-10


def _weighted_loss(W2, U, s, Vh, weight):
    Wk = (U * s.unsqueeze(0)) @ Vh
    return (weight * (W2 - Wk) ** 2).sum()


def als_refine(W2, U_k, sig_k, Vh_k, influence, args, return_loss=False):
    """Refine (U_k [m,k], sig_k [k], Vh_k [k,n]) in place-ish; returns the refined triple."""
    dev = W2.device
    U = U_k.float().clone()
    s = sig_k.float().clone()
    Vh = Vh_k.float().clone()                      # rows = right singular vectors

    # element-wise loss weight  (1 + delta*I)  -- influence already per-layer unit-mean normalized
    if influence is not None and args.relevance_impact != 0:
        weight = influence.float().to(dev) * args.relevance_impact + args.activation_impact
    else:
        weight = torch.full_like(W2, float(args.activation_impact))

    k = U.shape[1]
    E = W2 - (U * s.unsqueeze(0)) @ Vh              # residual W' - W_k, kept up to date per rank
    loss0 = _weighted_loss(W2, U, s, Vh, weight) if return_loss else None

    for _ in range(args.n_sweeps):
        for r in range(k - 1, -1, -1):              # backward sweep r = k-1 .. 0 (paper order)
            u_r, v_r, s_r = U[:, r], Vh[r, :], s[r]
            # residual with rank r added back:  E_r = (W' - W_k) + s_r u_r v_r^T
            resid = torch.addr(E, u_r * s_r, v_r)
            wres = resid * weight                                   # (1+δI) ⊙ E_r

            # v update (Eq. 5):  v' = (u_r^T (w⊙E)) / (s_r * (u_r^2)^T w)
            den_v = ((u_r ** 2) @ weight) * s_r
            v_new = torch.where(den_v.abs() > EPS, (u_r @ wres) / den_v, v_r)

            # u update (Eq. 6):  (u s)' = ((w⊙E) v') / (w v'^2);  then renormalize
            den_u = weight @ (v_new ** 2)
            us = torch.where(den_u.abs() > EPS, (wres @ v_new) / den_u, torch.zeros_like(u_r))
            s_new = torch.linalg.norm(us)
            u_new = us / (s_new + EPS)

            # commit:  E = E_r - s' u' v'^T
            E = torch.addr(resid, u_new * (-s_new), v_new)
            U[:, r], s[r], Vh[r, :] = u_new, s_new, v_new

    if return_loss:
        return U, s, Vh, loss0.item(), _weighted_loss(W2, U, s, Vh, weight).item()
    return U, s, Vh
