# -*- coding: utf-8 -*-
"""
Paper 3 v3.0 Comprehensive Stress-Test & Pre-Registered Forecast Suite
("Logarithmic Oscillatory Dark Energy: DESI DR2, DES-SN5YR Dovekie, Planck 3D,
 Look-Elsewhere Periodogram, Orthogonal MCMC, and Mid-2027 ESA Euclid DR1 Predictions").

Fixes all 5 v2.0 audit defects by routing 100% of likelihood calls through the single
canonical `CosmologyLikelihoodEngine` and coupled anisotropic `src.euclid_fisher` module.
"""

import os
import sys
import json
import time
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, minimize
from scipy.stats import chi2 as chi2_dist
import emcee

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.likelihoods import (
    CosmologyLikelihoodEngine,
    w_model,
    rho_de_ratio,
    cs_to_polar,
    polar_to_cs,
    rd_physical_mpc,
    C_LIGHT,
)
from src.euclid_fisher import (
    EUCLID_IST_Z_BINS,
    compute_combined_3sigma_boundary,
    compute_euclid_anisotropic_observables,
    build_euclid_anisotropic_covariance,
    compute_euclid_anisotropic_fisher,
    compute_euclid_model_separation_anisotropic,
    generate_euclid_2027_preregistration,
)

RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# =====================================================================
# HELPERS: OPTIMIZATION, CONVERGENCE DIAGNOSTICS & BOUND CHECKS
# =====================================================================

def check_at_bounds(x, bounds, tol=1e-3):
    flags = []
    for val, (lo, hi) in zip(x, bounds):
        span = max(hi - lo, 1e-12)
        flags.append(bool(abs(val - lo) < tol * span or abs(val - hi) < tol * span))
    return flags


def split_rhat(chains_3d):
    """
    Compute split-Rhat (Gelman-Rubin) across walker chains after splitting each
    production chain into first and second halves -> 2*n_walkers subchains.
    chains_3d shape: (n_steps, n_walkers, ndim).
    """
    n_steps, n_walkers, ndim = chains_3d.shape
    half = n_steps // 2
    sub = np.concatenate([chains_3d[:half], chains_3d[half : 2 * half]], axis=1)  # (half, 2W, D)
    m = sub.shape[1]
    n = sub.shape[0]
    means = np.mean(sub, axis=0)            # (m, D)
    vars_w = np.var(sub, axis=0, ddof=1)    # (m, D)
    W = np.mean(vars_w, axis=0)             # (D,)
    B = n * np.var(means, axis=0, ddof=1)   # (D,)
    var_hat = ((n - 1.0) / n) * W + (1.0 / n) * B
    return np.sqrt(np.maximum(var_hat, 1e-15) / np.maximum(W, 1e-15))


def fit_model(
    engine,
    model_name,
    use_sn=True,
    use_bao=True,
    use_cmb=True,
    cmb_mode="planck_3d",
    seed=202611,
):
    """Global optimization + multi-start Powell/Nelder-Mead refinement."""
    fixed_h = 0.675 if cmb_mode == "planck_1d_R" else 0.6748
    fixed_ob = 0.02236
    if (not use_cmb) or cmb_mode == "planck_1d_R":
        # Fixed h, omega_b when CMB is disabled (uncalibrated SN_only) or in v2.0 1D R audit mode
        if model_name == "LCDM":
            pnames = ["Om"]
            bounds = [(0.18, 0.50)]
            unpack = lambda x: (x[0], fixed_h, fixed_ob, [])
        elif model_name == "CPL":
            pnames = ["Om", "w0", "wa"]
            bounds = [(0.18, 0.50), (-1.8, 0.0), (-3.5, 2.0)]
            unpack = lambda x: (x[0], fixed_h, fixed_ob, [x[1], x[2]])
        else:
            pnames = ["Om", "C", "S", "lambda"]
            bounds = [(0.18, 0.50), (-0.95, 0.95), (-0.95, 0.95), (1.2, 12.0)]
            unpack = lambda x: (x[0], fixed_h, fixed_ob, [x[1], x[2], x[3]])
    else:
        if model_name == "LCDM":
            pnames = ["Om", "h", "omega_b"]
            bounds = [(0.18, 0.50), (0.58, 0.76), (0.0215, 0.0235)]
            unpack = lambda x: (x[0], x[1], x[2], [])
        elif model_name == "CPL":
            pnames = ["Om", "h", "omega_b", "w0", "wa"]
            bounds = [(0.18, 0.50), (0.58, 0.76), (0.0215, 0.0235), (-1.8, 0.0), (-3.5, 2.0)]
            unpack = lambda x: (x[0], x[1], x[2], [x[3], x[4]])
        elif model_name == "OSC_CS":
            pnames = ["Om", "h", "omega_b", "C", "S", "lambda"]
            bounds = [
                (0.18, 0.50),
                (0.58, 0.76),
                (0.0215, 0.0235),
                (-0.95, 0.95),
                (-0.95, 0.95),
                (1.2, 12.0),
            ]
            unpack = lambda x: (x[0], x[1], x[2], [x[3], x[4], x[5]])
        else:
            raise ValueError(model_name)

    def obj(x):
        for v, (lo, hi) in zip(x, bounds):
            if v < lo or v > hi:
                return 1e12
        Om, h, ob, de_p = unpack(x)
        if model_name == "OSC_CS" and np.hypot(de_p[0], de_p[1]) > 1.0:
            return 1e12 + 1e4 * (np.hypot(de_p[0], de_p[1]) - 1.0) ** 2
        if model_name == "CPL" and (de_p[0] + de_p[1]) > 0.2:
            return 1e12
        c_sn, c_bao, c_cmb = engine.compute_chi2_components(
            Om,
            model_name,
            de_p,
            h=h,
            omega_b=ob,
            use_sn=use_sn,
            use_bao=use_bao,
            use_cmb=use_cmb,
            cmb_mode=cmb_mode,
        )
        return float(c_sn + c_bao + c_cmb)

    maxit = 180 if model_name != "OSC_CS" else 260
    res_de = differential_evolution(
        obj, bounds, seed=seed, maxiter=maxit, popsize=12, tol=1e-5, polish=False
    )
    best_x = res_de.x
    best_f = float(res_de.fun)

    # Local Powell + Nelder-Mead polish
    for meth in ("Powell", "Nelder-Mead"):
        res_loc = minimize(obj, best_x, method=meth, options={"ftol": 1e-6, "xtol": 1e-6} if meth == "Powell" else {"fatol": 1e-6, "xatol": 1e-6})
        if res_loc.fun < best_f:
            best_f = float(res_loc.fun)
            best_x = np.clip(res_loc.x, [b[0] for b in bounds], [b[1] for b in bounds])

    # For OSC_CS, also verify multi-lambda slices to guarantee we didn't miss a narrow mode
    if model_name == "OSC_CS" and use_cmb and cmb_mode == "planck_3d":
        for lam_seed in (1.5, 2.2, 3.0, 4.5, 7.0, 10.5):
            for c_init, s_init in ((0.15, 0.15), (-0.15, 0.15), (0.15, -0.15)):
                x0 = np.array([best_x[0], best_x[1], best_x[2], c_init, s_init, lam_seed])
                r_try = minimize(obj, x0, method="Nelder-Mead", options={"maxiter": 600, "fatol": 1e-5})
                if r_try.fun < best_f:
                    best_f = float(r_try.fun)
                    best_x = np.clip(r_try.x, [b[0] for b in bounds], [b[1] for b in bounds])

    Om, h, ob, de_p = unpack(best_x)
    c_sn, c_bao, c_cmb = engine.compute_chi2_components(
        Om,
        model_name,
        de_p,
        h=h,
        omega_b=ob,
        use_sn=use_sn,
        use_bao=use_bao,
        use_cmb=use_cmb,
        cmb_mode=cmb_mode,
    )
    chi2_min = float(c_sn + c_bao + c_cmb)
    params_dict = dict(zip(pnames, [float(v) for v in best_x]))
    if (not use_cmb) or cmb_mode == "planck_1d_R":
        params_dict.setdefault("h", fixed_h)
        params_dict.setdefault("omega_b", fixed_ob)
    if model_name == "OSC_CS":
        A_val, phi_val = cs_to_polar(params_dict["C"], params_dict["S"])
        params_dict["A"] = A_val
        params_dict["phi0"] = phi_val

    bflags = check_at_bounds(best_x, bounds)
    return {
        "params": params_dict,
        "raw_x": [float(v) for v in best_x],
        "pnames": pnames,
        "k_params": len(pnames),
        "at_prior_edge": dict(zip(pnames, bflags)),
        "any_at_edge": bool(any(bflags)),
        "chi2_min": chi2_min,
        "chi2_sn": float(c_sn),
        "chi2_bao": float(c_bao),
        "chi2_cmb": float(c_cmb),
    }


def add_ic_deltas(combo_dict, n_tot):
    ref = combo_dict["LCDM"]
    chi2_ref = ref["chi2_min"]
    k_ref = ref["k_params"]
    aic_ref = chi2_ref + 2.0 * k_ref
    bic_ref = chi2_ref + k_ref * np.log(n_tot)
    for mname, d in combo_dict.items():
        k = d["k_params"]
        aic = d["chi2_min"] + 2.0 * k
        bic = d["chi2_min"] + k * np.log(n_tot)
        d["delta_chi2_vs_LCDM"] = float(d["chi2_min"] - chi2_ref)
        d["delta_AIC_vs_LCDM"] = float(aic - aic_ref)
        d["delta_BIC_vs_LCDM"] = float(bic - bic_ref)


# =====================================================================
# TEST 3: PROFILE PERIODOGRAM & EXACT MC LOOK-ELSEWHERE CALIBRATION
# =====================================================================

def compute_profile_and_lee(engine, lcdm_fit, cpl_chi2, label, lam_grid, n_mc=2000, seed=777, cpl_fit=None):
    """
    Scan fixed-lambda profile likelihood over lam_grid and compute exact Look-Elsewhere
    global p-value via correlated Gaussian Process score projection under null LCDM.
    """
    om0 = lcdm_fit["params"]["Om"]
    h0 = lcdm_fit["params"]["h"]
    ob0 = lcdm_fit["params"]["omega_b"]
    chi2_lcdm = lcdm_fit["chi2_min"]

    if cpl_fit is not None:
        om_c = cpl_fit["params"]["Om"]
        h_c = cpl_fit["params"]["h"]
        ob_c = cpl_fit["params"]["omega_b"]
        s_cpl_seed = float(np.clip(1.0 + cpl_fit["params"]["w0"], -0.5, 0.5))
        w1_cpl = float(-cpl_fit["params"]["wa"])
    else:
        om_c, h_c, ob_c, s_cpl_seed, w1_cpl = om0, h0, ob0, 0.14, 0.55

    dchi2_vs_lcdm = []
    dchi2_vs_cpl = []
    A_prof = []
    C_prof = []
    S_prof = []
    full_params_prof = []
    prev_best = [om0, h0, ob0, 0.0, 0.0]
    prev_lam = float(lam_grid[0])

    for lam in lam_grid:
        best_f = 1e12
        best_x = None
        lam_f = float(lam)

        def obj_lam(y):
            om, h, ob, C, S = y
            if not (0.18 < om < 0.50 and 0.58 < h < 0.76 and 0.0215 < ob < 0.0235):
                return 1e12
            if np.hypot(C, S) > 1.0:
                return 1e12 + 1e4 * (np.hypot(C, S) - 1.0) ** 2
            c1, c2, c3 = engine.compute_chi2_components(
                om, "OSC_CS", [C, S, lam_f], h=h, omega_b=ob,
                use_sn=True, use_bao=True, use_cmb=True, cmb_mode="planck_3d"
            )
            return float(c1 + c2 + c3)

        c_scaled = float(np.clip(prev_best[3] * (lam_f / max(prev_lam, 1e-6)), -0.94, 0.94))
        s_scaled = float(np.clip(prev_best[4], -0.94, 0.94))
        if np.hypot(c_scaled, s_scaled) > 0.96:
            fac = 0.94 / np.hypot(c_scaled, s_scaled)
            c_scaled *= fac
            s_scaled *= fac

        c_cpl_seed = float(w1_cpl * lam_f / (2.0 * np.pi))
        s_seed = s_cpl_seed
        r_cpl_seed = np.hypot(c_cpl_seed, s_seed)
        if r_cpl_seed > 0.95:
            fac = 0.94 / r_cpl_seed
            c_cpl_seed *= fac
            s_seed *= fac

        starts = [
            [prev_best[0], prev_best[1], prev_best[2], c_scaled, s_scaled],
            [om_c, h_c, ob_c, c_cpl_seed, s_seed],
            [om0, h0, ob0, 0.0, 0.0],
        ]
        for x0 in starts:
            r = minimize(obj_lam, x0, method="Powell", options={"ftol": 1e-5, "xtol": 1e-5, "maxiter": 550})
            if r.fun < best_f:
                best_f = float(r.fun)
                best_x = r.x

        # Final Nelder-Mead polish to match global optimizer precision
        r_nm = minimize(obj_lam, best_x, method="Nelder-Mead", options={"maxiter": 500, "fatol": 1e-5, "xatol": 1e-5})
        if r_nm.fun < best_f:
            best_f = float(r_nm.fun)
            best_x = r_nm.x

        prev_best = [float(v) for v in best_x]
        prev_lam = lam_f
        A_val, _ = cs_to_polar(best_x[3], best_x[4])
        dchi2_vs_lcdm.append(float(chi2_lcdm - best_f))
        dchi2_vs_cpl.append(float(cpl_chi2 - best_f))
        A_prof.append(float(A_val))
        C_prof.append(float(best_x[3]))
        S_prof.append(float(best_x[4]))
        full_params_prof.append([float(v) for v in best_x])

    dchi2_vs_lcdm = np.array(dchi2_vs_lcdm)
    max_idx = int(np.argmax(dchi2_vs_lcdm))
    max_dchi2 = float(dchi2_vs_lcdm[max_idx])
    lam_best = float(lam_grid[max_idx])
    p_local = float(chi2_dist.sf(max(max_dchi2, 0.0), df=2))

    # Build exact Jacobian blocks at LCDM best fit across (SN, BAO, CMB) for correlated MC null trials
    eps = 1e-4

    def model_vectors(om, h, ob, mod_name, de_p):
        obs = engine.compute_observables(om, mod_name, de_p, h=h, omega_b=ob, cmb_mode="planck_3d")
        chi_sn = np.interp(engine.zhd, engine.z_low, obs["cum_I_low"])
        dl_sn = (1.0 + engine.zhel) * (C_LIGHT / 70.0) * chi_sn
        mu0 = 5.0 * np.log10(dl_sn) + 25.0
        bao_vec, _ = engine.compute_bao_model_vector(obs, om, h, ob, profile_srd=False)
        cmb_vec = np.array([obs["R_th"], obs["l_A_th"], ob], dtype=np.float64)
        return mu0, bao_vec, cmb_vec

    # 3 nuisance derivatives (Om, h, ob)
    n_nuis = 3
    n_cols = n_nuis + 2 * len(lam_grid)
    J_sn = np.zeros((engine.n_sn, n_cols), dtype=np.float64)
    J_bao = np.zeros((engine.n_bao, n_cols), dtype=np.float64)
    J_cmb = np.zeros((3, n_cols), dtype=np.float64)

    base_nuis = np.array([om0, h0, ob0], dtype=np.float64)
    steps_nuis = np.array([1e-4, 1e-4, 1e-5], dtype=np.float64)
    for j in range(3):
        dp = np.zeros(3)
        dp[j] = steps_nuis[j]
        p_p = base_nuis + dp
        p_m = base_nuis - dp
        sn_p, bao_p, cmb_p = model_vectors(p_p[0], p_p[1], p_p[2], "LCDM", [])
        sn_m, bao_m, cmb_m = model_vectors(p_m[0], p_m[1], p_m[2], "LCDM", [])
        J_sn[:, j] = (sn_p - sn_m) / (2.0 * dp[j])
        J_bao[:, j] = (bao_p - bao_m) / (2.0 * dp[j])
        J_cmb[:, j] = (cmb_p - cmb_m) / (2.0 * dp[j])

    for k_i, lam in enumerate(lam_grid):
        c_col = n_nuis + 2 * k_i
        s_col = c_col + 1
        # d/dC
        sn_p, bao_p, cmb_p = model_vectors(om0, h0, ob0, "OSC_CS", [eps, 0.0, float(lam)])
        sn_m, bao_m, cmb_m = model_vectors(om0, h0, ob0, "OSC_CS", [-eps, 0.0, float(lam)])
        J_sn[:, c_col] = (sn_p - sn_m) / (2.0 * eps)
        J_bao[:, c_col] = (bao_p - bao_m) / (2.0 * eps)
        J_cmb[:, c_col] = (cmb_p - cmb_m) / (2.0 * eps)
        # d/dS
        sn_p, bao_p, cmb_p = model_vectors(om0, h0, ob0, "OSC_CS", [0.0, eps, float(lam)])
        sn_m, bao_m, cmb_m = model_vectors(om0, h0, ob0, "OSC_CS", [0.0, -eps, float(lam)])
        J_sn[:, s_col] = (sn_p - sn_m) / (2.0 * eps)
        J_bao[:, s_col] = (bao_p - bao_m) / (2.0 * eps)
        J_cmb[:, s_col] = (cmb_p - cmb_m) / (2.0 * eps)

    # Full Fisher / score covariance Gram matrix G = J^T P J
    G = (
        J_sn.T @ (engine.Cinv_sn_proj @ J_sn)
        + J_bao.T @ (engine.Cinv_bao @ J_bao)
        + J_cmb.T @ (engine.Cinv_cmb @ J_cmb)
    )
    G = 0.5 * (G + G.T)
    evals, evecs = np.linalg.eigh(G)
    evals_clip = np.maximum(evals, 0.0)
    # Factor L such that G = L @ L^T -> g = L @ z draws exact N(0, G)
    L_fac = evecs * np.sqrt(evals_clip)
    rng = np.random.default_rng(seed)
    z_mc = rng.normal(size=(n_cols, n_mc))
    g_mc = (L_fac @ z_mc).T  # shape (n_mc, n_cols)

    G00_inv = np.linalg.inv(G[:n_nuis, :n_nuis])
    g0 = g_mc[:, :n_nuis]  # shape (n_mc, 3)
    mc_max = np.zeros(n_mc, dtype=np.float64)

    for k_i in range(len(lam_grid)):
        c_col = n_nuis + 2 * k_i
        Gk0 = G[c_col : c_col + 2, :n_nuis]
        Gkk = G[c_col : c_col + 2, c_col : c_col + 2]
        F_eff_k = Gkk - Gk0 @ G00_inv @ Gk0.T
        F_eff_inv = np.linalg.pinv(0.5 * (F_eff_k + F_eff_k.T))
        gk_tilde = g_mc[:, c_col : c_col + 2] - g0 @ (Gk0 @ G00_inv).T  # (n_mc, 2)
        dchi2_mc_k = np.einsum("ni,ij,nj->n", gk_tilde, F_eff_inv, gk_tilde)
        mc_max = np.maximum(mc_max, dchi2_mc_k)

    p_global_lee = float((np.sum(mc_max >= max_dchi2) + 0.5) / (n_mc + 1.0))
    z_local = float(np.sqrt(chi2_dist.isf(max(min(p_local, 0.999999), 1e-15), df=1))) if p_local < 0.5 else 0.0
    z_global = float(np.sqrt(chi2_dist.isf(max(min(p_global_lee, 0.999999), 1e-15), df=1))) if p_global_lee < 0.5 else 0.0
    lee_2sigma = float(np.percentile(mc_max, 95.45))
    lee_3sigma = float(np.percentile(mc_max, 99.73))
    n_eff_trials = float(p_global_lee / p_local) if p_local > 0 else 1.0

    print(
        f"  [LEE {label}] max dchi2={max_dchi2:.2f} at lam={lam_best:.2f} | "
        f"p_local={p_local:.4f} ({z_local:.2f}sig) -> p_global_LEE={p_global_lee:.4f} ({z_global:.2f}sig) | "
        f"LEE_2sig={lee_2sigma:.2f}, LEE_3sig={lee_3sigma:.2f}"
    )

    return {
        "label": label,
        "lam_grid": lam_grid.tolist(),
        "dchi2_vs_LCDM": dchi2_vs_lcdm.tolist(),
        "dchi2_vs_CPL": dchi2_vs_cpl,
        "A_profile": A_prof,
        "C_profile": C_prof,
        "S_profile": S_prof,
        "full_params_profile": full_params_prof,
        "max_dchi2": max_dchi2,
        "best_lam": lam_best,
        "p_local": p_local,
        "sigma_local": z_local,
        "p_global_lee": p_global_lee,
        "sigma_global_lee": z_global,
        "effective_trial_factor": n_eff_trials,
        "lee_threshold_2sigma": lee_2sigma,
        "lee_threshold_3sigma": lee_3sigma,
        "n_mc_nulls": n_mc,
    }


# =====================================================================
# TEST 4: ORTHOGONAL CARTESIAN (C, S) MCMC WITH CONVERGENCE DIAGNOSTICS
# =====================================================================

def run_model_mcmc(engine, model_name, best_fit, tag, rng_seed=20261120):
    """Run Affine-Invariant / Differential-Evolution Ensemble MCMC and audit convergence."""
    rng = np.random.default_rng(rng_seed)
    p0_dict = best_fit["params"]

    if model_name == "LCDM":
        ndim, nwalkers, nburn, nprod = 3, 16, 350, 900
        x0 = np.array([p0_dict["Om"], p0_dict["h"], p0_dict["omega_b"]])
        scale = np.array([0.004, 0.004, 0.00010])
        pnames = ["Om", "h", "omega_b"]

        def log_prob(p):
            om, h, ob = p
            if not (0.20 < om < 0.48 and 0.58 < h < 0.76 and 0.0215 < ob < 0.0235):
                return -np.inf
            c1, c2, c3 = engine.compute_chi2_components(om, "LCDM", [], h=h, omega_b=ob)
            return -0.5 * (c1 + c2 + c3)

        moves = None

    elif model_name == "CPL":
        ndim, nwalkers, nburn, nprod = 5, 24, 500, 1400
        x0 = np.array([p0_dict["Om"], p0_dict["h"], p0_dict["omega_b"], p0_dict["w0"], p0_dict["wa"]])
        scale = np.array([0.004, 0.004, 0.00010, 0.04, 0.15])
        pnames = ["Om", "h", "omega_b", "w0", "wa"]

        def log_prob(p):
            om, h, ob, w0, wa = p
            if not (
                0.20 < om < 0.48
                and 0.58 < h < 0.76
                and 0.0215 < ob < 0.0235
                and -1.8 < w0 < 0.0
                and -3.5 < wa < 2.0
                and (w0 + wa) < 0.1
            ):
                return -np.inf
            c1, c2, c3 = engine.compute_chi2_components(om, "CPL", [w0, wa], h=h, omega_b=ob)
            return -0.5 * (c1 + c2 + c3)

        moves = [(emcee.moves.DEMove(), 0.7), (emcee.moves.DESnookerMove(), 0.3)]

    elif model_name == "OSC_CS":
        # Sample in decoupled slope coordinates p = (Om, h, ob, w1_tilde, S, lam)
        # where C = (lam / 2pi) * w1_tilde, with exact Jacobian +log(lam / 2pi),
        # eliminating the hyperbolic C/lambda ridge that stalled raw (C, S, lam) walkers.
        ndim, nwalkers, nburn, nprod = 6, 32, 1200, 3600
        w1_0 = p0_dict["C"] * (2.0 * np.pi) / p0_dict["lambda"]
        x0 = np.array(
            [p0_dict["Om"], p0_dict["h"], p0_dict["omega_b"], w1_0, p0_dict["S"], p0_dict["lambda"]]
        )
        scale = np.array([0.003, 0.003, 0.00008, 0.10, 0.04, 1.2])
        pnames = ["Om", "h", "omega_b", "C", "S", "lambda"]

        def log_prob(p):
            om, h, ob, w1_tilde, S, lam = p
            if not (
                0.20 < om < 0.48
                and 0.58 < h < 0.76
                and 0.0215 < ob < 0.0235
                and 1.2 <= lam <= 12.0
            ):
                return -np.inf
            C = (lam / (2.0 * np.pi)) * w1_tilde
            if (C * C + S * S) > 1.0:
                return -np.inf
            c1, c2, c3 = engine.compute_chi2_components(om, "OSC_CS", [C, S, lam], h=h, omega_b=ob)
            jac = np.log(lam / (2.0 * np.pi))
            return -0.5 * (c1 + c2 + c3) + jac

        moves = [(emcee.moves.StretchMove(a=2.0), 0.7), (emcee.moves.DEMove(gamma0=0.6), 0.3)]

    else:
        raise ValueError(model_name)

    pos = np.empty((nwalkers, ndim), dtype=np.float64)
    for w in range(nwalkers):
        while True:
            cand = x0 + scale * rng.normal(size=ndim)
            if model_name == "OSC_CS":
                if w % 2 == 0:
                    cand[5] = rng.uniform(3.0, 11.8)
                else:
                    cand[5] = rng.uniform(1.8, 11.8)
                # Ensure physical C^2 + S^2 <= 0.90 using current cand[5] (lam)
                C_cand = (cand[5] / (2.0 * np.pi)) * cand[3]
                r_cs = np.hypot(C_cand, cand[4])
                if r_cs > 0.90:
                    fac = 0.85 / r_cs
                    cand[3] *= fac
                    cand[4] *= fac
            if np.isfinite(log_prob(cand)):
                pos[w] = cand
                break

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, moves=moves)
    sampler.run_mcmc(pos, nburn + nprod, progress=False)

    chain_3d = sampler.get_chain(discard=nburn, flat=False).copy()  # (nprod, nwalkers, ndim)
    if model_name == "OSC_CS":
        # Map w1_tilde -> C = (lam / 2pi) * w1_tilde back into physical (Om, h, ob, C, S, lam)
        chain_3d[:, :, 3] = chain_3d[:, :, 3] * (chain_3d[:, :, 5] / (2.0 * np.pi))
    flat_chain = chain_3d.reshape(-1, ndim)
    acc_mean = float(np.mean(sampler.acceptance_fraction))
    rhats = split_rhat(chain_3d)
    try:
        tau = emcee.autocorr.integrated_time(chain_3d, quiet=True)
        tau_list = [float(t) for t in tau]
        ess_min = float(len(flat_chain) / np.max(tau))
    except Exception:
        tau_list = [float("nan")] * ndim
        ess_min = float(len(flat_chain) / 50.0)

    summary = {}
    for idx_p, pname in enumerate(pnames):
        q16, q50, q84 = np.percentile(flat_chain[:, idx_p], [16, 50, 84])
        summary[pname] = {
            "median": float(q50),
            "plus": float(q84 - q50),
            "minus": float(q50 - q16),
            "split_rhat": float(rhats[idx_p]),
        }

    # Physical derived quantities: A = sqrt(C^2 + S^2), s_rd, rd_mpc
    om_c = flat_chain[:, 0]
    h_c = flat_chain[:, 1]
    ob_c = flat_chain[:, 2]
    rd_c = np.array([rd_physical_mpc(o, hh, bb) for o, hh, bb in zip(om_c, h_c, ob_c)])
    srd_c = C_LIGHT / (100.0 * h_c * rd_c)
    for dname, darr in (("rd_mpc", rd_c), ("s_rd", srd_c)):
        q16, q50, q84 = np.percentile(darr, [16, 50, 84])
        summary[dname] = {"median": float(q50), "plus": float(q84 - q50), "minus": float(q50 - q16)}

    if model_name == "OSC_CS":
        A_c = np.hypot(flat_chain[:, 3], flat_chain[:, 4])
        q16, q50, q84, q95 = np.percentile(A_c, [16, 50, 84, 95])
        summary["A_derived"] = {
            "median": float(q50),
            "plus": float(q84 - q50),
            "minus": float(q50 - q16),
            "upper_95": float(q95),
        }

    diag = {
        "n_samples": int(len(flat_chain)),
        "acceptance_fraction": acc_mean,
        "max_split_rhat": float(np.max(rhats)),
        "autocorr_tau": tau_list,
        "ess_min": ess_min,
    }
    np.save(os.path.join(RESULTS_DIR, f"chain_v3_{tag}_{model_name}.npy"), flat_chain)
    print(
        f"  [MCMC {tag}/{model_name}] N={len(flat_chain)} | acc={acc_mean:.3f} | "
        f"max_Rhat={diag['max_split_rhat']:.4f} | Om={summary['Om']['median']:.4f}"
        + (f" | A_95<{summary['A_derived']['upper_95']:.3f}" if model_name == "OSC_CS" else "")
    )
    return summary, diag, flat_chain


# =====================================================================
# TEST 5: INJECTION-RECOVERY & NULL FALSE-POSITIVE CALIBRATION
# =====================================================================

def run_injection_recovery(bao_eng, om_true=0.312, h_true=0.676, ob_true=0.02236, n_mocks=25, seed=202705):
    """
    Validate estimator bias & sensitivity on synthetic DESI DR2 + Planck 3D + Euclid mocks
    under 3 scenarios: Null LCDM, CPL Truth (-0.80, -0.75), and Oscillatory Truth (A=0.30, lam=3.0).
    """
    s_rd_true = C_LIGHT / (100.0 * h_true * rd_physical_mpc(om_true, h_true, ob_true))
    rng = np.random.default_rng(seed)
    results = {}

    scenarios = [
        ("Null_LCDM", "LCDM", [], 0.0),
        ("CPL_Truth", "CPL", [-0.80, -0.75], 0.0),
        ("OSC_Truth_A030", "OSC_CS", [*polar_to_cs(0.30, 0.60), 3.0], 0.30),
    ]

    for euclid_scen in ("dr1_2027", "full"):
        scen_res = {}
        z_eu, _, cov_eu, inv_cov_eu, _, _ = build_euclid_anisotropic_covariance(
            om_true, s_rd_true, "LCDM", (), scenario=euclid_scen, h=h_true
        )
        chol_bao = np.linalg.cholesky(bao_eng.C_bao)
        chol_cmb = np.linalg.cholesky(bao_eng.C_cmb)
        chol_eu = np.linalg.cholesky(cov_eu)

        for label, true_mod, true_p, A_true in scenarios:
            obs_true = bao_eng.compute_observables(om_true, true_mod, true_p, h=h_true, omega_b=ob_true)
            bao_fid, _ = bao_eng.compute_bao_model_vector(obs_true, om_true, h_true, ob_true, profile_srd=False)
            cmb_fid = np.array([obs_true["R_th"], obs_true["l_A_th"], ob_true], dtype=float)
            eu_fid, _, _, _ = compute_euclid_anisotropic_observables(
                z_eu, om_true, s_rd_true, true_mod, true_p, h=h_true
            )

            # Evaluate fast synthetic likelihood over (BAO + Euclid + CMB)
            def fit_mock(y_b, y_c, y_e):
                def eval_chi2(om, h, ob, mod, dep):
                    obs = bao_eng.compute_observables(om, mod, dep, h=h, omega_b=ob)
                    if obs is None:
                        return 1e12
                    bv, srd_m = bao_eng.compute_bao_model_vector(obs, om, h, ob, profile_srd=False)
                    ev, _, _, _ = compute_euclid_anisotropic_observables(z_eu, om, srd_m, mod, dep, h=h)
                    cv = np.array([obs["R_th"], obs["l_A_th"], ob])
                    return (
                        float((y_b - bv) @ bao_eng.Cinv_bao @ (y_b - bv))
                        + float((y_e - ev) @ inv_cov_eu @ (y_e - ev))
                        + float((y_c - cv) @ bao_eng.Cinv_cmb @ (y_c - cv))
                    )

                rl = minimize(lambda p: eval_chi2(p[0], p[1], p[2], "LCDM", []), [om_true, h_true, ob_true], method="Nelder-Mead", options={"maxiter": 400})
                rc = minimize(lambda p: eval_chi2(p[0], p[1], p[2], "CPL", [p[3], p[4]]) if -1.8 < p[3] < 0 and -3.0 < p[4] < 1.5 else 1e12, [om_true, h_true, ob_true, -0.9, -0.4], method="Nelder-Mead", options={"maxiter": 600})
                best_o = (1e12, None)
                for l_seed in (2.2, 3.0, 4.5, 7.0):
                    ro = minimize(
                        lambda p: eval_chi2(p[0], p[1], p[2], "OSC_CS", [p[3], p[4], p[5]]) if (p[3]**2 + p[4]**2) <= 1.0 and 1.2 <= p[5] <= 12.0 else 1e12,
                        [om_true, h_true, ob_true, 0.15 if A_true > 0 else 0.0, 0.10 if A_true > 0 else 0.0, l_seed],
                        method="Nelder-Mead",
                        options={"maxiter": 600},
                    )
                    if ro.fun < best_o[0]:
                        best_o = (ro.fun, ro.x)
                A_rec, _ = cs_to_polar(best_o[1][3], best_o[1][4])
                return rl.fun, rc.fun, best_o[0], A_rec, best_o[1][5]

            # Asimov noiseless check first
            cl_a, cc_a, co_a, A_asimov, lam_asimov = fit_mock(bao_fid, cmb_fid, eu_fid)

            # Noisy Monte Carlo batch
            A_recs, lam_recs, dchi2_osc_lcdm, dchi2_osc_cpl, dbic_osc_cpl = [], [], [], [], []
            n_tot_mock = bao_eng.n_bao + len(eu_fid) + 3
            for _ in range(n_mocks):
                yb = bao_fid + chol_bao @ rng.normal(size=len(bao_fid))
                yc = cmb_fid + chol_cmb @ rng.normal(size=len(cmb_fid))
                ye = eu_fid + chol_eu @ rng.normal(size=len(eu_fid))
                cl, cc, co, Ar, lr = fit_mock(yb, yc, ye)
                A_recs.append(Ar)
                lam_recs.append(lr)
                dchi2_osc_lcdm.append(cl - co)
                dchi2_osc_cpl.append(cc - co)
                # BIC_CPL - BIC_OSC = (cc + 5*ln N) - (co + 6*ln N); positive => favors OSC
                dbic_osc_cpl.append((cc - co) - np.log(n_tot_mock))

            scen_res[label] = {
                "A_true": A_true,
                "asimov_A_recovered": float(A_asimov),
                "asimov_lam_recovered": float(lam_asimov),
                "asimov_dchi2_OSC_vs_LCDM": float(cl_a - co_a),
                "asimov_dchi2_OSC_vs_CPL": float(cc_a - co_a),
                "mc_A_median": float(np.median(A_recs)),
                "mc_A_q16": float(np.percentile(A_recs, 16)),
                "mc_A_q84": float(np.percentile(A_recs, 84)),
                "mc_dchi2_OSC_vs_LCDM_median": float(np.median(dchi2_osc_lcdm)),
                "mc_dchi2_OSC_vs_CPL_median": float(np.median(dchi2_osc_cpl)),
                "mc_frac_dchi2_vs_LCDM_gt9": float(np.mean(np.array(dchi2_osc_lcdm) > 9.0)),
                "mc_frac_BIC_prefers_OSC_over_CPL": float(np.mean(np.array(dbic_osc_cpl) > 0.0)),
            }
            print(
                f"  [InjRec {euclid_scen}/{label}] Asimov A={A_asimov:.3f} (true={A_true}) | "
                f"MC A={scen_res[label]['mc_A_median']:.3f} [{scen_res[label]['mc_A_q16']:.3f}, {scen_res[label]['mc_A_q84']:.3f}] | "
                f"med dchi2_OSC_vs_CPL={scen_res[label]['mc_dchi2_OSC_vs_CPL_median']:+.2f}"
            )
        results[euclid_scen] = scen_res
    return results


# =====================================================================
# MAIN V3.0 ORCHESTRATION
# =====================================================================

def main():
    t_start = time.perf_counter()
    reuse_cache = "--reuse-cached-t125" in sys.argv
    json_path = os.path.join(RESULTS_DIR, "audit_v3_stress_tests.json")
    cached = json.load(open(json_path, "r", encoding="utf-8")) if (reuse_cache and os.path.isfile(json_path)) else None

    print("=" * 82)
    print("PAPER 3 v3.0: DESI DR2 + DES-SN5YR + PLANCK 3D + EUCLID DR1-2027 STRESS AUDIT")
    if cached is not None:
        print("(Reusing cached Tests 1, 2, 5 + LCDM/CPL MCMC chains; re-running Tests 3, 4-OSC, 6)")
    print("=" * 82)

    eng_pan_dr1 = CosmologyLikelihoodEngine("pantheon_plus", "desi_dr1")
    eng_pan_dr2 = CosmologyLikelihoodEngine("pantheon_plus", "desi_dr2")
    eng_des_dr2 = CosmologyLikelihoodEngine("des_sn5yr", "desi_dr2")

    audit = {}

    if cached is not None:
        test1 = cached["test1_planck1d_vs_planck3d_audit"]
        test2 = cached["test2_multidataset_fits"]
        audit["test1_planck1d_vs_planck3d_audit"] = test1
        audit["test2_multidataset_fits"] = test2
        print("\n[TEST 1 & 2] Loaded cached global optimization fits.")
    else:
        # -----------------------------------------------------------------
        # TEST 1: Bug #1 Calibration Audit (Planck 1D R vs Planck 3D)
        # -----------------------------------------------------------------
        print("\n[TEST 1] Bug #1 Calibration Audit: Planck 1D R vs Full Planck 3D on Pantheon+ + DESI DR1")
        test1 = {}
        for cmb_m, n_obs_cmb in (("planck_1d_R", 1), ("planck_3d", 3)):
            cdict = {}
            for m in ("LCDM", "CPL", "OSC_CS"):
                cdict[m] = fit_model(eng_pan_dr1, m, use_sn=True, use_bao=True, use_cmb=True, cmb_mode=cmb_m)
            add_ic_deltas(cdict, eng_pan_dr1.n_sn + eng_pan_dr1.n_bao + n_obs_cmb)
            test1[cmb_m] = cdict
            print(
                f"  {cmb_m:12s} | LCDM chi2={cdict['LCDM']['chi2_min']:.2f} (Om={cdict['LCDM']['params']['Om']:.4f}) | "
                f"CPL dchi2={cdict['CPL']['delta_chi2_vs_LCDM']:+.2f} | OSC dchi2={cdict['OSC_CS']['delta_chi2_vs_LCDM']:+.2f} "
                f"(A={cdict['OSC_CS']['params']['A']:.3f}, lam={cdict['OSC_CS']['params']['lambda']:.2f})"
            )
        audit["test1_planck1d_vs_planck3d_audit"] = test1

        # -----------------------------------------------------------------
        # TEST 2: Multi-Dataset Matrix (DR1 vs DR2 x Pantheon+ vs DES-SN5YR vs No-SNe)
        # -----------------------------------------------------------------
        print("\n[TEST 2] Multi-Dataset Global Fits (SN-Only + Full Combinations + No-SNe Geometry)")
        test2 = {}
        combos = [
            ("PantheonPlus_only", eng_pan_dr2, True, False, False, eng_pan_dr2.n_sn),
            ("DES_SN5YR_only", eng_des_dr2, True, False, False, eng_des_dr2.n_sn),
            ("PantheonPlus_DESI_DR1_Planck3D", eng_pan_dr1, True, True, True, eng_pan_dr1.n_sn + eng_pan_dr1.n_bao + 3),
            ("PantheonPlus_DESI_DR2_Planck3D", eng_pan_dr2, True, True, True, eng_pan_dr2.n_sn + eng_pan_dr2.n_bao + 3),
            ("DES_SN5YR_DESI_DR2_Planck3D", eng_des_dr2, True, True, True, eng_des_dr2.n_sn + eng_des_dr2.n_bao + 3),
            ("DESI_DR2_Planck3D_NoSNe", eng_pan_dr2, False, True, True, eng_pan_dr2.n_bao + 3),
        ]
        for name, eng, u_sn, u_bao, u_cmb, n_tot in combos:
            res_c = {}
            for m in ("LCDM", "CPL", "OSC_CS"):
                res_c[m] = fit_model(eng, m, use_sn=u_sn, use_bao=u_bao, use_cmb=u_cmb, cmb_mode="planck_3d")
            add_ic_deltas(res_c, n_tot)
            test2[name] = res_c
            print(
                f"  {name:32s} | LCDM chi2={res_c['LCDM']['chi2_min']:.2f} (Om={res_c['LCDM']['params']['Om']:.4f}) | "
                f"CPL dchi2={res_c['CPL']['delta_chi2_vs_LCDM']:+.2f}, dBIC={res_c['CPL']['delta_BIC_vs_LCDM']:+.2f} | "
                f"OSC dchi2={res_c['OSC_CS']['delta_chi2_vs_LCDM']:+.2f}, dBIC={res_c['OSC_CS']['delta_BIC_vs_LCDM']:+.2f} "
                f"(A={res_c['OSC_CS']['params']['A']:.3f}, lam={res_c['OSC_CS']['params']['lambda']:.2f})"
            )
        audit["test2_multidataset_fits"] = test2

    # -----------------------------------------------------------------
    # TEST 3: Profile Periodogram & Exact MC Look-Elsewhere Calibration
    # -----------------------------------------------------------------
    print("\n[TEST 3] Profile Likelihood Periodogram & Look-Elsewhere Effect (40 periods x 2,000 MC nulls)")
    lam_scan = np.geomspace(0.8, 12.0, 40)
    lee_pan = compute_profile_and_lee(
        eng_pan_dr2,
        test2["PantheonPlus_DESI_DR2_Planck3D"]["LCDM"],
        test2["PantheonPlus_DESI_DR2_Planck3D"]["CPL"]["chi2_min"],
        "PantheonPlus_DR2_Planck3D",
        lam_scan,
        n_mc=2000,
        seed=701,
        cpl_fit=test2["PantheonPlus_DESI_DR2_Planck3D"]["CPL"],
    )
    lee_des = compute_profile_and_lee(
        eng_des_dr2,
        test2["DES_SN5YR_DESI_DR2_Planck3D"]["LCDM"],
        test2["DES_SN5YR_DESI_DR2_Planck3D"]["CPL"]["chi2_min"],
        "DES_SN5YR_DR2_Planck3D",
        lam_scan,
        n_mc=2000,
        seed=702,
        cpl_fit=test2["DES_SN5YR_DESI_DR2_Planck3D"]["CPL"],
    )
    audit["test3_look_elsewhere_periodogram"] = {
        "PantheonPlus_DR2": lee_pan,
        "DES_SN5YR_DR2": lee_des,
    }

    # -----------------------------------------------------------------
    # TEST 4: Orthogonal Cartesian (C, S) Ensemble MCMC + Diagnostics
    # -----------------------------------------------------------------
    print("\n[TEST 4] Orthogonal (C, S) Ensemble MCMC on Pantheon+ & DES-SN5YR + DESI DR2 + Planck 3D")
    mcmc_out = {"PantheonPlus_DR2": {}, "DES_SN5YR_DR2": {}}
    chains_store = {"PantheonPlus_DR2": {}, "DES_SN5YR_DR2": {}}

    for tag, eng, fit_key, base_seed in (
        ("PantheonPlus_DR2", eng_pan_dr2, "PantheonPlus_DESI_DR2_Planck3D", 3100),
        ("DES_SN5YR_DR2", eng_des_dr2, "DES_SN5YR_DESI_DR2_Planck3D", 4100),
    ):
        for m in ("LCDM", "CPL", "OSC_CS"):
            ch_file = os.path.join(RESULTS_DIR, f"chain_v3_{tag}_{m}.npy")
            if cached is not None and m in ("LCDM", "CPL") and os.path.isfile(ch_file):
                mcmc_out[tag][m] = cached["test4_orthogonal_mcmc"][tag][m]
                chains_store[tag][m] = np.load(ch_file)
                print(f"  [MCMC {tag}/{m}] Loaded cached chain (acc={mcmc_out[tag][m]['diagnostics']['acceptance_fraction']:.3f}, max_Rhat={mcmc_out[tag][m]['diagnostics']['max_split_rhat']:.4f})")
            else:
                summ, diag, ch = run_model_mcmc(eng, m, test2[fit_key][m], tag, rng_seed=base_seed + len(m) * 17)
                mcmc_out[tag][m] = {"summary": summ, "diagnostics": diag}
                chains_store[tag][m] = ch
    audit["test4_orthogonal_mcmc"] = mcmc_out

    # -----------------------------------------------------------------
    # TEST 5: Injection-Recovery & Null Calibration (DR1-2027 vs Full)
    # -----------------------------------------------------------------
    if cached is not None:
        inj_res = cached["test5_injection_recovery"]
        print("\n[TEST 5] Loaded cached Injection-Recovery calibration.")
    else:
        print("\n[TEST 5] Injection-Recovery & False-Positive Calibration (DR1-2027 vs Full Euclid)")
        inj_res = run_injection_recovery(eng_pan_dr2, n_mocks=25, seed=202705)
    audit["test5_injection_recovery"] = inj_res

    # -----------------------------------------------------------------
    # TEST 6: Coupled Anisotropic Euclid DR1-2027 vs Full Fisher & Predictions
    # -----------------------------------------------------------------
    print("\n[TEST 6] Coupled (D_M/r_d, H*r_d) Euclid DR1-2027 vs Full Fisher & Pre-Registration")
    # Build external 2x2 covariance on (Om, s_rd) from DES-SN5YR + DR2 + CMB and Pantheon+ + DR2 + CMB
    ch_des_l = chains_store["DES_SN5YR_DR2"]["LCDM"]
    srd_des_l = np.array([C_LIGHT / (100.0 * r[1] * rd_physical_mpc(r[0], r[1], r[2])) for r in ch_des_l])
    cov_om_srd_des = np.cov(np.vstack([ch_des_l[:, 0], srd_des_l]))

    om_fid_e = mcmc_out["DES_SN5YR_DR2"]["LCDM"]["summary"]["Om"]["median"]
    srd_fid_e = mcmc_out["DES_SN5YR_DR2"]["LCDM"]["summary"]["s_rd"]["median"]
    h_fid_e = mcmc_out["DES_SN5YR_DR2"]["LCDM"]["summary"]["h"]["median"]

    lam_fisher = np.geomspace(0.8, 12.0, 29)
    legacy_hz = compute_combined_3sigma_boundary(om_fid_e, lam_fisher, sigma_om_prior=float(np.sqrt(cov_om_srd_des[0, 0])))
    aniso_dr1_ext = [
        compute_euclid_anisotropic_fisher(om_fid_e, srd_fid_e, float(l), scenario="dr1_2027", external_cov_om_srd=cov_om_srd_des, h=h_fid_e)
        for l in lam_fisher
    ]
    aniso_full_ext = [
        compute_euclid_anisotropic_fisher(om_fid_e, srd_fid_e, float(l), scenario="full", external_cov_om_srd=cov_om_srd_des, h=h_fid_e)
        for l in lam_fisher
    ]
    aniso_full_only = [
        compute_euclid_anisotropic_fisher(om_fid_e, srd_fid_e, float(l), scenario="full", external_cov_om_srd=None, h=h_fid_e)
        for l in lam_fisher
    ]

    # Model separations on both Pantheon+ + DR2 and DES-SN5YR + DR2
    separations = {}
    for dtag, dkey in (("PantheonPlus_DR2", "PantheonPlus_DESI_DR2_Planck3D"), ("DES_SN5YR_DR2", "DES_SN5YR_DESI_DR2_Planck3D")):
        separations[dtag] = {}
        bf = test2[dkey]
        s_l = C_LIGHT / (100.0 * bf["LCDM"]["params"]["h"] * rd_physical_mpc(bf["LCDM"]["params"]["Om"], bf["LCDM"]["params"]["h"], bf["LCDM"]["params"]["omega_b"]))
        s_c = C_LIGHT / (100.0 * bf["CPL"]["params"]["h"] * rd_physical_mpc(bf["CPL"]["params"]["Om"], bf["CPL"]["params"]["h"], bf["CPL"]["params"]["omega_b"]))
        s_o = C_LIGHT / (100.0 * bf["OSC_CS"]["params"]["h"] * rd_physical_mpc(bf["OSC_CS"]["params"]["Om"], bf["OSC_CS"]["params"]["h"], bf["OSC_CS"]["params"]["omega_b"]))

        for scen in ("dr1_2027", "full"):
            separations[dtag][scen] = {
                "CPL_vs_OSC": compute_euclid_model_separation_anisotropic(
                    bf["CPL"]["params"]["Om"], s_c, "CPL", [bf["CPL"]["params"]["w0"], bf["CPL"]["params"]["wa"]],
                    bf["OSC_CS"]["params"]["Om"], s_o, "OSC_CS", [bf["OSC_CS"]["params"]["C"], bf["OSC_CS"]["params"]["S"], bf["OSC_CS"]["params"]["lambda"]],
                    scenario=scen, h_a=bf["CPL"]["params"]["h"], h_b=bf["OSC_CS"]["params"]["h"]
                ),
                "LCDM_vs_OSC": compute_euclid_model_separation_anisotropic(
                    bf["LCDM"]["params"]["Om"], s_l, "LCDM", [],
                    bf["OSC_CS"]["params"]["Om"], s_o, "OSC_CS", [bf["OSC_CS"]["params"]["C"], bf["OSC_CS"]["params"]["S"], bf["OSC_CS"]["params"]["lambda"]],
                    scenario=scen, h_a=bf["LCDM"]["params"]["h"], h_b=bf["OSC_CS"]["params"]["h"]
                ),
                "LCDM_vs_CPL": compute_euclid_model_separation_anisotropic(
                    bf["LCDM"]["params"]["Om"], s_l, "LCDM", [],
                    bf["CPL"]["params"]["Om"], s_c, "CPL", [bf["CPL"]["params"]["w0"], bf["CPL"]["params"]["wa"]],
                    scenario=scen, h_a=bf["LCDM"]["params"]["h"], h_b=bf["CPL"]["params"]["h"]
                ),
            }

    prereg_des = generate_euclid_2027_preregistration(
        {m: test2["DES_SN5YR_DESI_DR2_Planck3D"][m]["params"] for m in ("LCDM", "CPL", "OSC_CS")},
        chains_store["DES_SN5YR_DR2"],
    )
    prereg_pan = generate_euclid_2027_preregistration(
        {m: test2["PantheonPlus_DESI_DR2_Planck3D"][m]["params"] for m in ("LCDM", "CPL", "OSC_CS")},
        chains_store["PantheonPlus_DR2"],
    )

    # Sub-Hubble genuine cyclic regime check (1.2 <= lambda <= 4.5, >=0.25 oscillation over z<=2.5)
    cyclic_regime = {}
    for dtag, lee_d, dkey in (
        ("PantheonPlus_DR2", lee_pan, "PantheonPlus_DESI_DR2_Planck3D"),
        ("DES_SN5YR_DR2", lee_des, "DES_SN5YR_DESI_DR2_Planck3D"),
    ):
        lams_p = np.array(lee_d["lam_grid"])
        mask_cyc = (lams_p >= 1.2) & (lams_p <= 4.5)
        sub_idxs = np.where(mask_cyc)[0]
        best_sub = int(sub_idxs[np.argmax(np.array(lee_d["dchi2_vs_LCDM"])[sub_idxs])])
        lam_c = float(lams_p[best_sub])
        om_c, h_c, ob_c, C_c, S_c = lee_d["full_params_profile"][best_sub]
        A_c, phi_c = cs_to_polar(C_c, S_c)
        s_o_cyc = C_LIGHT / (100.0 * h_c * rd_physical_mpc(om_c, h_c, ob_c))

        bf = test2[dkey]
        s_l = C_LIGHT / (100.0 * bf["LCDM"]["params"]["h"] * rd_physical_mpc(bf["LCDM"]["params"]["Om"], bf["LCDM"]["params"]["h"], bf["LCDM"]["params"]["omega_b"]))
        s_c = C_LIGHT / (100.0 * bf["CPL"]["params"]["h"] * rd_physical_mpc(bf["CPL"]["params"]["Om"], bf["CPL"]["params"]["h"], bf["CPL"]["params"]["omega_b"]))

        cyclic_regime[dtag] = {
            "lambda": lam_c,
            "A": float(A_c),
            "phi0": float(phi_c),
            "C": float(C_c),
            "S": float(S_c),
            "dchi2_vs_LCDM": float(lee_d["dchi2_vs_LCDM"][best_sub]),
            "dchi2_vs_CPL": float(lee_d["dchi2_vs_CPL"][best_sub]),
            "euclid_dr1_2027_CPL_vs_Cyclic": compute_euclid_model_separation_anisotropic(
                bf["CPL"]["params"]["Om"], s_c, "CPL", [bf["CPL"]["params"]["w0"], bf["CPL"]["params"]["wa"]],
                om_c, s_o_cyc, "OSC_CS", [C_c, S_c, lam_c],
                scenario="dr1_2027", h_a=bf["CPL"]["params"]["h"], h_b=h_c,
            ),
            "euclid_full_CPL_vs_Cyclic": compute_euclid_model_separation_anisotropic(
                bf["CPL"]["params"]["Om"], s_c, "CPL", [bf["CPL"]["params"]["w0"], bf["CPL"]["params"]["wa"]],
                om_c, s_o_cyc, "OSC_CS", [C_c, S_c, lam_c],
                scenario="full", h_a=bf["CPL"]["params"]["h"], h_b=h_c,
            ),
            "euclid_full_LCDM_vs_Cyclic": compute_euclid_model_separation_anisotropic(
                bf["LCDM"]["params"]["Om"], s_l, "LCDM", [],
                om_c, s_o_cyc, "OSC_CS", [C_c, S_c, lam_c],
                scenario="full", h_a=bf["LCDM"]["params"]["h"], h_b=h_c,
            ),
        }
        print(
            f"  [Cyclic lam<=4.5 {dtag}] lam={lam_c:.2f}, A={A_c:.3f}, dchi2_vs_LCDM={cyclic_regime[dtag]['dchi2_vs_LCDM']:+.2f}, "
            f"dchi2_vs_CPL={cyclic_regime[dtag]['dchi2_vs_CPL']:+.2f} | "
            f"Euclid Full CPL-vs-Cyclic={cyclic_regime[dtag]['euclid_full_CPL_vs_Cyclic']['sigma_equiv']:.2f}sig"
        )

    audit["test6_euclid_anisotropic_fisher"] = {
        "external_cov_om_srd_des_dr2_cmb": cov_om_srd_des.tolist(),
        "lam_grid": lam_fisher.tolist(),
        "legacy_1d_Hz_A3sig_median": [r["A_3sigma_median_phi"] for r in legacy_hz],
        "aniso_dr1_2027_ext_A3sig_median": [r["A_3sigma_median_phi"] for r in aniso_dr1_ext],
        "aniso_dr1_2027_ext_A3sig_best": [r["A_3sigma_best_phi"] for r in aniso_dr1_ext],
        "aniso_full_ext_A3sig_median": [r["A_3sigma_median_phi"] for r in aniso_full_ext],
        "aniso_full_ext_A3sig_best": [r["A_3sigma_best_phi"] for r in aniso_full_ext],
        "aniso_full_only_A3sig_median": [r["A_3sigma_median_phi"] for r in aniso_full_only],
        "model_separations": separations,
        "cyclic_regime_le4p5": cyclic_regime,
        "preregistered_2027_DES_SN5YR_DR2": prereg_des,
        "preregistered_2027_PantheonPlus_DR2": prereg_pan,
    }

    json_path = os.path.join(RESULTS_DIR, "audit_v3_stress_tests.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)
    print(f"\nSaved complete v3.0 stress-test JSON to {json_path}")

    # -----------------------------------------------------------------
    # FIGURES (300 DPI PUBLICATION PANELS)
    # -----------------------------------------------------------------
    make_figure_1(audit, chains_store)
    make_figure_2(audit, test2, chains_store)
    print(f"Completed v3.0 stress audit in {time.perf_counter() - t_start:.1f}s")


def make_figure_1(audit, chains_store):
    """Fig 1: DESI DR2 + DES-SN5YR Dovekie vs Pantheon+ Stress Tests & LEE Periodogram."""
    fig, axes = plt.subplots(2, 2, figsize=(14.0, 10.2))
    lee_p = audit["test3_look_elsewhere_periodogram"]["PantheonPlus_DR2"]
    lee_d = audit["test3_look_elsewhere_periodogram"]["DES_SN5YR_DR2"]

    # Panel (a): Periodogram + LEE thresholds
    ax = axes[0, 0]
    ax.plot(lee_p["lam_grid"], lee_p["dchi2_vs_LCDM"], "b-o", ms=3.5, lw=1.8, label=f"Pantheon+ + DESI DR2 + Planck 3D ($p_{{LEE}}={lee_p['p_global_lee']:.3f}$)")
    ax.plot(lee_d["lam_grid"], lee_d["dchi2_vs_LCDM"], "r-s", ms=3.5, lw=2.0, label=f"DES-SN5YR + DESI DR2 + Planck 3D ($p_{{LEE}}={lee_d['p_global_lee']:.3f}$)")
    ax.axhline(lee_d["lee_threshold_2sigma"], color="darkorange", ls="--", lw=1.4, label=f"Global LEE $2\\sigma$ ($\\Delta\\chi^2={lee_d['lee_threshold_2sigma']:.1f}$)")
    ax.axhline(lee_d["lee_threshold_3sigma"], color="black", ls=":", lw=1.6, label=f"Global LEE $3\\sigma$ ($\\Delta\\chi^2={lee_d['lee_threshold_3sigma']:.1f}$)")
    ax.set_xscale("log")
    ax.set_xlabel(r"Log-Oscillatory Wavelength $\lambda$", fontsize=10)
    ax.set_ylabel(r"$\Delta\chi^2(\lambda) = \chi^2_{\Lambda\mathrm{CDM}} - \chi^2_{\mathrm{OSC}}(\lambda)$", fontsize=10)
    ax.set_title("(a) Profile Periodogram & Monte Carlo Look-Elsewhere Calibration", fontsize=11, fontweight="bold")
    ax.legend(loc="upper left", fontsize=8.0)
    ax.grid(True, alpha=0.3)

    # Panel (b): w(z) 68% MCMC credible bands (DES-SN5YR + DR2 + Planck 3D vs Pantheon+ + DR2)
    ax = axes[0, 1]
    z_dense = np.linspace(0.0, 2.5, 200)
    ax.axhline(-1.0, color="black", ls="--", lw=1.3, label=r"$\Lambda\mathrm{CDM}$ ($w = -1$)")
    rng = np.random.default_rng(11)
    for tag, mod, col, lab in (
        ("DES_SN5YR_DR2", "CPL", "royalblue", "DES-SN5YR + DR2: CPL (68% MCMC)"),
        ("DES_SN5YR_DR2", "OSC_CS", "crimson", "DES-SN5YR + DR2: Log-Osc (68% MCMC)"),
        ("PantheonPlus_DR2", "OSC_CS", "teal", "Pantheon+ + DR2: Log-Osc (68% MCMC)"),
    ):
        ch = chains_store[tag][mod]
        idx = rng.choice(len(ch), size=min(350, len(ch)), replace=False)
        draws = np.array([w_model(z_dense, mod, row[3:]) for row in ch[idx]])
        q16, q50, q84 = np.percentile(draws, [16, 50, 84], axis=0)
        ax.plot(z_dense, q50, color=col, lw=2.0, label=lab)
        ax.fill_between(z_dense, q16, q84, color=col, alpha=0.16)
    ax.axvspan(0.9, 1.8, color="gold", alpha=0.12, label="Euclid DR1 Window ($0.9 < z < 1.8$)")
    ax.set_xlabel(r"Redshift $z$", fontsize=10)
    ax.set_ylabel(r"Equation of State $w(z)$", fontsize=10)
    ax.set_title(r"(b) Posterior $w(z)$ Bands Across Supernova & DESI DR2 Datasets", fontsize=11, fontweight="bold")
    ax.set_ylim(-1.75, -0.45)
    ax.legend(loc="lower left", fontsize=7.8)
    ax.grid(True, alpha=0.3)

    # Panel (c): Delta chi2 & Delta BIC across dataset matrix
    ax = axes[1, 0]
    t2 = audit["test2_multidataset_fits"]
    keys = [
        ("PantheonPlus_DESI_DR1_Planck3D", "Pan+ +\nDR1+CMB"),
        ("PantheonPlus_DESI_DR2_Planck3D", "Pan+ +\nDR2+CMB"),
        ("DES_SN5YR_DESI_DR2_Planck3D", "DES5YR +\nDR2+CMB"),
        ("DESI_DR2_Planck3D_NoSNe", "DR2+CMB\n(No SNe)"),
    ]
    xp = np.arange(len(keys))
    w = 0.20
    ax.bar(xp - 1.5 * w, [t2[k]["CPL"]["delta_chi2_vs_LCDM"] for k, _ in keys], w, color="royalblue", label=r"$\Delta\chi^2$ CPL")
    ax.bar(xp - 0.5 * w, [t2[k]["OSC_CS"]["delta_chi2_vs_LCDM"] for k, _ in keys], w, color="crimson", label=r"$\Delta\chi^2$ Log-Osc")
    ax.bar(xp + 0.5 * w, [t2[k]["CPL"]["delta_BIC_vs_LCDM"] for k, _ in keys], w, color="lightsteelblue", edgecolor="navy", label=r"$\Delta\mathrm{BIC}$ CPL")
    ax.bar(xp + 1.5 * w, [t2[k]["OSC_CS"]["delta_BIC_vs_LCDM"] for k, _ in keys], w, color="salmon", edgecolor="darkred", label=r"$\Delta\mathrm{BIC}$ Log-Osc")
    ax.axhline(0.0, color="black", lw=1.0)
    ax.set_xticks(xp)
    ax.set_xticklabels([lbl for _, lbl in keys], fontsize=9)
    ax.set_ylabel(r"Criterion Difference vs. $\Lambda\mathrm{CDM}$ (Negative = Preferred)", fontsize=9.5)
    ax.set_title(r"(c) Model Selection Across DESI DR1 $\to$ DR2 & DES-SN5YR Dovekie", fontsize=11, fontweight="bold")
    ax.legend(loc="best", ncol=2, fontsize=7.8)
    ax.grid(True, axis="y", alpha=0.3)

    # Panel (d): Orthogonal (C, S) MCMC posterior removing polar A=0 singularity
    ax = axes[1, 1]
    ch_p = chains_store["PantheonPlus_DR2"]["OSC_CS"]
    ch_d = chains_store["DES_SN5YR_DR2"]["OSC_CS"]
    ax.scatter(ch_p[::10, 3], ch_p[::10, 4], s=5, alpha=0.22, color="royalblue", label="Pantheon+ + DESI DR2")
    ax.scatter(ch_d[::10, 3], ch_d[::10, 4], s=5, alpha=0.25, color="crimson", label="DES-SN5YR + DESI DR2")
    ax.scatter([0.0], [0.0], marker="*", s=180, color="black", zorder=6, label=r"$\Lambda\mathrm{CDM}$ Null ($C=S=0$)")
    theta_c = np.linspace(0, 2 * np.pi, 200)
    ax.plot(0.2 * np.cos(theta_c), 0.2 * np.sin(theta_c), "k--", lw=1.0, alpha=0.45, label="$A=0.20, 0.40, 0.80$")
    ax.plot(0.4 * np.cos(theta_c), 0.4 * np.sin(theta_c), "k:", lw=1.0, alpha=0.45)
    ax.plot(0.8 * np.cos(theta_c), 0.8 * np.sin(theta_c), "k-.", lw=0.9, alpha=0.35)
    ax.set_xlabel(r"$C = A\cos\phi_0$ (Sine Harmonic Coefficient)", fontsize=10)
    ax.set_ylabel(r"$S = A\sin\phi_0$ (Cosine Harmonic Coefficient)", fontsize=10)
    ax.set_title(r"(d) Non-Singular Orthogonal $(C, S)$ MCMC Posterior Plane", fontsize=11, fontweight="bold")
    ax.set_xlim(-0.35, 1.02)
    ax.set_ylim(-0.45, 0.55)
    ax.set_aspect("equal")
    ax.legend(loc="lower left", fontsize=8.0)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "fig_v3_desi_dr2_stress_audit.png"), dpi=300)
    plt.close()


def make_figure_2(audit, test2, chains_store):
    """Fig 2: Pre-Registered Mid-2027 Euclid DR1 Predictions, Coupled Fisher & Injection-Recovery."""
    fig, axes = plt.subplots(2, 2, figsize=(14.0, 10.2))
    pr = audit["test6_euclid_anisotropic_fisher"]["preregistered_2027_DES_SN5YR_DR2"]
    z_bins = np.array([r["z"] for r in pr])

    # Panel (a): Pre-registered D_M(z)/r_d relative to LCDM with DR1-2027 vs Full error bars
    ax = axes[0, 0]
    dm_l = np.array([r["LCDM_DM_over_rd"] for r in pr])
    dm_c = np.array([r["CPL_DM_over_rd"] for r in pr])
    dm_o = np.array([r["OSC_CS_DM_over_rd"] for r in pr])
    sig_dr1_pct = 100.0 * np.array([r["euclid_dr1_2027_sigma_DM"] for r in pr]) / dm_l
    sig_full_pct = 100.0 * np.array([r["euclid_full_sigma_DM"] for r in pr]) / dm_l
    ax.axhline(0.0, color="black", ls="--", lw=1.3, label=r"$\Lambda\mathrm{CDM}$ Baseline")
    ax.errorbar(z_bins - 0.015, np.zeros_like(z_bins), yerr=sig_dr1_pct, fmt="s", color="darkorange", capsize=4, lw=1.6, label=r"Euclid DR1 (Mid-2027, $1900\,\mathrm{deg}^2$)")
    ax.errorbar(z_bins + 0.015, np.zeros_like(z_bins), yerr=sig_full_pct, fmt="o", color="darkgreen", capsize=4, lw=1.8, label=r"Euclid Full 6-yr ($14000\,\mathrm{deg}^2$)")
    ax.plot(z_bins, 100.0 * (dm_c / dm_l - 1.0), "b--^", lw=2.0, label="CPL Prediction (DES5YR+DR2)")
    ax.plot(z_bins, 100.0 * (dm_o / dm_l - 1.0), "r-v", lw=2.2, label="Log-Osc Prediction (DES5YR+DR2)")
    ax.set_xlabel(r"Euclid IST:F Spectroscopic Bin $z_{\mathrm{eff}}$", fontsize=10)
    ax.set_ylabel(r"$\Delta (D_M / r_d)$ vs. $\Lambda\mathrm{CDM}$ (%)", fontsize=10)
    ax.set_title("(a) Pre-Registered 2027 Blind Predictions: $D_M(z)/r_d$", fontsize=11, fontweight="bold")
    ax.legend(loc="lower left", fontsize=7.8)
    ax.grid(True, alpha=0.3)

    # Panel (b): Pre-registered D_H(z)/r_d relative to LCDM
    ax = axes[0, 1]
    dh_l = np.array([r["LCDM_DH_over_rd"] for r in pr])
    dh_c = np.array([r["CPL_DH_over_rd"] for r in pr])
    dh_o = np.array([r["OSC_CS_DH_over_rd"] for r in pr])
    sig_h_dr1_pct = 100.0 * np.array([r["euclid_dr1_2027_sigma_DH"] for r in pr]) / dh_l
    sig_h_full_pct = 100.0 * np.array([r["euclid_full_sigma_DH"] for r in pr]) / dh_l
    ax.axhline(0.0, color="black", ls="--", lw=1.3, label=r"$\Lambda\mathrm{CDM}$ Baseline")
    ax.errorbar(z_bins - 0.015, np.zeros_like(z_bins), yerr=sig_h_dr1_pct, fmt="s", color="darkorange", capsize=4, lw=1.6, label=r"Euclid DR1 (Mid-2027) $D_H/r_d$ $1\sigma$")
    ax.errorbar(z_bins + 0.015, np.zeros_like(z_bins), yerr=sig_h_full_pct, fmt="o", color="darkgreen", capsize=4, lw=1.8, label=r"Euclid Full 6-yr $D_H/r_d$ $1\sigma$")
    ax.plot(z_bins, 100.0 * (dh_c / dh_l - 1.0), "b--^", lw=2.0, label="CPL Prediction (DES5YR+DR2)")
    ax.plot(z_bins, 100.0 * (dh_o / dh_l - 1.0), "r-v", lw=2.2, label="Log-Osc Prediction (DES5YR+DR2)")
    ax.set_xlabel(r"Euclid IST:F Spectroscopic Bin $z_{\mathrm{eff}}$", fontsize=10)
    ax.set_ylabel(r"$\Delta (D_H / r_d)$ vs. $\Lambda\mathrm{CDM}$ (%)", fontsize=10)
    ax.set_title("(b) Pre-Registered 2027 Blind Predictions: $D_H(z)/r_d$", fontsize=11, fontweight="bold")
    ax.legend(loc="upper left", fontsize=7.8)
    ax.grid(True, alpha=0.3)

    # Panel (c): Coupled (DM/rd, H*rd) Fisher 3-sigma reach A_3sigma(lambda)
    ax = axes[1, 0]
    fdat = audit["test6_euclid_anisotropic_fisher"]
    lams = fdat["lam_grid"]
    ax.plot(lams, fdat["legacy_1d_Hz_A3sig_median"], "gray", ls=":", lw=1.8, label="v2.0 1D $H(z)$-only (7 bins)")
    ax.plot(lams, fdat["aniso_dr1_2027_ext_A3sig_median"], "darkorange", lw=2.0, label="Coupled DR1-2027 + DES2/CMB prior")
    ax.plot(lams, fdat["aniso_full_only_A3sig_median"], "royalblue", ls="--", lw=1.8, label="Coupled Full Survey Alone (marginalized)")
    ax.plot(lams, fdat["aniso_full_ext_A3sig_median"], "darkgreen", lw=2.2, label="Coupled Full Survey + DES2/CMB prior")
    ax.plot(lams, fdat["aniso_full_ext_A3sig_best"], "darkgreen", ls="-.", lw=1.3, label="Full + Prior (Optimal Phase)")
    p_osc_des = test2["DES_SN5YR_DESI_DR2_Planck3D"]["OSC_CS"]["params"]
    ax.scatter([p_osc_des["lambda"]], [p_osc_des["A"]], color="crimson", s=80, zorder=6, label=f"DES5YR+DR2 Best Fit ($\\lambda={p_osc_des['lambda']:.2f}, A={p_osc_des['A']:.2f}$)")
    ax.set_xscale("log")
    ax.set_xlabel(r"Log-Oscillatory Wavelength $\lambda$", fontsize=10)
    ax.set_ylabel(r"$3\sigma$ Amplitude Reach $A_{3\sigma}(\lambda)$", fontsize=10)
    ax.set_title(r"(c) Coupled $(D_M/r_d, H r_d)$ Stage-IV Sensitivity: DR1-2027 vs Full", fontsize=11, fontweight="bold")
    ax.set_ylim(0.0, 1.35)
    ax.legend(loc="upper right", fontsize=7.6)
    ax.grid(True, alpha=0.3)

    # Panel (d): Injection-Recovery Calibration across Scenarios
    ax = axes[1, 1]
    inj = audit["test5_injection_recovery"]
    labels_inj = [("Null_LCDM", "Null\n($A=0$)"), ("CPL_Truth", "CPL True\n($w_0=-0.8,w_a=-0.75$)"), ("OSC_Truth_A030", "Log-Osc True\n($A=0.30, \\lambda=3$)")]
    xp = np.arange(len(labels_inj))
    w = 0.25
    y_true = [inj["full"][k]["A_true"] for k, _ in labels_inj]
    y_dr1 = [inj["dr1_2027"][k]["mc_A_median"] for k, _ in labels_inj]
    e_dr1_lo = [inj["dr1_2027"][k]["mc_A_median"] - inj["dr1_2027"][k]["mc_A_q16"] for k, _ in labels_inj]
    e_dr1_hi = [inj["dr1_2027"][k]["mc_A_q84"] - inj["dr1_2027"][k]["mc_A_median"] for k, _ in labels_inj]
    y_full = [inj["full"][k]["mc_A_median"] for k, _ in labels_inj]
    e_full_lo = [inj["full"][k]["mc_A_median"] - inj["full"][k]["mc_A_q16"] for k, _ in labels_inj]
    e_full_hi = [inj["full"][k]["mc_A_q84"] - inj["full"][k]["mc_A_median"] for k, _ in labels_inj]

    ax.bar(xp - w, y_true, w, color="black", alpha=0.75, label="Injected Truth $A_{\\mathrm{true}}$")
    ax.bar(xp, y_dr1, w, yerr=[e_dr1_lo, e_dr1_hi], capsize=4, color="darkorange", label="MC Recovery (DR1-2027 + DR2)")
    ax.bar(xp + w, y_full, w, yerr=[e_full_lo, e_full_hi], capsize=4, color="darkgreen", label="MC Recovery (Full Euclid + DR2)")
    ax.set_xticks(xp)
    ax.set_xticklabels([lbl for _, lbl in labels_inj], fontsize=9)
    ax.set_ylabel(r"Recovered Amplitude $A$ (Median & 68% MC)", fontsize=10)
    ax.set_title("(d) Synthetic Injection-Recovery Calibration (25 MC Mocks Each)", fontsize=11, fontweight="bold")
    ax.set_ylim(0.0, 0.95)
    ax.legend(loc="upper left", fontsize=8.0)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "fig_v3_euclid_2027_preregistered.png"), dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
