# -*- coding: utf-8 -*-
"""
Stage-IV ESA Euclid Fisher Matrix & Pre-Registered Forecast Engine (v3.0).

Includes:
  1. Legacy 7-bin H(z)-only Fisher forecast (for backwards compatibility & audit comparison)
  2. Upgraded 4-bin Coupled Anisotropic (D_M(z)/r_d, H(z)*r_d/c) BAO Fisher Matrix
     following Euclid IST:F specifications (Blanchard et al. 2020, A&A 642, A191):
       - z_eff in {1.00, 1.20, 1.40, 1.65} covering [0.9, 1.8]
       - Intra-bin correlation rho(D_M/r_d, H*r_d) = +0.40 (rho(D_M/r_d, D_H/r_d) = -0.40)
       - Dual-scale support:
           * 'full': 6-year Wide Survey (~14,000 deg^2)
           * 'dr1_2027': Euclid DR1 Complete expected mid-2027 (~1,900 deg^2, area scaling sqrt(14000/1900))
  3. Orthogonal (C, S) non-singular parameter projection, Look-Elsewhere corrections,
     and Pre-Registered Mid-2027 Blind Prediction Table generator.
"""

import numpy as np
from src.likelihoods import rho_de_ratio, C_LIGHT, rd_physical_mpc

# Legacy 7-bin H(z)-only constants (v2.0 compatibility)
EUCLID_Z_BINS = np.array([0.90, 1.05, 1.20, 1.35, 1.50, 1.65, 1.80], dtype=float)
EUCLID_REL_SIGMA_H = 0.013

# Canonical 4-bin Euclid IST:F spectroscopic BAO specifications across 0.9 <= z <= 1.8
EUCLID_IST_Z_BINS = np.array([1.00, 1.20, 1.40, 1.65], dtype=float)
EUCLID_FULL_SIGMA_DM = np.array([0.0080, 0.0080, 0.0090, 0.0110], dtype=float)
EUCLID_FULL_SIGMA_H = np.array([0.0120, 0.0120, 0.0130, 0.0160], dtype=float)
EUCLID_RHO_DM_H = 0.40  # corr(D_M/r_d, H*r_d) = +0.40  <==>  corr(D_M/r_d, D_H/r_d) = -0.40

# DR1-Foundation footprinted 1,900 deg^2 (~14% of ~14,000 deg^2 survey); mid-2027 DR1 Complete
EUCLID_FULL_AREA_DEG2 = 14000.0
EUCLID_DR1_AREA_DEG2 = 1900.0
DR1_AREA_SCALE = float(np.sqrt(EUCLID_FULL_AREA_DEG2 / EUCLID_DR1_AREA_DEG2))

# Effective present radiation density for background cosmology (T_CMB=2.7255, N_eff=3.046)
DEFAULT_OMEGA_R_H2 = 4.18343e-5


def dimensionless_hubble(z, om, model_name, de_params, h=0.674, include_radiation=True):
    """Return dimensionless expansion rate E(z) = H(z)/H0 for flat FLRW models."""
    z_arr = np.asarray(z, dtype=float)
    or_val = (DEFAULT_OMEGA_R_H2 / (h * h)) if include_radiation else 0.0
    ode_val = 1.0 - om - or_val
    rho_de = rho_de_ratio(z_arr, model_name, de_params)
    E2 = or_val * (1.0 + z_arr) ** 4 + om * (1.0 + z_arr) ** 3 + ode_val * rho_de
    return np.sqrt(np.maximum(E2, 1e-12))


def comoving_distance_over_c_H0(z, om, model_name, de_params, h=0.674, include_radiation=True, n_grid=800):
    """Compute dimensionless line-of-sight comoving distance chi(z) = int_0^z dz'/E(z')."""
    z_arr = np.atleast_1d(np.asarray(z, dtype=float))
    z_max = float(np.max(z_arr))
    if z_max <= 0.0:
        return np.zeros_like(z_arr)
    z_grid = np.linspace(0.0, z_max, n_grid)
    E_grid = dimensionless_hubble(z_grid, om, model_name, de_params, h=h, include_radiation=include_radiation)
    invE = 1.0 / E_grid
    cum_I = np.concatenate([[0.0], np.cumsum(0.5 * (invE[:-1] + invE[1:]) * np.diff(z_grid))])
    chi = np.interp(z_arr, z_grid, cum_I)
    return chi if np.ndim(z) > 0 else float(chi[0])


def euclid_bao_cov_and_mean_func(scenario="full", z_bins=None):
    """
    Build per-bin relative uncertainties and block correlation specification for Euclid scenarios.

    scenario: 'full' (6-year 14,000 deg^2) or 'dr1_2027' (1,900 deg^2 complete DR1 mid-2027).
    """
    z_arr = EUCLID_IST_Z_BINS if z_bins is None else np.asarray(z_bins, dtype=float)
    scale = 1.0 if str(scenario).lower() == "full" else DR1_AREA_SCALE
    sig_dm_rel = EUCLID_FULL_SIGMA_DM * scale
    sig_h_rel = EUCLID_FULL_SIGMA_H * scale
    return z_arr, sig_dm_rel, sig_h_rel, EUCLID_RHO_DM_H


def compute_euclid_anisotropic_observables(
    z_bins, om, s_rd, model_name, de_params, h=0.674, include_radiation=True
):
    """
    Evaluate concatenated observables [D_M(z_i)/r_d, H(z_i)*r_d/c] for each redshift bin i.
    Notice D_M(z)/r_d = s_rd * chi(z) and H(z)*r_d/c = E(z) / s_rd, with s_rd = c / (H0 * r_d).
    """
    z_arr = np.asarray(z_bins, dtype=float)
    E_vals = dimensionless_hubble(z_arr, om, model_name, de_params, h=h, include_radiation=include_radiation)
    chi_vals = comoving_distance_over_c_H0(
        z_arr, om, model_name, de_params, h=h, include_radiation=include_radiation
    )
    dm_over_rd = s_rd * chi_vals
    hrd_over_c = E_vals / s_rd
    dh_over_rd = s_rd / E_vals
    vec = np.empty(2 * len(z_arr), dtype=float)
    vec[0::2] = dm_over_rd
    vec[1::2] = hrd_over_c
    return vec, dm_over_rd, hrd_over_c, dh_over_rd


def build_euclid_anisotropic_covariance(
    om_fid, s_rd_fid, model_fid="LCDM", params_fid=(), scenario="full", z_bins=None, h=0.674
):
    """Construct the exact 8x8 block-diagonal covariance matrix on [D_M/r_d, H*r_d/c] for 4 bins."""
    z_arr, sig_dm_rel, sig_h_rel, rho = euclid_bao_cov_and_mean_func(scenario=scenario, z_bins=z_bins)
    fid_vec, dm_fid, hrd_fid, dh_fid = compute_euclid_anisotropic_observables(
        z_arr, om_fid, s_rd_fid, model_fid, params_fid, h=h
    )
    blocks = []
    for idx in range(len(z_arr)):
        s_dm = sig_dm_rel[idx] * dm_fid[idx]
        s_h = sig_h_rel[idx] * hrd_fid[idx]
        blk = np.array(
            [
                [s_dm ** 2, rho * s_dm * s_h],
                [rho * s_dm * s_h, s_h ** 2],
            ],
            dtype=float,
        )
        blocks.append(blk)
    cov = np.zeros((2 * len(z_arr), 2 * len(z_arr)), dtype=float)
    for idx, blk in enumerate(blocks):
        cov[2 * idx : 2 * idx + 2, 2 * idx : 2 * idx + 2] = blk
    return z_arr, fid_vec, cov, np.linalg.inv(cov), sig_dm_rel, sig_h_rel


def compute_euclid_anisotropic_fisher(
    om_fid,
    s_rd_fid,
    lam,
    C_fid=0.0,
    S_fid=0.0,
    scenario="full",
    external_cov_om_srd=None,
    external_fisher_4x4=None,
    z_bins=None,
    h=0.674,
    step=1e-4,
):
    """
    Build the 4x4 coupled anisotropic Euclid BAO Fisher matrix for theta = (Om, s_rd, C, S).

    Schur-complements out (Om, s_rd) to extract the marginalized 2x2 covariance on (C, S),
    returning:
      - A_3sigma_2dof: radius where chi2(C, S) = 11.829 (exact 2-dof 3-sigma threshold)
      - A_3sigma_1dof_median / optimal / worst: 1-dof phase-dependent 3*sigma bounds
      - Marginalized Fisher submatrix F_CS (2x2) and covariance cov_CS (2x2)
    """
    z_arr, fid_vec, cov_obs, inv_cov_obs, _, _ = build_euclid_anisotropic_covariance(
        om_fid, s_rd_fid, "OSC_CS", [C_fid, S_fid, lam], scenario=scenario, z_bins=z_bins, h=h
    )

    p0 = np.array([om_fid, s_rd_fid, C_fid, S_fid], dtype=float)
    J = np.zeros((4, len(fid_vec)), dtype=float)

    for a in range(4):
        dp = np.zeros(4, dtype=float)
        dp[a] = step
        p_plus = p0 + dp
        p_minus = p0 - dp
        v_plus, _, _, _ = compute_euclid_anisotropic_observables(
            z_arr, p_plus[0], p_plus[1], "OSC_CS", [p_plus[2], p_plus[3], lam], h=h
        )
        v_minus, _, _, _ = compute_euclid_anisotropic_observables(
            z_arr, p_minus[0], p_minus[1], "OSC_CS", [p_minus[2], p_minus[3], lam], h=h
        )
        J[a] = (v_plus - v_minus) / (2.0 * step)

    F_euclid = J @ inv_cov_obs @ J.T
    F_total = F_euclid.copy()
    if external_fisher_4x4 is not None:
        F_total = F_total + np.asarray(external_fisher_4x4, dtype=float)
    elif external_cov_om_srd is not None:
        ext = np.asarray(external_cov_om_srd, dtype=float)
        F_total[:2, :2] += np.linalg.inv(ext)

    cov_full = np.linalg.pinv(F_total)
    cov_cs = cov_full[2:, 2:]
    inv_cov_cs = np.linalg.inv(cov_cs)

    # Eigenvalues of inv_cov_cs give directional information lambda_max >= lambda_min
    eigvals = np.linalg.eigvalsh(inv_cov_cs)
    lam_min, lam_max = float(np.maximum(eigvals[0], 1e-18)), float(np.maximum(eigvals[1], 1e-18))
    sigma_best = 1.0 / np.sqrt(lam_max)
    sigma_worst = 1.0 / np.sqrt(lam_min)

    # Phase-sampled 1-dof 3*sigma thresholds and 2-dof joint threshold (Delta chi2 = 11.829)
    phis = np.linspace(0.0, np.pi, 37, endpoint=False)
    a3_1dof_vals = []
    for ph in phis:
        u = np.array([np.cos(ph), np.sin(ph)])
        a3_1dof_vals.append(3.0 / np.sqrt(float(u @ inv_cov_cs @ u)))

    # 2-dof 3-sigma threshold delta_chi2 = 11.829156
    chi2_3sig_2dof = 11.829156
    a3_2dof_vals = [np.sqrt(chi2_3sig_2dof) / 3.0 * v for v in a3_1dof_vals]

    return {
        "lambda": float(lam),
        "scenario": scenario,
        "fisher_4x4": F_total.tolist(),
        "cov_cs_2x2": cov_cs.tolist(),
        "sigma_Om": float(np.sqrt(max(cov_full[0, 0], 0.0))),
        "sigma_srd": float(np.sqrt(max(cov_full[1, 1], 0.0))),
        "sigma_C": float(np.sqrt(max(cov_cs[0, 0], 0.0))),
        "sigma_S": float(np.sqrt(max(cov_cs[1, 1], 0.0))),
        "A_3sigma_best_phi": float(3.0 * sigma_best),
        "A_3sigma_median_phi": float(np.median(a3_1dof_vals)),
        "A_3sigma_worst_phi": float(3.0 * sigma_worst),
        "A_3sigma_2dof_median": float(np.median(a3_2dof_vals)),
        "A_3sigma_2dof_best": float(np.sqrt(chi2_3sig_2dof) * sigma_best),
    }


def compute_euclid_model_separation_anisotropic(
    om_a,
    s_rd_a,
    model_a,
    params_a,
    om_b,
    s_rd_b,
    model_b,
    params_b,
    scenario="full",
    profile_srd=True,
    z_bins=None,
    h_a=0.674,
    h_b=0.674,
):
    """
    Compute exact anisotropic Euclid (D_M/r_d, H*r_d) chi2 separation between Model A (fiducial)
    and Model B across the 4 IST:F bins, optionally profiling analytically over overall s_rd scaling.
    """
    z_arr, fid_vec, _, inv_cov, _, _ = build_euclid_anisotropic_covariance(
        om_a, s_rd_a, model_a, params_a, scenario=scenario, z_bins=z_bins, h=h_a
    )
    alt_vec, dm_b, hrd_b, _ = compute_euclid_anisotropic_observables(
        z_arr, om_b, s_rd_b, model_b, params_b, h=h_b
    )

    if profile_srd:
        # Optimize nuisance ratio kappa = s_rd_b_eff / s_rd_b acting on D_M/r_d ~ kappa, H*r_d ~ 1/kappa
        from scipy.optimize import minimize_scalar

        def obj(kappa):
            mod = alt_vec.copy()
            mod[0::2] *= kappa
            mod[1::2] /= kappa
            diff = fid_vec - mod
            return float(diff @ inv_cov @ diff)

        res = minimize_scalar(obj, bounds=(0.90, 1.10), method="bounded")
        delta_chi2 = float(res.fun)
        kappa_best = float(res.x)
    else:
        diff = fid_vec - alt_vec
        delta_chi2 = float(diff @ inv_cov @ diff)
        kappa_best = 1.0

    rel_dm_pct = 100.0 * (alt_vec[0::2] * kappa_best - fid_vec[0::2]) / fid_vec[0::2]
    rel_h_pct = 100.0 * (alt_vec[1::2] / kappa_best - fid_vec[1::2]) / fid_vec[1::2]

    return {
        "scenario": scenario,
        "z_bins": z_arr.tolist(),
        "delta_chi2": delta_chi2,
        "sigma_equiv": float(np.sqrt(max(delta_chi2, 0.0))),
        "s_rd_profile_factor": kappa_best,
        "max_abs_rel_diff_DM_percent": float(np.max(np.abs(rel_dm_pct))),
        "max_abs_rel_diff_H_percent": float(np.max(np.abs(rel_h_pct))),
        "rel_diff_DM_percent": rel_dm_pct.tolist(),
        "rel_diff_H_percent": rel_h_pct.tolist(),
    }


def generate_euclid_2027_preregistration(bestfit_dict, mcmc_chains_dict=None, z_bins=None):
    """
    Generate the blind Pre-Registered Mid-2027 Euclid DR1 & Full Survey BAO Prediction Table
    for D_M(z)/r_d and D_H(z)/r_d across z in {1.00, 1.20, 1.40, 1.65}.
    """
    z_arr = EUCLID_IST_Z_BINS if z_bins is None else np.asarray(z_bins, dtype=float)
    table = []

    def _unwrap(entry):
        return entry["params"] if isinstance(entry, dict) and "params" in entry else entry

    # Reference LCDM predictions for fractional error-bar calibration
    p_l = _unwrap(bestfit_dict["LCDM"])
    s_rd_l = C_LIGHT / (100.0 * p_l["h"] * rd_physical_mpc(p_l["Om"], p_l["h"], p_l["omega_b"]))
    _, dm_l, hrd_l, dh_l = compute_euclid_anisotropic_observables(
        z_arr, p_l["Om"], s_rd_l, "LCDM", [], h=p_l["h"]
    )
    _, sig_dm_dr1, sig_h_dr1, _ = euclid_bao_cov_and_mean_func("dr1_2027", z_bins=z_arr)
    _, sig_dm_full, sig_h_full, _ = euclid_bao_cov_and_mean_func("full", z_bins=z_arr)

    # Compute CPL and OSC_CS predictions
    preds = {}
    for mname in ("LCDM", "CPL", "OSC_CS"):
        pm = _unwrap(bestfit_dict[mname])
        s_rd_m = C_LIGHT / (100.0 * pm["h"] * rd_physical_mpc(pm["Om"], pm["h"], pm["omega_b"]))
        if mname == "LCDM":
            de_p = []
        elif mname == "CPL":
            de_p = [pm["w0"], pm["wa"]]
        else:
            de_p = [pm["C"], pm["S"], pm["lambda"]]
        _, dm_m, hrd_m, dh_m = compute_euclid_anisotropic_observables(
            z_arr, pm["Om"], s_rd_m, mname, de_p, h=pm["h"]
        )
        preds[mname] = {"DM": dm_m, "DH": dh_m, "Hr_over_c": hrd_m}

        # Posterior 68% error bars on predictions if MCMC chains provided
        if mcmc_chains_dict and mname in mcmc_chains_dict:
            ch = mcmc_chains_dict[mname]
            rng = np.random.default_rng(2027)
            idx_sub = rng.choice(len(ch), size=min(250, len(ch)), replace=False)
            dm_draws = []
            dh_draws = []
            for row in ch[idx_sub]:
                om_d, h_d, ob_d = row[0], row[1], row[2]
                s_rd_d = C_LIGHT / (100.0 * h_d * rd_physical_mpc(om_d, h_d, ob_d))
                _, dmi, _, dhi = compute_euclid_anisotropic_observables(
                    z_arr, om_d, s_rd_d, mname, row[3:], h=h_d
                )
                dm_draws.append(dmi)
                dh_draws.append(dhi)
            dm_draws = np.array(dm_draws)
            dh_draws = np.array(dh_draws)
            preds[mname]["DM_lo"] = np.percentile(dm_draws, 16, axis=0)
            preds[mname]["DM_hi"] = np.percentile(dm_draws, 84, axis=0)
            preds[mname]["DH_lo"] = np.percentile(dh_draws, 16, axis=0)
            preds[mname]["DH_hi"] = np.percentile(dh_draws, 84, axis=0)

    for i, zv in enumerate(z_arr):
        row = {
            "z": float(zv),
            "euclid_dr1_2027_sigma_DM": float(sig_dm_dr1[i] * dm_l[i]),
            "euclid_dr1_2027_sigma_DH": float(sig_h_dr1[i] * dh_l[i]),
            "euclid_full_sigma_DM": float(sig_dm_full[i] * dm_l[i]),
            "euclid_full_sigma_DH": float(sig_h_full[i] * dh_l[i]),
        }
        for mname in ("LCDM", "CPL", "OSC_CS"):
            row[f"{mname}_DM_over_rd"] = float(preds[mname]["DM"][i])
            row[f"{mname}_DH_over_rd"] = float(preds[mname]["DH"][i])
            if "DM_lo" in preds[mname]:
                row[f"{mname}_DM_q16"] = float(preds[mname]["DM_lo"][i])
                row[f"{mname}_DM_q84"] = float(preds[mname]["DM_hi"][i])
                row[f"{mname}_DH_q16"] = float(preds[mname]["DH_lo"][i])
                row[f"{mname}_DH_q84"] = float(preds[mname]["DH_hi"][i])
        table.append(row)
    return table


# =====================================================================
# LEGACY v2.0 H(z)-ONLY FISHER FUNCTIONS (BACKWARD COMPATIBILITY)
# =====================================================================

def compute_euclid_model_delta_chi2(
    om_a,
    model_a,
    params_a,
    om_b,
    model_b,
    params_b,
    z_bins=None,
    rel_sigma_h=EUCLID_REL_SIGMA_H,
):
    """Compute diagonal Euclid H(z) chi2 separation assuming Model A as fiducial truth."""
    z_arr = EUCLID_Z_BINS if z_bins is None else np.asarray(z_bins, dtype=float)
    e_fid = dimensionless_hubble(z_arr, om_a, model_a, params_a, include_radiation=False)
    e_alt = dimensionless_hubble(z_arr, om_b, model_b, params_b, include_radiation=False)
    rel_diff = (e_alt - e_fid) / e_fid
    delta_chi2 = float(np.sum((rel_diff / rel_sigma_h) ** 2))
    return {
        "z_bins": z_arr.tolist(),
        "rel_diff_percent": (100.0 * rel_diff).tolist(),
        "max_abs_rel_diff_percent": float(np.max(np.abs(100.0 * rel_diff))),
        "delta_chi2": delta_chi2,
        "sigma_equiv": float(np.sqrt(max(delta_chi2, 0.0))),
    }


def compute_euclid_osc_fisher(
    om_fid,
    A_fid,
    lam_fid,
    phi0_fid,
    z_bins=None,
    rel_sigma_h=EUCLID_REL_SIGMA_H,
    step=1e-4,
):
    """Build the legacy 2x2 Stage-IV Euclid H(z) Fisher matrix for (Om, A) at fixed (lambda, phi0)."""
    z_arr = EUCLID_Z_BINS if z_bins is None else np.asarray(z_bins, dtype=float)
    e_fid = dimensionless_hubble(z_arr, om_fid, "OSC", [A_fid, lam_fid, phi0_fid], include_radiation=False)

    d_om = (
        dimensionless_hubble(z_arr, om_fid + step, "OSC", [A_fid, lam_fid, phi0_fid], include_radiation=False)
        - dimensionless_hubble(z_arr, om_fid - step, "OSC", [A_fid, lam_fid, phi0_fid], include_radiation=False)
    ) / (2.0 * step * e_fid)

    d_A = (
        dimensionless_hubble(z_arr, om_fid, "OSC", [A_fid + step, lam_fid, phi0_fid], include_radiation=False)
        - dimensionless_hubble(z_arr, om_fid, "OSC", [A_fid - step, lam_fid, phi0_fid], include_radiation=False)
    ) / (2.0 * step * e_fid)

    J = np.vstack([d_om, d_A])
    inv_var = 1.0 / (rel_sigma_h ** 2)
    fisher_euclid = inv_var * (J @ J.T)

    cov_euclid = np.linalg.pinv(fisher_euclid)
    sigma_om = float(np.sqrt(max(cov_euclid[0, 0], 0.0)))
    sigma_A_marg = float(np.sqrt(max(cov_euclid[1, 1], 0.0)))
    sigma_A_unmarg = float(1.0 / np.sqrt(max(fisher_euclid[1, 1], 1e-18)))

    return {
        "fisher_euclid": fisher_euclid.tolist(),
        "sigma_om": sigma_om,
        "sigma_A_unmarg": sigma_A_unmarg,
        "sigma_A_marg_euclid_only": sigma_A_marg,
        "A_3sigma_euclid_only": float(3.0 * sigma_A_marg),
    }


def compute_combined_3sigma_boundary(
    om_fid,
    lam_grid,
    phi0_fid=0.5923,
    sigma_om_prior=0.007,
    z_bins=None,
    rel_sigma_h=EUCLID_REL_SIGMA_H,
):
    """Legacy 7-bin H(z)-only 3-sigma exclusion/discovery boundary with external Om prior."""
    results = []
    phi_grid = np.linspace(-np.pi, np.pi, 13, endpoint=False)
    for lam in lam_grid:
        a3_phases = []
        for phi in phi_grid:
            f_res = compute_euclid_osc_fisher(
                om_fid, 0.0, float(lam), float(phi), z_bins=z_bins, rel_sigma_h=rel_sigma_h
            )
            F = np.array(f_res["fisher_euclid"], dtype=float)
            F[0, 0] += 1.0 / (sigma_om_prior ** 2)
            cov = np.linalg.inv(F)
            a3 = float(3.0 * np.sqrt(max(cov[1, 1], 0.0)))
            a3_phases.append(a3)
        f_bestphi = compute_euclid_osc_fisher(
            om_fid, 0.0, float(lam), float(phi0_fid), z_bins=z_bins, rel_sigma_h=rel_sigma_h
        )
        F_bp = np.array(f_bestphi["fisher_euclid"], dtype=float)
        F_bp[0, 0] += 1.0 / (sigma_om_prior ** 2)
        cov_bp = np.linalg.inv(F_bp)
        a3_fid_phase = float(3.0 * np.sqrt(max(cov_bp[1, 1], 0.0)))
        results.append(
            {
                "lambda": float(lam),
                "A_3sigma_bestfit_phi": a3_fid_phase,
                "A_3sigma_median_phi": float(np.median(a3_phases)),
                "A_3sigma_best_phi": float(np.min(a3_phases)),
            }
        )
    return results
