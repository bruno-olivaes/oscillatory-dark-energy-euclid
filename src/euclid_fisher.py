# -*- coding: utf-8 -*-
"""Stage-IV ESA Euclid H(z) Fisher Matrix & Degeneracy-Breaking Forecast."""

import numpy as np
from src.likelihoods import rho_de_ratio

# Seven spectroscopic H(z) bins spanning Euclid's primary redshift window [0.9, 1.8]
# with ~1.3% fractional uncertainty per bin (Blanchard et al., A&A 642, A191, 2020).
EUCLID_Z_BINS = np.array([0.90, 1.05, 1.20, 1.35, 1.50, 1.65, 1.80], dtype=float)
EUCLID_REL_SIGMA_H = 0.013


def dimensionless_hubble(z, om, model_name, de_params):
    """Return dimensionless expansion rate E(z) = H(z)/H0 for flat FLRW models."""
    z_arr = np.asarray(z, dtype=float)
    rho_de = rho_de_ratio(z_arr, model_name, de_params)
    return np.sqrt(om * (1.0 + z_arr) ** 3 + (1.0 - om) * rho_de)


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
    """Compute diagonal Euclid chi2 separation assuming Model A as fiducial truth."""
    z_arr = EUCLID_Z_BINS if z_bins is None else np.asarray(z_bins, dtype=float)
    e_fid = dimensionless_hubble(z_arr, om_a, model_a, params_a)
    e_alt = dimensionless_hubble(z_arr, om_b, model_b, params_b)
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
    """
    Build the 2x2 Stage-IV Euclid H(z) Fisher matrix for (Om, A) at fixed (lambda, phi0).

    Marginalizing over Om via Schur complement yields sigma_A_marg and the 3-sigma
    detection/exclusion amplitude A_3sigma = 3 * sigma_A_marg. Notice that evaluating at
    A_fid = 0 reduces to the exact Fisher information against flat LCDM because H(z; A)
    depends linearly on A to first order around A = 0.
    """
    z_arr = EUCLID_Z_BINS if z_bins is None else np.asarray(z_bins, dtype=float)
    e_fid = dimensionless_hubble(z_arr, om_fid, "OSC", [A_fid, lam_fid, phi0_fid])

    # Central finite-difference derivatives d(ln E)/dtheta_alpha
    d_om = (
        dimensionless_hubble(z_arr, om_fid + step, "OSC", [A_fid, lam_fid, phi0_fid])
        - dimensionless_hubble(z_arr, om_fid - step, "OSC", [A_fid, lam_fid, phi0_fid])
    ) / (2.0 * step * e_fid)

    d_A = (
        dimensionless_hubble(z_arr, om_fid, "OSC", [A_fid + step, lam_fid, phi0_fid])
        - dimensionless_hubble(z_arr, om_fid, "OSC", [A_fid - step, lam_fid, phi0_fid])
    ) / (2.0 * step * e_fid)

    J = np.vstack([d_om, d_A])  # shape (2, N_bins)
    inv_var = 1.0 / (rel_sigma_h ** 2)
    fisher_euclid = inv_var * (J @ J.T)

    # Regularize lightly only if Om-A degenerates near singular nodes
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
    """
    Compute the Stage-IV 3-sigma exclusion/discovery boundary A_3sigma(lambda)
    combining Euclid's 7 spectroscopic H(z) bins with a conservative external Om prior
    (sigma_Om ~ 0.007 from Pantheon+ + BAO + Planck) by adding 1/sigma_Om^2 to F_00.
    Also optimizes over phi0_grid to report the phase-averaged and worst-case thresholds.
    """
    results = []
    phi_grid = np.linspace(-np.pi, np.pi, 13, endpoint=False)
    for lam in lam_grid:
        a3_phases = []
        a3_fid_phase = None
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
        results.append({
            "lambda": float(lam),
            "A_3sigma_bestfit_phi": a3_fid_phase,
            "A_3sigma_median_phi": float(np.median(a3_phases)),
            "A_3sigma_best_phi": float(np.min(a3_phases)),
        })
    return results

