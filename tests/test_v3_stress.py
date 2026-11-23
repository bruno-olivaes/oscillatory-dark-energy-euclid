# -*- coding: utf-8 -*-
"""
Unit and Integration Tests for Paper 3 v3.0 Upgrade:
  - DES-SN5YR Dovekie (1,820 SNe) & Pantheon+ (1,580 SNe) via src.data_loader
  - DESI DR2 (13 observables, z=2.330 DH/DM row ordering)
  - Full 3D Planck 2018 compressed CMB prior (R, l_A, omega_b) & coupled r_d
  - Orthogonal Cartesian OSC_CS equivalence and quadrature accuracy
  - Coupled Anisotropic (D_M/r_d, H*r_d) Euclid DR1-2027 vs Full-Survey Fisher & Pre-Registration
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar, minimize

from src.data_loader import load_pantheon_plus, load_des_sn5yr, load_desi_bao
from src.likelihoods import (
    CosmologyLikelihoodEngine,
    w_model,
    rho_de_ratio,
    polar_to_cs,
    cs_to_polar,
    rd_physical_mpc,
)
from src.euclid_fisher import (
    EUCLID_IST_Z_BINS,
    DR1_AREA_SCALE,
    compute_euclid_anisotropic_fisher,
    compute_euclid_model_separation_anisotropic,
    compute_combined_3sigma_boundary,
    generate_euclid_2027_preregistration,
)


def test_des_sn5yr_dovekie_loading_and_lcdm_fit():
    des = load_des_sn5yr()
    assert des["n_sn"] == 1820
    assert np.allclose(des["inv_cov"], des["inv_cov"].T, atol=1e-7)
    eig_min = float(np.min(np.linalg.eigvalsh(des["inv_cov"])))
    assert eig_min > 0.0

    eng = CosmologyLikelihoodEngine(sn_dataset="des_sn5yr", bao_dataset="desi_dr2")
    res = minimize_scalar(
        lambda om: eng.compute_chi2_components(om, "LCDM", [], use_bao=False, use_cmb=False)[0],
        bounds=(0.20, 0.50),
        method="bounded",
    )
    # Recalibrated DES-Dovekie recovers Om ~ 0.330 with reduced chi2 ~ 0.90
    assert 0.30 <= res.x <= 0.36
    assert abs(res.x - 0.330) < 0.015


def test_desi_dr2_schema_ordering_and_covariance():
    specs, y_bao, C_bao, Cinv_bao = load_desi_bao("dr2")
    assert len(specs) == 13
    assert y_bao.shape == (13,)
    assert C_bao.shape == (13, 13)
    assert np.allclose(C_bao, C_bao.T, atol=1e-10)
    assert np.all(np.linalg.eigvalsh(C_bao) > 0.0)
    # Critical check: Cobaya DR2 places DH_over_rs before DM_over_rs at z=2.33
    assert abs(specs[11][0] - 2.33) < 1e-4 and specs[11][1] == "DH_over_rs"
    assert abs(specs[12][0] - 2.33) < 1e-4 and specs[12][1] == "DM_over_rs"


def test_planck_3d_cmb_and_rd_calibration():
    eng = CosmologyLikelihoodEngine(sn_dataset="pantheon_plus", bao_dataset="desi_dr2")
    # Optimize Planck 3D prior alone to confirm internal consistency with Planck 2018
    res = minimize(
        lambda p: eng.compute_chi2_components(
            p[0], "LCDM", [], h=p[1], omega_b=p[2], use_sn=False, use_bao=False, use_cmb=True
        )[2],
        [0.315, 0.674, 0.02236],
        method="Nelder-Mead",
    )
    om_b, h_b, ob_b = res.x
    assert res.fun < 0.05
    assert 0.310 <= om_b <= 0.322
    assert 0.668 <= h_b <= 0.680
    rd_val = rd_physical_mpc(om_b, h_b, ob_b)
    assert 146.5 <= rd_val <= 147.8


def test_osc_cs_exact_equivalence_and_quadrature():
    A, lam, phi0 = 0.28, 3.4, 0.85
    C, S = polar_to_cs(A, phi0)
    A_back, phi_back = cs_to_polar(C, S)
    assert abs(A_back - A) < 1e-12
    assert abs(phi_back - phi0) < 1e-12

    z_arr = np.linspace(0.05, 3.0, 25)
    assert np.max(np.abs(w_model(z_arr, "OSC", [A, lam, phi0]) - w_model(z_arr, "OSC_CS", [C, S, lam]))) < 1e-12
    assert np.max(np.abs(rho_de_ratio(z_arr, "OSC", [A, lam, phi0]) - rho_de_ratio(z_arr, "OSC_CS", [C, S, lam]))) < 1e-12

    for z_t in (0.4, 1.4, 2.6):
        rho_exact = float(rho_de_ratio(z_t, "OSC_CS", [C, S, lam]))
        val_int, _ = quad(lambda zp: 3.0 * (1.0 + float(w_model(zp, "OSC_CS", [C, S, lam]))) / (1.0 + zp), 0.0, z_t)
        rho_num = float(np.exp(val_int))
        assert abs(rho_exact - rho_num) / rho_num < 1e-7


def test_coupled_euclid_fisher_dr1_vs_full_and_prereg():
    cov_ext = np.diag([0.007 ** 2, 0.08 ** 2])
    f_dr1 = compute_euclid_anisotropic_fisher(
        0.31, 30.3, lam=3.1, scenario="dr1_2027", external_cov_om_srd=cov_ext
    )
    f_full = compute_euclid_anisotropic_fisher(
        0.31, 30.3, lam=3.1, scenario="full", external_cov_om_srd=cov_ext
    )
    F_full = np.array(f_full["fisher_4x4"])
    assert np.allclose(F_full, F_full.T, atol=1e-9)
    assert np.all(np.linalg.eigvalsh(F_full) > 0.0)

    # Full survey (14,000 deg^2) must strictly outperform DR1-2027 (1,900 deg^2)
    assert DR1_AREA_SCALE > 2.5
    assert f_full["A_3sigma_median_phi"] < f_dr1["A_3sigma_median_phi"]

    # Adding D_M/r_d beats legacy H(z)-only Fisher at lambda=3.1 with matching Om prior
    f_leg = compute_combined_3sigma_boundary(0.31, [3.1], sigma_om_prior=0.007)[0]
    assert f_full["A_3sigma_median_phi"] < f_leg["A_3sigma_median_phi"]

    sep_full = compute_euclid_model_separation_anisotropic(
        0.31, 30.3, "LCDM", [], 0.31, 30.3, "OSC_CS", [0.25, 0.15, 3.1], scenario="full"
    )
    sep_dr1 = compute_euclid_model_separation_anisotropic(
        0.31, 30.3, "LCDM", [], 0.31, 30.3, "OSC_CS", [0.25, 0.15, 3.1], scenario="dr1_2027"
    )
    assert sep_full["delta_chi2"] > sep_dr1["delta_chi2"] > 0.0

    bf_mock = {
        "LCDM": {"Om": 0.31, "h": 0.675, "omega_b": 0.02236},
        "CPL": {"Om": 0.31, "h": 0.675, "omega_b": 0.02236, "w0": -0.85, "wa": -0.60},
        "OSC_CS": {"Om": 0.31, "h": 0.675, "omega_b": 0.02236, "C": 0.18, "S": 0.12, "lambda": 3.1},
    }
    prereg = generate_euclid_2027_preregistration(bf_mock)
    assert len(prereg) == len(EUCLID_IST_Z_BINS) == 4
    for row in prereg:
        assert row["euclid_dr1_2027_sigma_DM"] > row["euclid_full_sigma_DM"] > 0.0
        assert row["euclid_dr1_2027_sigma_DH"] > row["euclid_full_sigma_DH"] > 0.0
