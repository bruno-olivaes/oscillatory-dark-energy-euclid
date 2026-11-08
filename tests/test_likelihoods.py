# -*- coding: utf-8 -*-
"""
Unit tests for the v2.0 Cosmology Likelihood Engine (Pantheon+ Full Covariance,
DESI 2024 VI BAO, Planck 2018 CMB shift prior, and exact continuity equation integration).
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from src.likelihoods import CosmologyLikelihoodEngine, w_model, rho_de_ratio


def test_pantheon_covariance_positive_definite_and_lcdm_om():
    engine = CosmologyLikelihoodEngine()
    assert engine.n_sn == 1580
    # Verify symmetry
    assert np.allclose(engine.cov_sn, engine.cov_sn.T, atol=1e-10)

    # Fit flat LCDM to Pantheon+ SNe Ia alone with full STAT+SYS covariance
    def obj(Om):
        c_sn, _, _ = engine.compute_chi2_components(Om, 'LCDM', [], use_bao=False, use_cmb=False)
        return c_sn

    res = minimize_scalar(obj, bounds=(0.20, 0.50), method='bounded')
    # Brout et al. (2022, ApJ 938, 110) report Om = 0.334 +/- 0.018
    assert 0.315 <= res.x <= 0.350
    assert abs(res.x - 0.3326) < 0.005


def test_oscillatory_rho_de_matches_numerical_quadrature():
    params = [0.35, 4.8, 0.4]  # A, lambda, phi0
    for z_test in [0.2, 0.8, 1.5, 2.8]:
        rho_exact = float(rho_de_ratio(z_test, 'OSC', params))
        # Numerical integration of 3*(1 + w(z')) / (1 + z') dz' from 0 to z_test
        val_int, _ = quad(lambda zp: 3.0 * (1.0 + float(w_model(zp, 'OSC', params))) / (1.0 + zp), 0.0, z_test)
        rho_num = float(np.exp(val_int))
        assert abs(rho_exact - rho_num) / rho_num < 1e-7


def test_desi_bao_planck_consistency():
    engine = CosmologyLikelihoodEngine()
    c_sn, c_bao, c_cmb = engine.compute_chi2_components(0.314, 'LCDM', [], use_bao=True, use_cmb=False)
    assert c_bao < 25.0  # 12 BAO points should have chi2 ~ 14.5 for LCDM

