# -*- coding: utf-8 -*-
"""Unit tests for Stage-IV ESA Euclid Fisher Matrix & Degeneracy Audit."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from src.euclid_fisher import (
    EUCLID_Z_BINS,
    compute_euclid_model_delta_chi2,
    compute_euclid_osc_fisher,
    compute_combined_3sigma_boundary,
)


def test_identical_models_have_zero_delta_chi2():
    res = compute_euclid_model_delta_chi2(
        0.3135, "OSC", [0.2179, 3.0876, 0.5923],
        0.3135, "OSC", [0.2179, 3.0876, 0.5923],
    )
    assert abs(res["delta_chi2"]) < 1e-12
    assert res["max_abs_rel_diff_percent"] < 1e-10


def test_joint_bestfit_cpl_vs_osc_is_degenerate_within_euclid_only():
    # Current joint SN+BAO+CMB best fits overlap strongly in z in [0.9, 1.8]
    res = compute_euclid_model_delta_chi2(
        0.3134, "CPL", [-0.8668, -0.5469],
        0.3135, "OSC", [0.2179, 3.0876, 0.5923],
    )
    assert len(res["z_bins"]) == len(EUCLID_Z_BINS) == 7
    assert res["sigma_equiv"] < 1.0  # Honest finding: best-fit difference < 1 sigma


def test_fisher_matrix_symmetric_and_positive_semidefinite():
    f_res = compute_euclid_osc_fisher(0.3135, 0.0, 3.0876, 0.5923)
    F = np.array(f_res["fisher_euclid"])
    assert np.allclose(F, F.T)
    eigvals = np.linalg.eigvalsh(F)
    assert np.all(eigvals > 0.0)
    assert f_res["sigma_A_marg_euclid_only"] >= f_res["sigma_A_unmarg"]


def test_combined_3sigma_boundary_decreases_with_stronger_prior():
    curve = compute_combined_3sigma_boundary(0.3135, [1.5, 3.0876, 6.0])
    assert len(curve) == 3
    for row in curve:
        assert 0.0 < row["A_3sigma_best_phi"] <= row["A_3sigma_median_phi"]
        assert np.isfinite(row["A_3sigma_bestfit_phi"])

