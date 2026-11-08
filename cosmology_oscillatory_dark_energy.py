# -*- coding: utf-8 -*-
"""
Logarithmic Oscillatory Dark Energy vs. LCDM and CPL (v2.0 Entrypoint).

Delegates to `run_full_analysis.py` (Full Pantheon+ 1580x1580 covariance matrix
+ DESI 2024 VI BAO + Planck 2018 CMB shift parameter + emcee MCMC), while preserving
analytical helper functions for backwards compatibility.

Author: Bruno Luçardo Olivaes
"""

import numpy as np
from src.likelihoods import w_model, rho_de_ratio
from run_full_analysis import run_cosmology_v2_audit


def E_LCDM(z, Om):
    return np.sqrt(Om * (1.0 + np.asarray(z))**3 + (1.0 - Om))


def E_CPL(z, Om, w0, wa):
    z = np.asarray(z)
    return np.sqrt(Om * (1.0 + z)**3 + (1.0 - Om) * rho_de_ratio(z, 'CPL', [w0, wa]))


def E_Oscillatory(z, Om, A_bulk, lambda_bulk, phi0):
    z = np.asarray(z)
    return np.sqrt(Om * (1.0 + z)**3 + (1.0 - Om) * rho_de_ratio(z, 'OSC', [A_bulk, lambda_bulk, phi0]))


def w_CPL(z, w0, wa):
    return w_model(z, 'CPL', [w0, wa])


def w_Oscillatory(z, A_bulk, lambda_bulk, phi0):
    return w_model(z, 'OSC', [A_bulk, lambda_bulk, phi0])


if __name__ == '__main__':
    run_cosmology_v2_audit()
