# -*- coding: utf-8 -*-
"""
Exact Cosmological Likelihoods for Pantheon+ SNe Ia (Full Covariance),
DESI Year 1 BAO (arXiv:2404.03002), and Planck 2018 Compressed CMB Distance Priors
(Chen, Huang & Wang, JCAP 02 (2019) 028, arXiv:1808.05724).

Fixes the three critical v1 methodology errors:
1. Replaces the diagonal-only MU_SH0ES_ERR_DIAG approximation with the full
   1701x1701 Pantheon+SH0ES_STAT+SYS.cov covariance matrix restricted to the
   cosmology sample (zHD > 0.01, IS_CALIBRATOR == 0, N = 1580), analytically
   marginalized over the absolute magnitude / H0 nuisance offset M.
2. Combines Pantheon+ SNe Ia with DESI 2024 VI BAO (7 redshift bins, 12 data points
   across 0.295 <= z <= 2.330) and Planck 2018 compressed CMB shift/acoustic priors
   (R, l_A, omega_b) with Hu-Sugiyama decoupling redshift z_*.
3. Evaluates models across broad, non-restrictive physical parameter domains so
   that best-fit solutions and MCMC posteriors are data-driven rather than
   pinned against artificial optimizer box boundaries.
"""

import os
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve, block_diag

C_LIGHT = 299792.458  # km/s
T_CMB = 2.7255        # K
N_EFF = 3.046
# Omega_r * h^2 from T_CMB = 2.7255 K and N_eff = 3.046:
# omega_gamma = 2.469e-5 * (T_CMB/2.7255)^4 = 2.4728e-5
# omega_r = omega_gamma * (1 + 0.22710731766 * N_eff) = 4.18343e-5
OMEGA_GAMMA_H2 = 2.47282e-5
OMEGA_R_H2 = OMEGA_GAMMA_H2 * (1.0 + 0.22710731766 * N_EFF)

PANTHEON_DIR = r"C:\Users\bruno\OneDrive\Desktop\_Archive_Ouroboros_Antigo\Pantheon_Data\Pantheon+_Data\4_DISTANCES_AND_COVAR"


# =====================================================================
# 1. DARK ENERGY EQUATION OF STATE & EXACT CONTINUITY SOLUTIONS
# =====================================================================

def w_model(z, model_name, de_params):
    z = np.asarray(z, dtype=float)
    a = 1.0 / (1.0 + z)
    if model_name == 'LCDM':
        return -np.ones_like(z)
    elif model_name == 'CPL':
        w0, wa = de_params
        return w0 + wa * (1.0 - a)
    elif model_name == 'OSC':
        A, lam, phi0 = de_params
        return -1.0 + A * np.sin((2.0 * np.pi / lam) * np.log(a) + phi0)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def rho_de_ratio(z, model_name, de_params):
    """Exact analytical solution to d(ln rho_DE)/d(ln a) = -3(1 + w(a))."""
    z = np.asarray(z, dtype=float)
    if model_name == 'LCDM':
        return np.ones_like(z)
    elif model_name == 'CPL':
        w0, wa = de_params
        return (1.0 + z)**(3.0 * (1.0 + w0 + wa)) * np.exp(-3.0 * wa * z / (1.0 + z))
    elif model_name == 'OSC':
        A, lam, phi0 = de_params
        k = 2.0 * np.pi / lam
        arg = -k * np.log(1.0 + z) + phi0
        integral_val = (3.0 * A / k) * (np.cos(phi0) - np.cos(arg))
        return np.exp(-integral_val)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def hu_sugiyama_z_star(omega_b, omega_m):
    """Hu & Sugiyama (1996, ApJ 471, 542) fitting formula for photon decoupling redshift z_*."""
    g1 = (0.0783 * omega_b**(-0.238)) / (1.0 + 39.5 * omega_b**(0.763))
    g2 = 0.560 / (1.0 + 21.1 * omega_b**(1.81))
    return 1048.0 * (1.0 + 0.00124 * omega_b**(-0.738)) * (1.0 + g1 * omega_m**g2)


# =====================================================================
# 2. COMBINED LIKELIHOOD ENGINE (PANTHEON+ + DESI BAO + PLANCK CMB)
# =====================================================================

class CosmologyLikelihoodEngine:
    """
    Fast, exact likelihood evaluator using rank-1 marginalized Cholesky precision
    matrix for Pantheon+ SNe Ia, block-diagonal covariance for DESI 2024 VI BAO,
    and the 3x3 covariance matrix for Planck 2018 compressed CMB distance priors.
    """

    def __init__(self, pantheon_dir=PANTHEON_DIR):
        dat_path = os.path.join(pantheon_dir, "Pantheon+SH0ES.dat")
        cov_path = os.path.join(pantheon_dir, "Pantheon+SH0ES_STAT+SYS.cov")

        df = pd.read_csv(dat_path, sep=r'\s+')
        with open(cov_path, 'r') as f:
            n_tot = int(f.readline().strip())
            cov_full = np.fromfile(f, dtype=float, count=n_tot * n_tot, sep=' ').reshape(n_tot, n_tot)

        # Standard uncalibrated Pantheon+ Hubble-flow sample (Brout et al. 2022)
        mask = (df['zHD'] > 0.01) & (df['IS_CALIBRATOR'] == 0)
        self.idx_sn = np.where(mask)[0]
        self.n_sn = len(self.idx_sn)
        self.zhd = df['zHD'].values[self.idx_sn].astype(np.float64)
        self.zhel = df['zHEL'].values[self.idx_sn].astype(np.float64)
        self.mb = df['m_b_corr'].values[self.idx_sn].astype(np.float64)

        self.cov_sn = cov_full[np.ix_(self.idx_sn, self.idx_sn)]
        self.cho_sn = cho_factor(self.cov_sn, lower=True, check_finite=False)
        Cinv_sn = cho_solve(self.cho_sn, np.eye(self.n_sn, dtype=np.float64), check_finite=False)
        ones = np.ones(self.n_sn, dtype=np.float64)
        Cinv_1 = Cinv_sn @ ones
        self.e_sn = float(ones @ Cinv_1)
        # Rank-1 projected precision matrix for analytical marginalization over M:
        self.Cinv_sn_proj = np.ascontiguousarray(Cinv_sn - np.outer(Cinv_1, Cinv_1) / self.e_sn)

        # DESI 2024 VI (arXiv:2404.03002, JCAP 02 (2025) 021, Table 1)
        self.desi_bao_spec = [
            (0.295, 'DV',    7.93, 0.15,  0.0,  0.0,   0.0),
            (0.510, 'DM_DH', 13.62, 0.25, 20.98, 0.61, -0.445),
            (0.706, 'DM_DH', 16.85, 0.32, 20.08, 0.60, -0.420),
            (0.930, 'DM_DH', 21.71, 0.28, 17.88, 0.35, -0.389),
            (1.317, 'DM_DH', 27.79, 0.69, 13.82, 0.42, -0.444),
            (1.491, 'DV',    26.07, 0.67,  0.0,  0.0,   0.0),
            (2.330, 'DM_DH', 39.71, 0.94,  8.52, 0.17, -0.477),
        ]
        y_b = []
        blocks = []
        for z_b, kind, v1, s1, v2, s2, r_c in self.desi_bao_spec:
            if kind == 'DV':
                y_b.append(v1)
                blocks.append(np.array([[s1**2]]))
            else:
                y_b.extend([v1, v2])
                blocks.append(np.array([[s1**2, r_c * s1 * s2], [r_c * s1 * s2, s2**2]]))
        self.y_bao = np.array(y_b, dtype=np.float64)
        self.n_bao = len(self.y_bao)  # 12 measurements
        self.C_bao = block_diag(*blocks)
        self.Cinv_bao = np.linalg.inv(self.C_bao)

        # Planck 2018 compressed CMB distance priors (Chen, Huang & Wang 2019, JCAP 02, 028, Table 1)
        # X = (R, l_A, omega_b) = (1.7502, 301.471, 0.02236)
        self.y_cmb = np.array([1.7502, 301.471, 0.02236], dtype=np.float64)
        sig_cmb = np.array([0.0046, 0.090, 0.00015], dtype=np.float64)
        corr_cmb = np.array([
            [1.0,   0.46, -0.66],
            [0.46,  1.0,  -0.33],
            [-0.66, -0.33, 1.0],
        ], dtype=np.float64)
        self.C_cmb = np.outer(sig_cmb, sig_cmb) * corr_cmb
        self.Cinv_cmb = np.linalg.inv(self.C_cmb)

        # Precompute integration grids:
        # 1. Low-z grid [0, 2.5] for SNe and BAO
        self.z_low = np.linspace(0.0, 2.5, 800)
        self.dz_low = np.diff(self.z_low)
        # 2. High-z grid [2.5, 1150] for CMB comoving distance D_M(z_*)
        self.z_high = np.geomspace(2.5, 1150.0, 600)
        self.dz_high = np.diff(self.z_high)
        # 3. Sound horizon integration grid [z_*, 1e6] in log(a)
        self.u_rs = np.linspace(0.0, 1.0, 400)

    def compute_chi2_components(self, Om, model_name, de_params, h=0.678, omega_b=0.02236,
                                use_bao=True, use_cmb=True):
        """
        Computes (chi2_sn, chi2_bao, chi2_cmb) for given cosmological parameters.
        Note:
          - For SN-only or SN+BAO without CMB, we profile analytically over M (for SN)
            and over s_rd = c / (H0 * r_d) (for BAO), so h and omega_b do not affect
            chi2_sn or chi2_bao (except negligible ~1e-5 radiation at low z).
          - When use_cmb=True, both R = sqrt(Om) H0 D_M(z_*)/c and l_A = pi D_M(z_*)/r_s(z_*)
            are evaluated consistently, and BAO uses the exact drag-epoch sound horizon
            scaling or profiles s_rd consistently.
        """
        if Om <= 0.05 or Om >= 0.70:
            return 1e12, 1e12, 1e12
        Or = OMEGA_R_H2 / (h * h)
        Ode = 1.0 - Om - Or
        if Ode <= 0.05:
            return 1e12, 1e12, 1e12

        de_low = rho_de_ratio(self.z_low, model_name, de_params)
        E2_low = Or * (1.0 + self.z_low)**4 + Om * (1.0 + self.z_low)**3 + Ode * de_low
        if np.any(E2_low <= 1e-8) or not np.all(np.isfinite(E2_low)):
            return 1e12, 1e12, 1e12
        E_low = np.sqrt(E2_low)
        invE_low = 1.0 / E_low
        cum_I_low = np.concatenate([[0.0], np.cumsum(0.5 * (invE_low[:-1] + invE_low[1:]) * self.dz_low)])

        # 1. Pantheon+ SNe Ia chi2 (analytically marginalized over M)
        chi_sn = np.interp(self.zhd, self.z_low, cum_I_low)
        dl_sn = (1.0 + self.zhel) * (C_LIGHT / 70.0) * chi_sn
        mu0 = 5.0 * np.log10(dl_sn) + 25.0
        r_sn = self.mb - mu0
        chi2_sn = float(r_sn @ (self.Cinv_sn_proj @ r_sn))

        # 2. DESI 2024 VI BAO chi2 (analytically profiled over s_rd = c / (H0 * r_d) when CMB is off,
        #    or using exact physical r_d when CMB is on or profiled — note DESI 2024 VI treats r_d*h as free
        #    for BAO+SN, and uses physical sound horizon when combined with CMB).
        chi2_bao = 0.0
        if use_bao:
            m0 = np.empty(self.n_bao, dtype=np.float64)
            k_idx = 0
            for z_b, kind, *_ in self.desi_bao_spec:
                I_b = float(np.interp(z_b, self.z_low, cum_I_low))
                E_b = float(np.interp(z_b, self.z_low, E_low))
                dm_0 = I_b
                dh_0 = 1.0 / E_b
                if kind == 'DV':
                    m0[k_idx] = (z_b * dm_0 * dm_0 * dh_0)**(1.0 / 3.0)
                    k_idx += 1
                else:
                    m0[k_idx] = dm_0
                    m0[k_idx + 1] = dh_0
                    k_idx += 2

            if not use_cmb:
                # Profile analytically over s_rd = c / (H0 * r_d)
                A_b = float(m0 @ (self.Cinv_bao @ m0))
                B_b = float(m0 @ (self.Cinv_bao @ self.y_bao))
                s_best = B_b / A_b
                res_b = self.y_bao - s_best * m0
                chi2_bao = float(res_b @ (self.Cinv_bao @ res_b))
            else:
                # Compute physical r_d (in Mpc) using standard BBN/Planck calibrated sound horizon
                # r_d = 147.09 * (omega_m / 0.1430)^(-0.255) * (omega_b / 0.02236)^(-0.128) Mpc
                omega_m = Om * h * h
                rd_mpc = 147.09 * (omega_m / 0.1430)**(-0.255) * (omega_b / 0.02236)**(-0.128)
                s_phys = C_LIGHT / (100.0 * h * rd_mpc)
                res_b = self.y_bao - s_phys * m0
                chi2_bao = float(res_b @ (self.Cinv_bao @ res_b))

        # 3. Planck 2018 compressed CMB distance priors (R, l_A, omega_b)
        chi2_cmb = 0.0
        if use_cmb:
            omega_m = Om * h * h
            z_star = hu_sugiyama_z_star(omega_b, omega_m)
            de_high = rho_de_ratio(self.z_high, model_name, de_params)
            E2_high = Or * (1.0 + self.z_high)**4 + Om * (1.0 + self.z_high)**3 + Ode * de_high
            if np.any(E2_high <= 1e-8) or not np.all(np.isfinite(E2_high)):
                return 1e12, 1e12, 1e12
            invE_high = 1.0 / np.sqrt(E2_high)
            cum_I_high = cum_I_low[-1] + np.concatenate([
                [0.0], np.cumsum(0.5 * (invE_high[:-1] + invE_high[1:]) * self.dz_high)
            ])
            I_star = float(np.interp(z_star, self.z_high, cum_I_high))
            R_th = np.sqrt(Om) * I_star

            # Sound horizon at decoupling r_s(z_*) via exact integral from a=0 to a_* = 1/(1+z_*):
            a_star = 1.0 / (1.0 + z_star)
            a_arr = a_star * self.u_rs[1:]  # avoid a=0 division
            z_rs = (1.0 / a_arr) - 1.0
            de_rs = rho_de_ratio(z_rs, model_name, de_params)
            E2_rs = Or * (1.0 + z_rs)**4 + Om * (1.0 + z_rs)**3 + Ode * de_rs
            if np.any(E2_rs <= 1e-8):
                return 1e12, 1e12, 1e12
            # Baryon-to-photon momentum density ratio R_b(a) = 3 rho_b / (4 rho_gamma) = 30313 * omega_b * a
            R_b = (3.0 * omega_b / (4.0 * OMEGA_GAMMA_H2)) * a_arr
            c_s = 1.0 / np.sqrt(3.0 * (1.0 + R_b))
            integrand_rs = c_s / (a_arr * a_arr * np.sqrt(E2_rs))
            # Include a=0 limit (where integrand_rs -> 1 / (sqrt(3) * sqrt(Or)))
            int0 = 1.0 / np.sqrt(3.0 * Or)
            integrand_full = np.concatenate([[int0], integrand_rs])
            da = a_star * (self.u_rs[1] - self.u_rs[0])
            I_rs = float(np.sum(0.5 * (integrand_full[:-1] + integrand_full[1:]) * da))
            l_A_th = np.pi * I_star / I_rs

            delta_cmb = np.array([R_th - self.y_cmb[0], l_A_th - self.y_cmb[1], omega_b - self.y_cmb[2]])
            chi2_cmb = float(delta_cmb @ (self.Cinv_cmb @ delta_cmb))

        return chi2_sn, chi2_bao, chi2_cmb

