# -*- coding: utf-8 -*-
"""
Unified Cosmological Likelihood Engine (v3.0).

Integrates:
  1. Full-covariance Type Ia Supernovae (Pantheon+SH0ES N=1,580 or DES-SN5YR Dovekie N=1,820),
     analytically marginalized over the absolute magnitude / H0 nuisance offset M.
  2. DESI BAO measurements (Year 1 arXiv:2404.03002 with 12 observables, or DR2 arXiv:2503.14738
     with 13 observables), with exact dynamic observable ordering and physical sound horizon r_d.
  3. Planck 2018 compressed 3D CMB distance priors (R, l_A, omega_b) following Chen, Huang & Wang
     (JCAP 02 (2019) 028, arXiv:1808.05724), plus optional 1D shift-parameter mode for v2.0 auditing.
  4. Non-singular orthogonal Cartesian parameterization 'OSC_CS' (C, S, lambda) alongside polar
     'OSC' (A, lambda, phi0), CPL (w0, wa), and flat LCDM.
"""

import numpy as np
from src.data_loader import (
    resolve_pantheon_dir,
    load_pantheon_plus,
    load_des_sn5yr,
    load_desi_bao,
)

C_LIGHT = 299792.458  # km/s
T_CMB = 2.7255        # K
N_EFF = 3.046
OMEGA_GAMMA_H2 = 2.47282e-5
OMEGA_R_H2 = OMEGA_GAMMA_H2 * (1.0 + 0.22710731766 * N_EFF)

PANTHEON_DIR = resolve_pantheon_dir()


# =====================================================================
# 1. DARK ENERGY EQUATIONS OF STATE & EXACT CONTINUITY SOLUTIONS
# =====================================================================

def polar_to_cs(A, phi0):
    """Convert polar (A, phi0) to orthogonal Cartesian (C, S)."""
    return float(A * np.cos(phi0)), float(A * np.sin(phi0))


def cs_to_polar(C, S):
    """Convert orthogonal Cartesian (C, S) to polar (A, phi0) in [0, inf) x [-pi, pi]."""
    A = float(np.hypot(C, S))
    phi0 = float(np.arctan2(S, C))
    return A, phi0


def w_model(z, model_name, de_params):
    """Evaluate dark energy equation of state w(z) across supported models."""
    z = np.asarray(z, dtype=float)
    a = 1.0 / (1.0 + z)
    if model_name == "LCDM":
        return -np.ones_like(z)
    elif model_name == "CPL":
        w0, wa = de_params
        return w0 + wa * (1.0 - a)
    elif model_name == "OSC":
        A, lam, phi0 = de_params
        return -1.0 + A * np.sin((2.0 * np.pi / lam) * np.log(a) + phi0)
    elif model_name == "OSC_CS":
        C, S, lam = de_params
        ln_a = np.log(a)
        k = 2.0 * np.pi / lam
        return -1.0 + C * np.sin(k * ln_a) + S * np.cos(k * ln_a)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def rho_de_ratio(z, model_name, de_params):
    """Exact analytical solution to d(ln rho_DE)/d(ln a) = -3(1 + w(a))."""
    z = np.asarray(z, dtype=float)
    ln1pz = np.log1p(z)
    if model_name == "LCDM":
        return np.ones_like(z)
    elif model_name == "CPL":
        w0, wa = de_params
        return (1.0 + z) ** (3.0 * (1.0 + w0 + wa)) * np.exp(-3.0 * wa * z / (1.0 + z))
    elif model_name == "OSC":
        A, lam, phi0 = de_params
        k = 2.0 * np.pi / lam
        arg = -k * ln1pz + phi0
        integral_val = (3.0 * A / k) * (np.cos(phi0) - np.cos(arg))
        return np.exp(-integral_val)
    elif model_name == "OSC_CS":
        C, S, lam = de_params
        k = 2.0 * np.pi / lam
        phase = k * ln1pz
        # int_{-ln(1+z)}^{0} [-3*(C*sin(k u) + S*cos(k u))] du
        log_rho = (3.0 / k) * (C * (np.cos(phase) - 1.0) + S * np.sin(phase))
        return np.exp(np.clip(log_rho, -80.0, 80.0))
    else:
        raise ValueError(f"Unknown model: {model_name}")


def hu_sugiyama_z_star(omega_b, omega_m):
    """Hu & Sugiyama (1996, ApJ 471, 542) fitting formula for photon decoupling redshift z_*."""
    g1 = (0.0783 * omega_b ** (-0.238)) / (1.0 + 39.5 * omega_b ** 0.763)
    g2 = 0.560 / (1.0 + 21.1 * omega_b ** 1.81)
    return 1048.0 * (1.0 + 0.00124 * omega_b ** (-0.738)) * (1.0 + g1 * omega_m ** g2)


def rd_physical_mpc(Om, h, omega_b):
    """Aubourg et al. / Planck-calibrated physical drag-epoch sound horizon r_d in Mpc."""
    omega_m = Om * h * h
    return 147.09 * (omega_m / 0.1430) ** (-0.255) * (omega_b / 0.02236) ** (-0.128)


# =====================================================================
# 2. UNIFIED COSMOLOGICAL LIKELIHOOD ENGINE
# =====================================================================

class CosmologyLikelihoodEngine:
    """
    Canonical likelihood evaluator unifying SNe Ia (Pantheon+ or DES-SN5YR Dovekie),
    DESI BAO (DR1 or DR2), and Planck 2018 compressed CMB priors (3D or 1D diagnostic).
    """

    def __init__(
        self,
        sn_dataset="pantheon_plus",
        bao_dataset="desi_dr1",
        pantheon_dir=None,
        des_dir=None,
        bao_dir=None,
    ):
        self.sn_dataset = str(sn_dataset).lower().strip()
        self.bao_dataset = str(bao_dataset).lower().strip()

        if self.sn_dataset in ("pantheon_plus", "pantheon+", "pantheon"):
            sn_data = load_pantheon_plus(pantheon_dir)
        elif self.sn_dataset in ("des_sn5yr", "des_y5", "des5yr", "dovekie"):
            sn_data = load_des_sn5yr(des_dir)
        else:
            raise ValueError(f"Unknown sn_dataset '{sn_dataset}'.")

        self.n_sn = sn_data["n_sn"]
        self.zhd = sn_data["zhd"]
        self.zhel = sn_data["zhel"]
        self.mb = sn_data["mb"]
        self.cov_sn = sn_data["cov"]
        self.inv_cov_sn = sn_data["inv_cov"]
        self.Cinv_sn_proj = sn_data["Cinv_proj"]
        self.e_sn = sn_data["e_sn"]

        # Load BAO release (preserving dynamic observable ordering)
        self.bao_specs, self.y_bao, self.C_bao, self.Cinv_bao = load_desi_bao(
            self.bao_dataset, data_dir=bao_dir
        )
        self.n_bao = len(self.y_bao)

        # Preserve legacy desi_bao_spec attribute name for compatibility when using DR1
        self.desi_bao_spec = self.bao_specs

        # Planck 2018 compressed CMB distance priors (Chen, Huang & Wang 2019, JCAP 02, 028, Table 1)
        # X = (R, l_A, omega_b) = (1.7502, 301.471, 0.02236)
        self.y_cmb = np.array([1.7502, 301.471, 0.02236], dtype=np.float64)
        self.sig_cmb = np.array([0.0046, 0.090, 0.00015], dtype=np.float64)
        corr_cmb = np.array(
            [
                [1.0, 0.46, -0.66],
                [0.46, 1.0, -0.33],
                [-0.66, -0.33, 1.0],
            ],
            dtype=np.float64,
        )
        self.C_cmb = np.outer(self.sig_cmb, self.sig_cmb) * corr_cmb
        self.Cinv_cmb = np.linalg.inv(self.C_cmb)

        # Refined integration grids matching quad to < 1e-5
        self.z_low = np.linspace(0.0, 3.0, 1000)
        self.dz_low = np.diff(self.z_low)
        self.z_high = np.geomspace(3.0, 1200.0, 900)
        self.dz_high = np.diff(self.z_high)
        self.u_rs = np.linspace(0.0, 1.0, 800)

    def _effective_radiation(self, Om, h, omega_b, cmb_mode="planck_3d"):
        """
        Compute effective present-day radiation density Omega_r and baryon-photon ratio coeff.

        Under Chen, Huang & Wang (2019) ('planck_3d'), 1 + z_eq = 2.5e4 * omega_m * (T_CMB/2.7)^-4
        so Omega_r = Omega_m / (1 + z_eq) = (T_CMB/2.7)^4 / (2.5e4 * h^2) and
        R_b(a) = 31500 * omega_b * (T_CMB/2.7)^-4 * a.
        """
        t_ratio4 = (T_CMB / 2.7) ** (-4)
        if cmb_mode == "planck_1d_R":
            Or = OMEGA_R_H2 / (h * h)
            Rb_coeff = 3.0 * omega_b / (4.0 * OMEGA_GAMMA_H2)
        else:
            Or = 1.0 / (2.5e4 * h * h * t_ratio4)
            Rb_coeff = 31500.0 * omega_b * t_ratio4
        return Or, Rb_coeff

    def compute_observables(self, Om, model_name, de_params, h=0.675, omega_b=0.02236,
                            cmb_mode="planck_3d"):
        """Return background cosmological observables (E_low, cum_I_low, R_th, l_A_th, rd_mpc)."""
        Or, Rb_coeff = self._effective_radiation(Om, h, omega_b, cmb_mode=cmb_mode)
        Ode = 1.0 - Om - Or
        if Om <= 0.05 or Om >= 0.70 or Ode <= 0.05:
            return None

        de_low = rho_de_ratio(self.z_low, model_name, de_params)
        E2_low = Or * (1.0 + self.z_low) ** 4 + Om * (1.0 + self.z_low) ** 3 + Ode * de_low
        if np.any(E2_low <= 1e-8) or not np.all(np.isfinite(E2_low)):
            return None

        E_low = np.sqrt(E2_low)
        invE_low = 1.0 / E_low
        cum_I_low = np.concatenate([[0.0], np.cumsum(0.5 * (invE_low[:-1] + invE_low[1:]) * self.dz_low)])

        de_high = rho_de_ratio(self.z_high, model_name, de_params)
        E2_high = Or * (1.0 + self.z_high) ** 4 + Om * (1.0 + self.z_high) ** 3 + Ode * de_high
        if np.any(E2_high <= 1e-8) or not np.all(np.isfinite(E2_high)):
            return None

        invE_high = 1.0 / np.sqrt(E2_high)
        cum_I_high = cum_I_low[-1] + np.concatenate(
            [[0.0], np.cumsum(0.5 * (invE_high[:-1] + invE_high[1:]) * self.dz_high)]
        )

        omega_m = Om * h * h
        z_star = 1089.92 if cmb_mode == "planck_1d_R" else hu_sugiyama_z_star(omega_b, omega_m)
        I_star = float(np.interp(z_star, self.z_high, cum_I_high))
        R_th = float(np.sqrt(Om) * I_star)

        # Sound horizon at decoupling r_s(z_*)
        a_star = 1.0 / (1.0 + z_star)
        a_arr = a_star * self.u_rs[1:]
        z_rs = (1.0 / a_arr) - 1.0
        de_rs = rho_de_ratio(z_rs, model_name, de_params)
        E2_rs = Or * (1.0 + z_rs) ** 4 + Om * (1.0 + z_rs) ** 3 + Ode * de_rs
        if np.any(E2_rs <= 1e-8) or not np.all(np.isfinite(E2_rs)):
            return None

        R_b = Rb_coeff * a_arr
        c_s = 1.0 / np.sqrt(3.0 * (1.0 + R_b))
        integrand_rs = c_s / (a_arr * a_arr * np.sqrt(E2_rs))
        int0 = 1.0 / np.sqrt(3.0 * Or)
        integrand_full = np.concatenate([[int0], integrand_rs])
        da = a_star * (self.u_rs[1] - self.u_rs[0])
        I_rs = float(np.sum(0.5 * (integrand_full[:-1] + integrand_full[1:]) * da))
        l_A_th = float(np.pi * I_star / I_rs)

        return {
            "E_low": E_low,
            "cum_I_low": cum_I_low,
            "z_star": float(z_star),
            "R_th": R_th,
            "l_A_th": l_A_th,
            "rd_mpc": float(rd_physical_mpc(Om, h, omega_b)),
        }

    def compute_bao_model_vector(self, obs, Om, h, omega_b, specs=None, profile_srd=False):
        """Evaluate BAO theory vector aligned strictly with `specs` (or `self.bao_specs`)."""
        use_specs = self.bao_specs if specs is None else specs
        E_low = obs["E_low"]
        cum_I_low = obs["cum_I_low"]
        m0 = np.empty(len(use_specs), dtype=np.float64)
        for idx, (z_b, qty) in enumerate(use_specs):
            I_b = float(np.interp(z_b, self.z_low, cum_I_low))
            E_b = float(np.interp(z_b, self.z_low, E_low))
            if qty == "DV_over_rs":
                m0[idx] = (z_b * I_b * I_b / E_b) ** (1.0 / 3.0)
            elif qty == "DM_over_rs":
                m0[idx] = I_b
            elif qty == "DH_over_rs":
                m0[idx] = 1.0 / E_b
            else:
                raise ValueError(f"Unsupported BAO quantity '{qty}'")

        if profile_srd:
            return m0, None
        s_phys = C_LIGHT / (100.0 * h * obs["rd_mpc"])
        return s_phys * m0, s_phys

    def compute_chi2_components(
        self,
        Om,
        model_name,
        de_params,
        h=0.6748,
        omega_b=0.02236,
        use_sn=True,
        use_bao=True,
        use_cmb=True,
        cmb_mode="planck_3d",
    ):
        """
        Compute (chi2_sn, chi2_bao, chi2_cmb) consistently across all active probes.
        """
        obs = self.compute_observables(
            Om, model_name, de_params, h=h, omega_b=omega_b, cmb_mode=cmb_mode
        )
        if obs is None:
            return 1e12, 1e12, 1e12

        chi2_sn = 0.0
        if use_sn:
            chi_sn = np.interp(self.zhd, self.z_low, obs["cum_I_low"])
            dl_sn = (1.0 + self.zhel) * (C_LIGHT / 70.0) * chi_sn
            if np.any(dl_sn <= 0.0) or not np.all(np.isfinite(dl_sn)):
                return 1e12, 1e12, 1e12
            mu0 = 5.0 * np.log10(dl_sn) + 25.0
            r_sn = self.mb - mu0
            chi2_sn = float(r_sn @ (self.Cinv_sn_proj @ r_sn))

        chi2_bao = 0.0
        if use_bao:
            # In SN+BAO without CMB or in v2.0 1D diagnostic mode, profile analytically over s_rd
            profile_srd = (not use_cmb) or (cmb_mode == "planck_1d_R")
            m_vec, _ = self.compute_bao_model_vector(
                obs, Om, h, omega_b, specs=self.bao_specs, profile_srd=profile_srd
            )
            if profile_srd:
                A_b = float(m_vec @ (self.Cinv_bao @ m_vec))
                B_b = float(m_vec @ (self.Cinv_bao @ self.y_bao))
                s_best = B_b / A_b
                res_b = self.y_bao - s_best * m_vec
            else:
                res_b = self.y_bao - m_vec
            chi2_bao = float(res_b @ (self.Cinv_bao @ res_b))

        chi2_cmb = 0.0
        if use_cmb:
            if cmb_mode == "planck_1d_R":
                chi2_cmb = float(((obs["R_th"] - self.y_cmb[0]) / self.sig_cmb[0]) ** 2)
            else:
                delta_cmb = np.array(
                    [obs["R_th"] - self.y_cmb[0], obs["l_A_th"] - self.y_cmb[1], omega_b - self.y_cmb[2]],
                    dtype=np.float64,
                )
                chi2_cmb = float(delta_cmb @ (self.Cinv_cmb @ delta_cmb))

        return chi2_sn, chi2_bao, chi2_cmb
