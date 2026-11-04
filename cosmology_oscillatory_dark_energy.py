"""
Logarithmic Oscillatory Dark Energy vs. LCDM and CPL
Empirical validation against real Pantheon+ Supernovae (1590 SNe Ia)
and predictive forecast for the ESA Euclid Space Telescope (z in [0.5, 3.0]).

Author: Bruno Luçardo Olivaes
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import minimize
import os

C_LIGHT = 299792.458  # km/s

# =====================================================================
# 1. EXPANSION RATE MODELS
# =====================================================================

def E_LCDM(z, Om):
    return np.sqrt(Om * (1.0 + z)**3 + (1.0 - Om))

def E_CPL(z, Om, w0, wa):
    de_term = (1.0 + z)**(3.0 * (1.0 + w0 + wa)) * np.exp(-3.0 * wa * z / (1.0 + z))
    return np.sqrt(Om * (1.0 + z)**3 + (1.0 - Om) * de_term)

def E_Oscillatory(z, Om, A_bulk, lambda_bulk, phi0):
    k = 2.0 * np.pi / lambda_bulk
    arg = -k * np.log(1.0 + z) + phi0
    integral_val = (3.0 * A_bulk / k) * (np.cos(phi0) - np.cos(arg))
    de_ratio = np.exp(-integral_val)
    return np.sqrt(Om * (1.0 + z)**3 + (1.0 - Om) * de_ratio)

def w_CPL(z, w0, wa):
    a = 1.0 / (1.0 + z)
    return w0 + wa * (1.0 - a)

def w_Oscillatory(z, A_bulk, lambda_bulk, phi0):
    a = 1.0 / (1.0 + z)
    return -1.0 + A_bulk * np.sin((2.0 * np.pi / lambda_bulk) * np.log(a) + phi0)

# =====================================================================
# 2. COSMOLOGICAL DISTANCE INTEGRALS
# =====================================================================

def dl_calc(z, H0, E_func, *args):
    def integrand(zp):
        ez = E_func(zp, *args)
        return 1.0 / ez if ez > 0 else 1e10
    val, _ = quad(integrand, 0.0, z)
    return (1.0 + z) * (C_LIGHT / H0) * val

def mu_calc(z, H0, E_func, *args):
    d = dl_calc(z, H0, E_func, *args)
    return 5.0 * np.log10(d) + 25.0

# Vectorized distance modulus using fast grid interpolation
def compute_mu_grid(z_vals, H0, E_func, *args):
    z_dense = np.linspace(0.001, max(3.5, np.max(z_vals) * 1.05), 300)
    mu_dense = np.array([mu_calc(zi, H0, E_func, *args) for zi in z_dense])
    return np.interp(z_vals, z_dense, mu_dense)

# =====================================================================
# 3. FIT TO PANTHEON+ DATA
# =====================================================================

def run_cosmology_audit():
    print("=" * 70)
    print("COSMOLOGY AUDIT: PANTHEON+ DATA & EUCLID TURNAROUND FORECAST")
    print("=" * 70)

    data_path = r"C:\Users\bruno\OneDrive\Desktop\_Archive_Ouroboros_Antigo\Pantheon_Data\Pantheon+_Data\4_DISTANCES_AND_COVAR\Pantheon+SH0ES.dat"
    df = pd.read_csv(data_path, sep=r'\s+')
    df_clean = df[(df['zHD'] > 0.01) & (df['m_b_corr_err_DIAG'] > 0)].copy()

    z_obs = df_clean['zHD'].values
    mu_obs = df_clean['MU_SH0ES'].values
    mu_err = df_clean['MU_SH0ES_ERR_DIAG'].values
    N_data = len(z_obs)
    print(f"Loaded {N_data} clean Supernovae Ia from Pantheon+ (z in [{z_obs.min():.4f}, {z_obs.max():.4f}])")

    # Sort by redshift
    idx_sort = np.argsort(z_obs)
    z_obs = z_obs[idx_sort]
    mu_obs = mu_obs[idx_sort]
    mu_err = mu_err[idx_sort]

    # --- Fit Model 1: LCDM (H0, Om) ---
    print("\n[1/3] Fitting Standard Lambda-CDM...")
    def chi2_LCDM(params):
        H0, Om = params
        mu_pred = compute_mu_grid(z_obs, H0, E_LCDM, Om)
        return np.sum(((mu_obs - mu_pred) / mu_err)**2)

    res_lcdm = minimize(chi2_LCDM, [73.0, 0.30], bounds=[(60.0, 80.0), (0.1, 0.5)], method='L-BFGS-B')
    H0_l, Om_l = res_lcdm.x
    chi2_l = res_lcdm.fun
    bic_l = chi2_l + 2 * np.log(N_data)
    aic_l = chi2_l + 2 * 2
    print(f"  LCDM: H0 = {H0_l:.2f} km/s/Mpc, Om = {Om_l:.3f} | Chi2 = {chi2_l:.2f}, BIC = {bic_l:.2f}")

    # --- Fit Model 2: DESI CPL (H0, Om, w0, wa) ---
    print("\n[2/3] Fitting Dynamical Dark Energy (CPL)...")
    def chi2_CPL(params):
        H0, Om, w0, wa = params
        mu_pred = compute_mu_grid(z_obs, H0, E_CPL, Om, w0, wa)
        return np.sum(((mu_obs - mu_pred) / mu_err)**2)

    res_cpl = minimize(chi2_CPL, [H0_l, Om_l, -0.83, -0.75], 
                       bounds=[(65.0, 78.0), (0.2, 0.4), (-1.5, -0.5), (-2.0, 0.5)], method='L-BFGS-B')
    H0_c, Om_c, w0_c, wa_c = res_cpl.x
    chi2_c = res_cpl.fun
    bic_c = chi2_c + 4 * np.log(N_data)
    aic_c = chi2_c + 2 * 4
    print(f"  CPL: H0 = {H0_c:.2f}, Om = {Om_c:.3f}, w0 = {w0_c:.3f}, wa = {wa_c:.3f} | Chi2 = {chi2_c:.2f}, BIC = {bic_c:.2f}")

    # --- Fit Model 3: Logarithmic Oscillatory (H0, Om, A_bulk, lambda_bulk, phi0) ---
    print("\n[3/3] Fitting Logarithmic Oscillatory Dark Energy...")
    def chi2_Osc(params):
        H0, Om, A_b, lam_b, phi0 = params
        mu_pred = compute_mu_grid(z_obs, H0, E_Oscillatory, Om, A_b, lam_b, phi0)
        return np.sum(((mu_obs - mu_pred) / mu_err)**2)

    res_osc = minimize(chi2_Osc, [H0_l, Om_l, 0.47, 6.42, 0.12],
                       bounds=[(65.0, 78.0), (0.2, 0.4), (0.1, 0.9), (3.0, 10.0), (-np.pi, np.pi)], method='L-BFGS-B')
    H0_o, Om_o, A_o, lam_o, phi_o = res_osc.x
    chi2_o = res_osc.fun
    bic_o = chi2_o + 5 * np.log(N_data)
    aic_o = chi2_o + 2 * 5
    print(f"  Oscillatory: H0 = {H0_o:.2f}, Om = {Om_o:.3f}, A_bulk = {A_o:.3f}, lambda = {lam_o:.2f}, phi0 = {phi_o:.3f}")
    print(f"  Chi2 = {chi2_o:.2f}, BIC = {bic_o:.2f}")

    # =====================================================================
    # 4. PREDICTIONS FOR EUCLID IN THE DARK AGES (z in [0.5, 3.0])
    # =====================================================================
    print("\n" + "=" * 70)
    print("FALSIFIABLE PREDICTIONS FOR ESA EUCLID MISSION (z = 0.5 to 3.0)")
    print("=" * 70)
    z_forecast = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    print(f"{'Redshift z':<10} {'w(z) CPL':<12} {'w(z) Osc':<12} {'mu_LCDM':<10} {'mu_CPL':<10} {'mu_Osc':<10} {'Delta_mu (Osc-CPL)':<18}")
    print("-" * 84)
    for zf in z_forecast:
        wc = w_CPL(zf, w0_c, wa_c)
        wo = w_Oscillatory(zf, A_o, lam_o, phi_o)
        ml = mu_calc(zf, H0_l, E_LCDM, Om_l)
        mc = mu_calc(zf, H0_c, E_CPL, Om_c, w0_c, wa_c)
        mo = mu_calc(zf, H0_o, E_Oscillatory, Om_o, A_o, lam_o, phi_o)
        delta_mag = mo - mc
        print(f"{zf:<10.2f} {wc:<12.4f} {wo:<12.4f} {ml:<10.3f} {mc:<10.3f} {mo:<10.3f} {delta_mag:<+18.4f}")

    # =====================================================================
    # 5. GENERATE PUBLICATION FIGURE
    # =====================================================================
    print("\nGenerating publication figure...")
    out_dir = r"C:\Users\bruno\casimir-afm-spectroscopy\output"
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(11, 14), gridspec_kw={'height_ratios': [2, 1.3, 1.3]})

    # Plot 1: Hubble Diagram
    ax0 = axes[0]
    ax0.errorbar(z_obs, mu_obs, yerr=mu_err, fmt='.', color='lightgray', alpha=0.4, markersize=3, label='Pantheon+ SNe Ia (1590 points)')
    z_plot = np.linspace(0.01, 3.2, 300)
    mu_plot_l = compute_mu_grid(z_plot, H0_l, E_LCDM, Om_l)
    mu_plot_c = compute_mu_grid(z_plot, H0_c, E_CPL, Om_c, w0_c, wa_c)
    mu_plot_o = compute_mu_grid(z_plot, H0_o, E_Oscillatory, Om_o, A_o, lam_o, phi_o)

    ax0.plot(z_plot, mu_plot_l, 'k--', lw=1.8, label=r'$\Lambda\mathrm{CDM}$ ($H_0 = ' + f'{H0_l:.1f}' + r'$)')
    ax0.plot(z_plot, mu_plot_c, 'b-', lw=1.8, label=r'DESI CPL ($w_0=' + f'{w0_c:.2f}, w_a={wa_c:.2f}' + r'$)')
    ax0.plot(z_plot, mu_plot_o, 'r-', lw=2.2, label=r'Oscillatory Quintessence ($A_{\mathrm{bulk}}=' + f'{A_o:.2f}' + r'$)')
    ax0.set_ylabel('Distance Modulus $\mu(z)$ (mag)', fontsize=11)
    ax0.set_title('Cosmological Expansion: Empirical Fit to Pantheon+ and Forecast for Euclid', fontsize=13, fontweight='bold')
    ax0.legend(loc='lower right', fontsize=10)
    ax0.grid(True, alpha=0.3)

    # Plot 2: Dark Energy Equation of State w(z)
    ax1 = axes[1]
    w_vals_c = [w_CPL(zi, w0_c, wa_c) for zi in z_plot]
    w_vals_o = [w_Oscillatory(zi, A_o, lam_o, phi_o) for zi in z_plot]
    ax1.axhline(-1.0, color='black', linestyle='--', label=r'$\Lambda\mathrm{CDM}$ ($w = -1$)')
    ax1.plot(z_plot, w_vals_c, 'b-', lw=2.0, label='Linear CPL Parameterization (DESI 2024)')
    ax1.plot(z_plot, w_vals_o, 'r-', lw=2.5, label='Logarithmic Oscillatory Model (Turnaround Prediction)')
    ax1.axvspan(1.0, 3.0, color='gold', alpha=0.15, label='Euclid Deep Redshift Window ($1 < z < 3$)')
    ax1.set_ylabel('Equation of State $w(z)$', fontsize=11)
    ax1.set_title('Equation of State Evolution: CPL Monotonicity vs. Oscillatory Concavity', fontsize=12, fontweight='bold')
    ax1.set_ylim(-1.8, -0.4)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot 3: Residual Divergence Delta_mu
    ax2 = axes[2]
    delta_mu_plot = mu_plot_o - mu_plot_c
    ax2.plot(z_plot, delta_mu_plot, color='purple', lw=2.2, label=r'$\Delta\mu = \mu_{\mathrm{Osc}} - \mu_{\mathrm{CPL}}$')
    ax2.axhline(0.0, color='black', linestyle=':')
    ax2.axvspan(1.0, 3.0, color='gold', alpha=0.15)
    ax2.fill_between(z_plot, -0.05, 0.05, color='green', alpha=0.15, label='Euclid Weak Lensing Sensitivity Band ($\pm 0.05$ mag)')
    ax2.set_xlabel('Redshift $z$', fontsize=11)
    ax2.set_ylabel('$\Delta\mu(z)$ (mag)', fontsize=11)
    ax2.set_title('Predictive Divergence: Measurable Separation in Euclid Data', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(out_dir, "fig_euclid_hubble_forecast.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Saved publication figure: {fig_path}")
    print("\nCosmological Audit complete!")

if __name__ == '__main__':
    run_cosmology_audit()

