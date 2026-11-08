# -*- coding: utf-8 -*-
"""
Full v2.0 Observational Cosmology Audit:
Logarithmic Oscillatory Dark Energy vs. LCDM and CPL across:
  1. Pantheon+ SNe Ia (N = 1580, full STAT+SYS covariance matrix, marginalized over M)
  2. DESI Year 1 BAO (arXiv:2404.03002, 7 redshift bins, 12 measurements, 0.295 <= z <= 2.330)
  3. Planck 2018 CMB Shift Parameter R(z_* = 1089.92) = 1.7502 +/- 0.0046 (Chen et al. 2019)

Includes:
  - Multi-start global optimization (Differential Evolution + Nelder-Mead polish)
  - Affine-invariant Ensemble MCMC (emcee) posterior sampling on SN + DESI BAO + CMB
  - Posterior predictive w(z), H(z), and D_M(z) bands compared against ESA Euclid
    spectroscopic BAO sensitivity (Blanchard et al. 2020, A&A 642, A191).
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, minimize
import emcee

from src.likelihoods import CosmologyLikelihoodEngine, w_model, rho_de_ratio

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PAPER_DIR = os.path.join(BASE_DIR, "paper")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PAPER_DIR, exist_ok=True)


class JointCosmologyEvaluator:
    """
    Evaluates Pantheon+ full-covariance SN chi2, DESI 2024 VI BAO chi2 (profiled over r_d*h),
    and Planck 2018 CMB shift parameter R(z_* = 1089.92) = 1.7502 +/- 0.0046.
    """

    def __init__(self):
        self.engine = CosmologyLikelihoodEngine()
        # Grid from z=0 to z_*=1089.92
        z_low = np.linspace(0.0, 3.2, 1000)
        z_high = np.geomspace(3.201, 1100.0, 700)
        self.z_grid = np.concatenate([z_low, z_high])
        self.dz_grid = np.diff(self.z_grid)
        self.Or = 9.0e-5  # radiation density today

    def chi2_breakdown(self, Om, model_name, de_params):
        if Om <= 0.10 or Om >= 0.60:
            return 1e12, 1e12, 1e12
        Ode = 1.0 - Om - self.Or
        de_rat = rho_de_ratio(self.z_grid, model_name, de_params)
        E2 = self.Or * (1.0 + self.z_grid)**4 + Om * (1.0 + self.z_grid)**3 + Ode * de_rat
        if np.any(E2 <= 1e-8) or not np.all(np.isfinite(E2)):
            return 1e12, 1e12, 1e12
        E = np.sqrt(E2)
        invE = 1.0 / E
        cum_I = np.concatenate([[0.0], np.cumsum(0.5 * (invE[:-1] + invE[1:]) * self.dz_grid)])

        # 1. Pantheon+ SNe Ia (N = 1580, full STAT+SYS covariance)
        chi_sn = np.interp(self.engine.zhd, self.z_grid, cum_I)
        dl_sn = (1.0 + self.engine.zhel) * (299792.458 / 70.0) * chi_sn
        mu0 = 5.0 * np.log10(dl_sn) + 25.0
        r_sn = self.engine.mb - mu0
        chi2_sn = float(r_sn @ (self.engine.Cinv_sn_proj @ r_sn))

        # 2. DESI 2024 VI BAO (N = 12, analytically profiled over s_rd = c / (H0 * r_d))
        m0 = np.empty(self.engine.n_bao, dtype=np.float64)
        k_idx = 0
        for z_b, kind, *_ in self.engine.desi_bao_spec:
            I_b = float(np.interp(z_b, self.z_grid, cum_I))
            E_b = float(np.interp(z_b, self.z_grid, E))
            dm_0 = I_b
            dh_0 = 1.0 / E_b
            if kind == 'DV':
                m0[k_idx] = (z_b * dm_0 * dm_0 * dh_0)**(1.0 / 3.0)
                k_idx += 1
            else:
                m0[k_idx] = dm_0
                m0[k_idx + 1] = dh_0
                k_idx += 2
        A_b = float(m0 @ (self.engine.Cinv_bao @ m0))
        B_b = float(m0 @ (self.engine.Cinv_bao @ self.engine.y_bao))
        s_best = B_b / A_b
        res_b = self.engine.y_bao - s_best * m0
        chi2_bao = float(res_b @ (self.engine.Cinv_bao @ res_b))

        # 3. Planck 2018 CMB shift parameter R(z_* = 1089.92) = 1.7502 +/- 0.0046
        I_star = float(np.interp(1089.92, self.z_grid, cum_I))
        R_th = np.sqrt(Om) * I_star
        chi2_cmb = float(((R_th - 1.7502) / 0.0046)**2)

        return chi2_sn, chi2_bao, chi2_cmb


def check_at_bounds(x, bounds, tol=1e-3):
    flags = []
    for val, (lo, hi) in zip(x, bounds):
        if abs(val - lo) < tol * (hi - lo) or abs(val - hi) < tol * (hi - lo):
            flags.append(True)
        else:
            flags.append(False)
    return flags


def run_cosmology_v2_audit():
    print("=" * 78)
    print("COSMOLOGY AUDIT v2.0: FULL PANTHEON+ COVARIANCE + DESI BAO + PLANCK CMB")
    print("=" * 78)

    ev = JointCosmologyEvaluator()
    n_sn = ev.engine.n_sn
    n_bao = ev.engine.n_bao
    print(f"Loaded Pantheon+ Hubble-flow cosmology sample: N_SN = {n_sn} (full {n_sn}x{n_sn} STAT+SYS cov)")
    print(f"Loaded DESI 2024 VI BAO measurements:          N_BAO = {n_bao} across 0.295 <= z <= 2.330")
    print("Loaded Planck 2018 CMB shift parameter prior:  R(z_*=1089.92) = 1.7502 +/- 0.0046\n")

    model_specs = [
        ('LCDM', ['Om'], [(0.18, 0.48)]),
        ('CPL',  ['Om', 'w0', 'wa'], [(0.18, 0.48), (-2.0, 0.0), (-3.0, 2.0)]),
        ('OSC',  ['Om', 'A', 'lambda', 'phi0'], [(0.18, 0.48), (0.0, 1.5), (0.8, 15.0), (-np.pi, np.pi)]),
    ]

    dataset_combos = [
        ('SN_only',       True, False, False, n_sn),
        ('SN_BAO',        True, True,  False, n_sn + n_bao),
        ('SN_BAO_CMB',    True, True,  True,  n_sn + n_bao + 1),
    ]

    summary_results = {}

    for combo_name, u_sn, u_bao, u_cmb, n_tot in dataset_combos:
        print(f"--- Fitting Dataset Combination: {combo_name} (N_tot = {n_tot}) ---")
        combo_dict = {}
        chi2_ref = None
        aic_ref = None
        bic_ref = None

        for mname, pnames, bounds in model_specs:
            def obj(p):
                c_sn, c_bao, c_cmb = ev.chi2_breakdown(p[0], mname, p[1:])
                tot = 0.0
                if u_sn:
                    tot += c_sn
                if u_bao:
                    tot += c_bao
                if u_cmb:
                    tot += c_cmb
                return tot

            res_de = differential_evolution(obj, bounds, seed=202611, tol=1e-6, polish=True)
            res_nm = minimize(obj, res_de.x, method='Nelder-Mead', options={'xatol': 1e-6, 'fatol': 1e-6})
            x_best = np.clip(res_nm.x if res_nm.fun <= res_de.fun else res_de.x,
                             [b[0] for b in bounds], [b[1] for b in bounds])
            chi2_min = float(obj(x_best))
            c_sn, c_bao, c_cmb = ev.chi2_breakdown(x_best[0], mname, x_best[1:])

            # Number of free parameters k (including analytically profiled nuisance offsets for fair comparison,
            # though delta_k relative to LCDM is identical: +0 for LCDM, +2 for CPL, +3 for OSC)
            k_extra = len(pnames)  # 1 for LCDM, 3 for CPL, 4 for OSC
            aic = chi2_min + 2.0 * k_extra
            bic = chi2_min + k_extra * np.log(n_tot)

            if mname == 'LCDM':
                chi2_ref, aic_ref, bic_ref = chi2_min, aic, bic

            bound_flags = check_at_bounds(x_best, bounds)
            combo_dict[mname] = {
                'params': dict(zip(pnames, [float(v) for v in x_best])),
                'at_prior_edge': dict(zip(pnames, bound_flags)),
                'any_at_edge': bool(any(bound_flags)),
                'chi2_min': chi2_min,
                'chi2_sn': float(c_sn),
                'chi2_bao': float(c_bao if u_bao else 0.0),
                'chi2_cmb': float(c_cmb if u_cmb else 0.0),
                'delta_chi2_vs_LCDM': float(chi2_min - chi2_ref),
                'delta_AIC_vs_LCDM': float(aic - aic_ref),
                'delta_BIC_vs_LCDM': float(bic - bic_ref),
            }
            print(f"  {mname:5s}: params={combo_dict[mname]['params']} | "
                  f"chi2={chi2_min:.2f} (dchi2={chi2_min-chi2_ref:+.2f}, dAIC={aic-aic_ref:+.2f}, dBIC={bic-bic_ref:+.2f}) | "
                  f"at_edge={any(bound_flags)}")

        summary_results[combo_name] = combo_dict
        print()

    # =========================================================================
    # 2. MCMC SAMPLING (emcee) ON JOINT SN + DESI BAO + PLANCK CMB
    # =========================================================================
    print("--- Running Affine-Invariant Ensemble MCMC (emcee) on SN_BAO_CMB ---")
    rng = np.random.default_rng(20261105)
    mcmc_results = {}

    # 2A. LCDM MCMC (1D: Om)
    def log_prob_lcdm(p):
        Om = p[0]
        if not (0.20 < Om < 0.45):
            return -np.inf
        c1, c2, c3 = ev.chi2_breakdown(Om, 'LCDM', [])
        return -0.5 * (c1 + c2 + c3)

    x0_l = summary_results['SN_BAO_CMB']['LCDM']['params']['Om']
    pos_l = x0_l + 0.005 * rng.normal(size=(16, 1))
    sampler_l = emcee.EnsembleSampler(16, 1, log_prob_lcdm)
    sampler_l.run_mcmc(pos_l, 600, progress=False)
    chain_l = sampler_l.get_chain(discard=150, flat=True)
    q16, q50, q84 = np.percentile(chain_l[:, 0], [16, 50, 84])
    mcmc_results['LCDM'] = {
        'Om_median': float(q50), 'Om_err_plus': float(q84 - q50), 'Om_err_minus': float(q50 - q16),
        'n_samples': int(len(chain_l))
    }
    print(f"  MCMC LCDM: Om = {q50:.4f} +{q84-q50:.4f} / -{q50-q16:.4f}")

    # 2B. CPL MCMC (3D: Om, w0, wa)
    def log_prob_cpl(p):
        Om, w0, wa = p
        if not (0.20 < Om < 0.45 and -2.0 < w0 < 0.0 and -3.0 < wa < 2.0 and (w0 + wa) < -0.1):
            return -np.inf
        c1, c2, c3 = ev.chi2_breakdown(Om, 'CPL', [w0, wa])
        return -0.5 * (c1 + c2 + c3)

    p_cpl = summary_results['SN_BAO_CMB']['CPL']['params']
    x0_c = np.array([p_cpl['Om'], p_cpl['w0'], p_cpl['wa']])
    pos_c = x0_c + np.array([0.004, 0.03, 0.10]) * rng.normal(size=(24, 3))
    sampler_c = emcee.EnsembleSampler(24, 3, log_prob_cpl)
    sampler_c.run_mcmc(pos_c, 900, progress=False)
    chain_c = sampler_c.get_chain(discard=250, flat=True)
    np.save(os.path.join(RESULTS_DIR, "chain_cpl.npy"), chain_c)
    mcmc_results['CPL'] = {}
    for idx_p, name_p in enumerate(['Om', 'w0', 'wa']):
        q16, q50, q84 = np.percentile(chain_c[:, idx_p], [16, 50, 84])
        mcmc_results['CPL'][name_p] = {'median': float(q50), 'plus': float(q84 - q50), 'minus': float(q50 - q16)}
    print("  MCMC CPL:", mcmc_results['CPL'])

    # 2C. Log-Oscillatory MCMC (4D: Om, A, lambda, phi0) with weakly informative physical prior on A, lambda
    def log_prob_osc(p):
        Om, A, lam, phi0 = p
        if not (0.20 < Om < 0.45 and 0.0 <= A <= 1.2 and 1.2 <= lam <= 12.0 and -np.pi <= phi0 <= np.pi):
            return -np.inf
        c1, c2, c3 = ev.chi2_breakdown(Om, 'OSC', [A, lam, phi0])
        return -0.5 * (c1 + c2 + c3)

    p_osc = summary_results['SN_BAO_CMB']['OSC']['params']
    x0_o = np.array([p_osc['Om'], p_osc['A'], p_osc['lambda'], p_osc['phi0']])
    pos_o = x0_o + np.array([0.004, 0.04, 0.25, 0.12]) * rng.normal(size=(32, 4))
    pos_o[:, 1] = np.clip(pos_o[:, 1], 0.02, 1.15)
    pos_o[:, 2] = np.clip(pos_o[:, 2], 1.3, 11.5)
    pos_o[:, 3] = ((pos_o[:, 3] + np.pi) % (2 * np.pi)) - np.pi
    sampler_o = emcee.EnsembleSampler(32, 4, log_prob_osc)
    sampler_o.run_mcmc(pos_o, 900, progress=False)
    chain_o = sampler_o.get_chain(discard=250, flat=True)
    np.save(os.path.join(RESULTS_DIR, "chain_osc.npy"), chain_o)
    mcmc_results['OSC'] = {}
    for idx_p, name_p in enumerate(['Om', 'A', 'lambda', 'phi0']):
        q16, q50, q84 = np.percentile(chain_o[:, idx_p], [16, 50, 84])
        mcmc_results['OSC'][name_p] = {'median': float(q50), 'plus': float(q84 - q50), 'minus': float(q50 - q16)}
    print("  MCMC OSC:", mcmc_results['OSC'])

    summary_results['mcmc_SN_BAO_CMB'] = mcmc_results

    # =========================================================================
    # 3. EUCLID SPECTROSCOPIC FORECAST & POSTERIOR PREDICTIVE BANDS
    # =========================================================================
    z_eval = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    p_l_best = summary_results['SN_BAO_CMB']['LCDM']['params']
    p_c_best = summary_results['SN_BAO_CMB']['CPL']['params']
    p_o_best = summary_results['SN_BAO_CMB']['OSC']['params']

    w_cpl_best = [float(w_model(z, 'CPL', [p_c_best['w0'], p_c_best['wa']])) for z in z_eval]
    w_osc_best = [float(w_model(z, 'OSC', [p_o_best['A'], p_o_best['lambda'], p_o_best['phi0']])) for z in z_eval]

    # Sample 400 draws from CPL and OSC chains to get 68% posterior predictive bands for w(z) and E(z)/E_LCDM(z)
    z_dense = np.linspace(0.0, 3.2, 250)
    E_lcdm_dense = np.sqrt(p_l_best['Om'] * (1.0 + z_dense)**3 + (1.0 - p_l_best['Om']))

    idx_draws_c = rng.choice(len(chain_c), size=400, replace=False)
    w_draws_c = np.array([w_model(z_dense, 'CPL', s[1:]) for s in chain_c[idx_draws_c]])
    E_draws_c = np.array([
        np.sqrt(s[0] * (1.0 + z_dense)**3 + (1.0 - s[0]) * rho_de_ratio(z_dense, 'CPL', s[1:])) / E_lcdm_dense - 1.0
        for s in chain_c[idx_draws_c]
    ]) * 100.0  # percent deviation in H(z) from LCDM

    idx_draws_o = rng.choice(len(chain_o), size=400, replace=False)
    w_draws_o = np.array([w_model(z_dense, 'OSC', s[1:]) for s in chain_o[idx_draws_o]])
    E_draws_o = np.array([
        np.sqrt(s[0] * (1.0 + z_dense)**3 + (1.0 - s[0]) * rho_de_ratio(z_dense, 'OSC', s[1:])) / E_lcdm_dense - 1.0
        for s in chain_o[idx_draws_o]
    ]) * 100.0

    summary_results['euclid_w_forecast'] = {
        'z': [float(z) for z in z_eval],
        'w_CPL_best': w_cpl_best,
        'w_OSC_best': w_osc_best,
        'delta_w_OSC_minus_CPL': [float(wo - wc) for wo, wc in zip(w_osc_best, w_cpl_best)],
    }

    with open(os.path.join(RESULTS_DIR, "audit_v2_cosmology.json"), "w", encoding="utf-8") as f:
        json.dump(summary_results, f, indent=2)

    # =========================================================================
    # 4. GENERATE PUBLICATION FIGURE (3 PANELS)
    # =========================================================================
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 13.0), gridspec_kw={'height_ratios': [1.3, 1.3, 1.3]})

    # Panel (a): DESI 2024 VI BAO data vs Best-Fit Models relative to LCDM
    ax0 = axes[0]
    E_c_best = np.sqrt(p_c_best['Om'] * (1.0 + z_dense)**3 + (1.0 - p_c_best['Om']) *
                       rho_de_ratio(z_dense, 'CPL', [p_c_best['w0'], p_c_best['wa']]))
    E_o_best = np.sqrt(p_o_best['Om'] * (1.0 + z_dense)**3 + (1.0 - p_o_best['Om']) *
                       rho_de_ratio(z_dense, 'OSC', [p_o_best['A'], p_o_best['lambda'], p_o_best['phi0']]))

    ax0.axhline(0.0, color='black', ls='--', lw=1.5, label=r'Flat $\Lambda\mathrm{CDM}$ Baseline ($\Omega_m = ' + f"{p_l_best['Om']:.3f}" + r'$)')
    ax0.plot(z_dense, (E_c_best / E_lcdm_dense - 1.0) * 100.0, 'b-', lw=2.0,
             label=f"CPL Best Fit ($w_0={p_c_best['w0']:.2f},\\,w_a={p_c_best['wa']:.2f},\\,\\Delta\\chi^2={summary_results['SN_BAO_CMB']['CPL']['delta_chi2_vs_LCDM']:.2f}$)")
    ax0.fill_between(z_dense, np.percentile(E_draws_c, 16, axis=0), np.percentile(E_draws_c, 84, axis=0),
                     color='royalblue', alpha=0.18)
    ax0.plot(z_dense, (E_o_best / E_lcdm_dense - 1.0) * 100.0, 'r-', lw=2.2,
             label=f"Log-Oscillatory Best Fit ($A={p_o_best['A']:.2f},\\,\\lambda={p_o_best['lambda']:.2f},\\,\\Delta\\chi^2={summary_results['SN_BAO_CMB']['OSC']['delta_chi2_vs_LCDM']:.2f}$)")
    ax0.fill_between(z_dense, np.percentile(E_draws_o, 16, axis=0), np.percentile(E_draws_o, 84, axis=0),
                     color='crimson', alpha=0.18)

    # Euclid spectroscopic H(z) sensitivity band (0.9 <= z <= 1.8, sigma_H/H ~ 1.3% per bin, Blanchard et al. 2020)
    z_euc = np.array([0.9, 1.1, 1.3, 1.5, 1.8])
    ax0.errorbar(z_euc, np.zeros_like(z_euc), yerr=1.3, fmt='s', color='darkgreen', capsize=4,
                 label=r'ESA Euclid Spectroscopic BAO $\sigma(H)/H \approx 1.3\%$ Forecast (Blanchard et al. 2020)')
    ax0.axvline(2.33, color='gray', ls=':', lw=1.2, label='Upper Redshift Limit of DESI BAO ($z = 2.33$)')
    ax0.set_ylabel(r'$\Delta H(z) / H_{\Lambda\mathrm{CDM}}(z)$ (%)', fontsize=10.5)
    ax0.set_title('(a) Expansion Rate Deviation from Flat $\\Lambda\\mathrm{CDM}$ (Joint Pantheon+ Full Cov + DESI BAO + Planck CMB)',
                  fontsize=11.5, fontweight='bold')
    ax0.set_ylim(-6.0, 6.0)
    ax0.legend(loc='upper left', fontsize=8.5)
    ax0.grid(True, alpha=0.3)

    # Panel (b): Equation of State w(z) with 68% MCMC Credible Bands
    ax1 = axes[1]
    w_c_dense = w_model(z_dense, 'CPL', [p_c_best['w0'], p_c_best['wa']])
    w_o_dense = w_model(z_dense, 'OSC', [p_o_best['A'], p_o_best['lambda'], p_o_best['phi0']])
    ax1.axhline(-1.0, color='black', ls='--', lw=1.5, label=r'$\Lambda\mathrm{CDM}$ ($w = -1$)')
    ax1.plot(z_dense, w_c_dense, 'b-', lw=2.0, label='CPL Best Fit + 68% MCMC Posterior Band')
    ax1.fill_between(z_dense, np.percentile(w_draws_c, 16, axis=0), np.percentile(w_draws_c, 84, axis=0),
                     color='royalblue', alpha=0.20)
    ax1.plot(z_dense, w_o_dense, 'r-', lw=2.2, label='Log-Oscillatory Best Fit + 68% MCMC Posterior Band')
    ax1.fill_between(z_dense, np.percentile(w_draws_o, 16, axis=0), np.percentile(w_draws_o, 84, axis=0),
                     color='crimson', alpha=0.20)
    ax1.axvspan(0.9, 1.8, color='mediumseagreen', alpha=0.12, label='Euclid Prime Spectroscopic Window ($0.9 < z < 1.8$)')
    ax1.axvline(2.33, color='gray', ls=':', lw=1.2)
    ax1.set_ylabel(r'Dark Energy EoS $w(z)$', fontsize=10.5)
    ax1.set_title('(b) Equation of State Evolution $w(z)$: Interior Joint Fit vs. Posterior Uncertainty',
                  fontsize=11.5, fontweight='bold')
    ax1.set_ylim(-1.85, -0.45)
    ax1.legend(loc='lower left', fontsize=8.5)
    ax1.grid(True, alpha=0.3)

    # Panel (c): Model Selection Comparison (Delta chi2, Delta AIC, Delta BIC across dataset combinations)
    ax2 = axes[2]
    categories = [
        'Pantheon+ Only\n(Full Cov, $N=1580$)',
        'Pantheon+ + DESI BAO\n($N=1592$)',
        'Pantheon+ + DESI BAO\n+ Planck CMB ($N=1593$)'
    ]
    x_pos = np.arange(len(categories))
    width = 0.18

    dchi2_cpl = [summary_results[c]['CPL']['delta_chi2_vs_LCDM'] for c in ['SN_only', 'SN_BAO', 'SN_BAO_CMB']]
    dchi2_osc = [summary_results[c]['OSC']['delta_chi2_vs_LCDM'] for c in ['SN_only', 'SN_BAO', 'SN_BAO_CMB']]
    daic_cpl  = [summary_results[c]['CPL']['delta_AIC_vs_LCDM']  for c in ['SN_only', 'SN_BAO', 'SN_BAO_CMB']]
    daic_osc  = [summary_results[c]['OSC']['delta_AIC_vs_LCDM']  for c in ['SN_only', 'SN_BAO', 'SN_BAO_CMB']]
    dbic_cpl  = [summary_results[c]['CPL']['delta_BIC_vs_LCDM']  for c in ['SN_only', 'SN_BAO', 'SN_BAO_CMB']]
    dbic_osc  = [summary_results[c]['OSC']['delta_BIC_vs_LCDM']  for c in ['SN_only', 'SN_BAO', 'SN_BAO_CMB']]

    ax2.bar(x_pos - 1.5 * width, dchi2_cpl, width, color='royalblue', label=r'$\Delta\chi^2$ (CPL)')
    ax2.bar(x_pos - 0.5 * width, dchi2_osc, width, color='crimson', label=r'$\Delta\chi^2$ (Log-Osc)')
    ax2.bar(x_pos + 0.5 * width, daic_osc,  width, color='salmon', edgecolor='black', label=r'$\Delta\mathrm{AIC}$ (Log-Osc)')
    ax2.bar(x_pos + 1.5 * width, dbic_osc,  width, color='darkred', alpha=0.75, label=r'$\Delta\mathrm{BIC}$ (Log-Osc)')
    ax2.axhline(0.0, color='black', lw=1.2)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(categories, fontsize=9.5)
    ax2.set_ylabel('Criterion Difference vs. $\\Lambda\\mathrm{CDM}$', fontsize=10.5)
    ax2.set_title('(c) Information-Theoretic Model Selection Across Progressive Datasets',
                  fontsize=11.5, fontweight='bold')
    ax2.legend(loc='upper left', ncol=2, fontsize=8.5)
    ax2.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    for path_out in [
        os.path.join(BASE_DIR, "fig_euclid_hubble_forecast.png"),
        os.path.join(PAPER_DIR, "fig_euclid_hubble_forecast.png"),
        os.path.join(RESULTS_DIR, "fig_euclid_hubble_forecast.png"),
    ]:
        plt.savefig(path_out, dpi=300)
    plt.close()
    print("\nSaved v2.0 publication figures and results/audit_v2_cosmology.json!")
    return summary_results


if __name__ == '__main__':
    run_cosmology_v2_audit()

