# Logarithmic Oscillatory Dark Energy: Full-Covariance Pantheon+, DESI BAO & Planck CMB Audit (v2.0)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24121691.svg)](https://doi.org/10.5281/zenodo.24121691)
[![Paper PDF](https://img.shields.io/badge/Paper-PDF_Manuscript-red.svg)](manuscript/main.pdf)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-green.svg)](https://www.python.org/)
[![Release: v2.0 Audited](https://img.shields.io/badge/Release-v2.0_Full_Covariance_&_MCMC-brightgreen.svg)]()

An observational cosmology pipeline testing a bounded, scale-invariant **Logarithmic Oscillatory Dark Energy** equation of state against flat $\Lambda\mathrm{CDM}$ and the linear **Chevallier–Polarski–Linder (CPL)** parameterization using the **full Pantheon+ $1580\times 1580$ systematic covariance matrix**, **DESI Year 1 BAO**, and **Planck 2018 CMB** shift priors.

---

## What Changed in Version 2.0 (Post-Publication Audit)

1. **Full $1580\times 1580$ Pantheon+ Covariance Matrix (Rank-1 Cholesky Marginalization):** Replaced the diagonal-only `MU_SH0ES_ERR_DIAG` approximation of v1 (which produced an unphysical $\chi^2/\mathrm{dof} = 0.49$, biased flat $\Lambda\mathrm{CDM}$ to $\Omega_m = 0.413$, and yielded a spurious $\Delta\chi^2 \approx -11$ from fitting correlated survey calibration offsets) with the exact `Pantheon+SH0ES_STAT+SYS.cov` covariance matrix over the uncalibrated Hubble-flow sample (`zHD > 0.01`, `IS_CALIBRATOR == 0`, $N = 1,580$), analytically marginalized over the nuisance offset $\mathcal{M}$. This recovers **$\Omega_m = 0.3325 \pm 0.0182$** ($\chi^2_{\min} = 1386.41$, $\chi^2/\mathrm{dof} = 0.879$), in exact agreement with Brout et al. (2022, $\Omega_m = 0.334 \pm 0.018$), and proves that Pantheon+ SNe Ia *alone* give $\Delta\chi^2 = -0.49$ (no preference over $\Lambda\mathrm{CDM}$).
2. **Joint Multi-Probe Likelihood (Pantheon+ + DESI 2024 VI BAO + Planck 2018 CMB):** Combines Pantheon+ with all 12 BAO observables across 7 redshift bins ($0.295 \le z \le 2.330$, including $(D_M/r_d, D_H/r_d)$ covariances) from DESI 2024 VI (arXiv:2404.03002) and the Planck 2018 CMB shift parameter $R(z_* = 1089.92) = 1.7502 \pm 0.0046$ (Chen, Huang & Wang 2019). All models now converge to an **interior minimum** (`at_edge = False`), eliminating the v1 optimizer box-boundary artifacts (`wa = -2.0`, `A = 0.9`, `lambda = 3.0`).
3. **Affine-Invariant Ensemble MCMC (`emcee`):** Computes full $68\%$ posterior credible intervals and propagates posterior predictive bands for $w(z)$ and $\Delta H(z)/H_{\Lambda\mathrm{CDM}}(z)$ into the ESA Euclid spectroscopic BAO window ($0.9 < z < 1.8$, $\sigma_H/H \approx 1.3\%$, Blanchard et al. 2020).

---

## Audited Multi-Probe Results (v2.0)

| Dataset Combination | Model | $\Delta k$ | Best-Fit Parameters | At Prior Edge? | $\chi^2_{\min}$ | $\Delta\chi^2$ | $\Delta\mathrm{AIC}$ | $\Delta\mathrm{BIC}$ |
| :--- | :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **Pantheon+ Only** ($N=1580$) | $\Lambda\mathrm{CDM}$ | $0$ | $\Omega_m = 0.3325 \pm 0.0182$ | No | $1386.41$ | $0.00$ | $0.00$ | $0.00$ |
| **Pantheon+ Only** ($N=1580$) | DESI CPL | $+2$ | $\Omega_m = 0.2369, w_0 = -0.852, w_a = +0.361$ | No | $1385.94$ | $-0.47$ | $+3.53$ | $+14.26$ |
| **Pantheon+ Only** ($N=1580$) | Log-Oscillatory | $+3$ | $\Omega_m = 0.2743, A = 0.165, \lambda = 1.87, \phi_0 = 2.61$ | No | $1385.92$ | $-0.49$ | $+5.51$ | $+21.60$ |
| **Joint SN + DESI BAO + CMB** ($N=1593$) | $\Lambda\mathrm{CDM}$ | $0$ | $\Omega_m = 0.3142$ (MCMC: $0.3144^{+0.0068}_{-0.0070}$) | **No** | **$1402.04$** | **$0.00$** | **$0.00$** | **$0.00$** |
| **Joint SN + DESI BAO + CMB** ($N=1593$) | **DESI CPL** | $+2$ | $\Omega_m = 0.3134, w_0 = -0.867, w_a = -0.547$ | **No** | **$1398.30$** | **$-3.75$** | **$+0.25$** | **$+11.00$** |
| **Joint SN + DESI BAO + CMB** ($N=1593$) | **Log-Oscillatory** | $+3$ | $\Omega_m = 0.3135, A = 0.218, \lambda = 3.09, \phi_0 = 0.592$ | **No** | **$1398.15$** | **$-3.90$** | **$+2.10$** | **$+18.22$** |

---

## Reproducibility

```bash
# Run full v2.0 multi-probe optimization, emcee MCMC sampling, and figure generation
python run_full_analysis.py

# Run pytest validation suite
pytest tests
```

---

## License

MIT License.
