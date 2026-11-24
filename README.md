# Logarithmic Oscillatory Dark Energy: DESI DR2, DES-SN5YR Dovekie & Pre-Registered 2027 Euclid Forecast (v3.0)

[![Zenodo DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24485004.svg)](https://doi.org/10.5281/zenodo.24485004)
[![Paper PDF](https://img.shields.io/badge/Paper-PDF_Manuscript-red.svg)](Oscillatory_Dark_Energy_Euclid_PreRegistration.pdf)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-green.svg)](https://www.python.org/)
[![Tests: 12/12](https://img.shields.io/badge/pytest-12%2F12_passed-brightgreen.svg)]()
[![Release: v3.0 Audited](https://img.shields.io/badge/Release-v3.0_DESI_DR2_&_Euclid_2027-brightgreen.svg)]()

A pre-registered observational cosmology and Stage-IV Fisher forecasting pipeline testing bounded **Logarithmic Oscillatory Dark Energy** $w(a) = -1 + A \sin(\frac{2\pi}{\lambda}\ln a + \phi_0) = -1 + A \cos(\omega\ln(1+z) + \tilde{\phi}_0)$ ($\omega = 2\pi/\lambda$, $\tilde{\phi}_0 = \pi/2 - \phi_0 \pmod{2\pi}$) against flat $\Lambda\mathrm{CDM}$ and linear **CPL** using **Pantheon+ ($N=1,580$)**, official **DES-SN5YR Dovekie ($N=1,820$)**, official **DESI DR1 ($N=12$) & DESI DR2 ($N=13$, arXiv:2503.14738)** BAO, and the **full 3D Planck 2018 $(R, l_A, \omega_b)$** distance prior.

---

## Key Upgrades in Version 3.0

1. **Full 3D Planck 2018 CMB Prior (`(R, l_A, omega_b)`):** Replaces the scalar 1D $R$ prior with the correlated $3\times 3$ Chen, Huang & Wang (2019) covariance, dynamic $z_*(\omega_b, \omega_c)$, and explicit $r_d(\Omega_m, h, \omega_b)$.
2. **Official DESI DR2 BAO & DES-SN5YR Dovekie Full Covariances:** Upgrades BAO to all 13 DESI DR2 observables across $0.295 \le z \le 2.330$ (`data/desi_dr2/`) and ingests official DES-SN5YR Dovekie SNe Ia (`data/des_sn5yr/`, $N=1,820$).
3. **Monte Carlo Look-Elsewhere Calibration ($N_{\mathrm{MC}} = 2,000$):** Quantifies effective GP-score trial factors across $\lambda \in [0.8, 12.0]$ ($N_{\mathrm{eff}} = 1.633$ on Pantheon+ + DR2, $p_{\mathrm{global}} = 0.0292$ [$2.18\sigma$]; $N_{\mathrm{eff}} = 1.976$ on DES-SN5YR + DR2, $p_{\mathrm{global}} = 0.0102$ [$2.57\sigma$]), showing that $\Delta\chi^2$ over CPL is only $+0.21\text{--}+0.62$.
4. **Slope-Decoupled Orthogonal `(C, S)` MCMC:** Employs mixed `(StretchMove, DEMove)` proposals in $(\Omega_m, h, \omega_b, \tilde{w}_1, S, \lambda)$ with uniform prior on the disk $C^2 + S^2 \le 1$ and exact Jacobian $+\ln(\lambda/2\pi)$ ($\hat{R}_{\mathrm{split}} \le 1.037$ on DES-SN5YR + DR2; $\hat{R}_{\mathrm{split}} \le 1.033$ on cosmological/amplitude params and $1.116$ along the $\lambda$-plateau on Pantheon+ + DR2).
5. **Coupled Anisotropic `(D_M/r_d, H*r_d)` Euclid 2027 Pre-Registration:** Evaluates Stage-IV $\Lambda\mathrm{CDM}$-null reaches DR1-2027 ($A_{3\sigma} = 0.167\text{--}0.206$) and Full Survey ($A_{3\sigma} = 0.069\text{--}0.086$) across $\lambda \in [2.1, 12.0]$ and publishes blind 2027 Euclid predictions for $z \in \{1.00, 1.20, 1.40, 1.65\}$, explicitly showing $\Delta\chi^2_{\mathrm{CPL\text{ vs }OSC}} = 0.0071$ ($0.08\sigma$) in $z \in [1.0, 1.65]$ BAO alone.

---

## Reproducibility

```bash
# Execute all 6 v3.0 stress tests, MCMC chains, and publication figures
python run_v3_stress_tests.py

# Run complete 12-test validation suite
pytest -v
```

---

## License

MIT License.
