# Logarithmic Oscillatory Dark Energy & Euclid Turnaround Forecast

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Cosmology: DESI & Euclid](https://img.shields.io/badge/Target-Euclid_ESA-purple.svg)]()

Empirical fit of a **Logarithmic Oscillatory Dark Energy** equation of state against 1,590 Type Ia Supernovae from the **Pantheon+** catalog, with exact falsifiable forecasts for the **ESA Euclid Space Telescope** ($z = 0.5$ to $3.0$).

---

## 1. Theoretical Model & Exact Analytical Solution

Standard linear CPL dark energy parameterization ($w(a) = w_0 + w_a(1-a)$) predicts an unbounded phantom descent into the deep past ($z > 1.5$). We formulate a bounded, harmonic parameterization:
$$w(a) = -1 + A_{\mathrm{bulk}} \sin\left(\frac{2\pi}{\lambda_{\mathrm{bulk}}} \ln(a) + \phi_0\right)$$

Integrating the cosmological continuity equation $\dot{\rho} + 3H(1+w)\rho = 0$ yields the **exact closed-form dark energy density**:
$$\frac{\rho_{\mathrm{DE}}(z)}{\rho_0} = \exp\left( -\frac{3 A_{\mathrm{bulk}} \lambda_{\mathrm{bulk}}}{2\pi} \left[ \cos(\phi_0) - \cos\left(-\frac{2\pi}{\lambda_{\mathrm{bulk}}} \ln(1+z) + \phi_0\right) \right] \right)$$

---

## 2. Empirical Validation (Pantheon+ 1,590 SNe Ia)

| Model | Free Parameters | $H_0$ ($\mathrm{km/s/Mpc}$) | $\Omega_m$ | $\chi^2$ (1590 SNe) | $\Delta\chi^2$ vs $\Lambda\mathrm{CDM}$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **$\Lambda\mathrm{CDM}$** | 2 | $71.81$ | $0.413$ | $783.45$ | Baseline |
| **DESI CPL ($w_0, w_a$)** | 4 | $71.06$ | $0.339$ | $772.29$ | $-11.16$ |
| **Logarithmic Oscillatory** | 5 | $71.08$ | $0.355$ | **$772.59$** | **$-10.86$** |

Both dynamical models substantially improve the fit over standard $\Lambda\mathrm{CDM}$ ($\Delta\chi^2 \approx -11$).

---

## 3. Falsifiable Forecasts for the ESA Euclid Space Telescope

At high redshift ($1 < z < 3$), linear CPL and the Oscillatory model diverge sharply:

```
    w(z)
      │
 -1.0 ┼─────── ΛCDM (w = -1) ────────────────────────────────
      │
 -1.4 ┼                    ╭─────── Oscillatory Model (Turnaround!)
      │                   ╱
 -1.8 ┼──────────╮       ╱
      │           ╲     ╱
 -2.2 ┼            ╲   v
      │             ╲________ Linear CPL (Monotonic Phantom Plunge)
      └─────┬───────┬───────┬───────┬───────┬───────> Redshift z
           0.5     1.0     1.5     2.0     2.5     3.0
```

| Redshift $z$ | $w(z)$ Linear CPL | $w(z)$ Oscillatory | Difference $\Delta w$ | Observable Signature |
| :---: | :---: | :---: | :---: | :--- |
| **$0.50$** | $-1.2080$ | $-1.3171$ | $-0.109$ | Early phantom phase |
| **$1.00$** | $-1.5413$ | $-1.7386$ | $-0.197$ | Continued descent |
| **$1.50$** | $-1.7413$ | $-1.8911$ | $-0.150$ | Oscillatory minimum |
| **$2.00$** | **$-1.8746$** | **$-1.8740$** | **$0.000$** | **Inflection point (Cross-over)** |
| **$2.50$** | $-1.9699$ | **$-1.7607$** | **$+0.209$** | **Oscillatory curve reverses upward** |
| **$3.00$** | $-2.0413$ | **$-1.5984$** | **$+0.443$** | **Massive separation ($\Delta w \approx +0.44$)** |

---

## Reproducibility

```bash
git clone https://github.com/your-username/oscillatory-dark-energy-euclid.git
cd oscillatory-dark-energy-euclid
pip install -r requirements.txt
python cosmology_oscillatory_dark_energy.py
```
*Generates `fig_euclid_hubble_forecast.png` plotting the Hubble diagram, $w(z)$ evolution, and $\Delta\mu(z)$ sensitivity band.*

---

## Citation

```bibtex
@article{olivaes2026oscillatory,
  author    = {Olivaes, Bruno Lu\c{c}ardo},
  title     = {Logarithmic Oscillatory Dark Energy: Empirical Fit to Pantheon+ SNe Ia and Predictive Turnaround Forecasts for the ESA Euclid Space Telescope},
  year      = {2026},
  journal   = {Preprint},
  doi       = {10.5281/zenodo.xxxxxx}
}
```

---

## License

Creative Commons Attribution 4.0 International (CC-BY-4.0).

