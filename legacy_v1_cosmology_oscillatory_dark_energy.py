# -*- coding: utf-8 -*-
"""
LEGACY v1 SCRIPT (ARCHIVED FOR TRANSPARENCY — DO NOT USE FOR SCIENTIFIC INFERENCE).

Known methodological flaws in v1 (fixed in v2 `run_full_analysis.py` and `src/likelihoods.py`):
1. Used diagonal-only uncertainties `MU_SH0ES_ERR_DIAG` instead of the full 1701x1701
   `Pantheon+SH0ES_STAT+SYS.cov` covariance matrix, yielding an unphysical chi2/dof = 0.49
   and biasing flat LCDM to Om = 0.413 (vs. Brout et al. 2022's true Om = 0.334 +/- 0.018).
2. Fitted SNe Ia in isolation without BAO or CMB geometric anchors; consequently, the
   L-BFGS-B optimizer hit the hard box boundaries (`wa = -2.0`, `A_bulk = 0.9`, `lambda = 3.0`),
   meaning the v1 high-redshift turnaround forecast was an artifact of the box bounds.
"""
# See `run_full_analysis.py` for the corrected v2.0 pipeline.

