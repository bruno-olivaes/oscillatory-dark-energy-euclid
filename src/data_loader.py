# -*- coding: utf-8 -*-
"""
Reproducible Cosmological Data Loaders (v3.0).

Supports:
  1. Pantheon+SH0ES SNe Ia (N = 1,580 Hubble-flow sample, full 1580x1580 STAT+SYS cov)
  2. DES-SN5YR Dovekie Recalibration (N = 1,820 SNe Ia, official STAT+SYS inverse cov)
  3. DESI Year 1 BAO (arXiv:2404.03002, 7 bins / 12 observables)
  4. DESI DR2 BAO (arXiv:2503.14738, 7 bins / 13 observables from CobayaSampler/bao_data)

Eliminates hardcoded machine-specific paths via environment variables, local `data/`,
optional automated download helpers, and a legacy fallback for local backward compatibility.
"""

import os
import urllib.request
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve, block_diag

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(REPO_ROOT, "data")
LEGACY_PANTHEON_DIR = (
    r"C:\Users\bruno\OneDrive\Desktop\_Archive_Ouroboros_Antigo"
    r"\Pantheon_Data\Pantheon+_Data\4_DISTANCES_AND_COVAR"
)

PANTHEON_BASE_URL = (
    "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/"
    "Pantheon%2B_Data/4_DISTANCES_AND_COVAR"
)
DES_DOVEKIE_BASE_URL = (
    "https://raw.githubusercontent.com/des-science/DES-SN5YR/main/4_DISTANCES_COVMAT"
)
DESI_DR2_BASE_URL = (
    "https://raw.githubusercontent.com/CobayaSampler/bao_data/master/desi_bao_dr2"
)


def _rank1_project_precision(Cinv):
    """Analytically marginalize an additive distance-modulus offset M via rank-1 projection."""
    ones = np.ones(Cinv.shape[0], dtype=np.float64)
    Cinv_1 = Cinv @ ones
    denom = float(ones @ Cinv_1)
    Cinv_proj = np.ascontiguousarray(Cinv - np.outer(Cinv_1, Cinv_1) / denom)
    return Cinv_proj, denom


def resolve_pantheon_dir(pantheon_dir=None):
    """Resolve the Pantheon+ directory via arg -> env -> repo data/ -> legacy fallback."""
    candidates = [
        pantheon_dir,
        os.environ.get("PANTHEON_DIR"),
        os.path.join(DATA_DIR, "pantheon_plus"),
        LEGACY_PANTHEON_DIR,
    ]
    for cand in candidates:
        if cand and os.path.isfile(os.path.join(cand, "Pantheon+SH0ES.dat")):
            return cand
    return os.path.join(DATA_DIR, "pantheon_plus")


def download_pantheon_plus(target_dir=None):
    """Download official Pantheon+SH0ES data and STAT+SYS covariance matrix from GitHub."""
    dest = target_dir or os.path.join(DATA_DIR, "pantheon_plus")
    os.makedirs(dest, exist_ok=True)
    for fname in ("Pantheon+SH0ES.dat", "Pantheon+SH0ES_STAT+SYS.cov"):
        fpath = os.path.join(dest, fname)
        if not os.path.isfile(fpath):
            urllib.request.urlretrieve(f"{PANTHEON_BASE_URL}/{fname}", fpath)
    return dest


def load_pantheon_plus(pantheon_dir=None):
    """Load the uncalibrated Pantheon+ Hubble-flow cosmology sample (N = 1,580)."""
    pdir = resolve_pantheon_dir(pantheon_dir)
    dat_path = os.path.join(pdir, "Pantheon+SH0ES.dat")
    cov_path = os.path.join(pdir, "Pantheon+SH0ES_STAT+SYS.cov")
    if not os.path.isfile(dat_path) or not os.path.isfile(cov_path):
        raise FileNotFoundError(
            f"Pantheon+ data not found in '{pdir}'. Set PANTHEON_DIR or call "
            "src.data_loader.download_pantheon_plus()."
        )

    df = pd.read_csv(dat_path, sep=r"\s+")
    with open(cov_path, "r", encoding="utf-8") as f:
        n_tot = int(f.readline().strip())
        cov_full = np.fromfile(f, dtype=np.float64, count=n_tot * n_tot, sep=" ").reshape(
            n_tot, n_tot
        )

    mask = (df["zHD"].values > 0.01) & (df["IS_CALIBRATOR"].values == 0)
    idx_sn = np.where(mask)[0]
    cov_sn = cov_full[np.ix_(idx_sn, idx_sn)]
    cho_sn = cho_factor(cov_sn, lower=True, check_finite=False)
    Cinv = cho_solve(cho_sn, np.eye(len(idx_sn), dtype=np.float64), check_finite=False)
    Cinv_proj, e_sn = _rank1_project_precision(Cinv)

    return {
        "name": "pantheon_plus",
        "n_sn": int(len(idx_sn)),
        "zhd": df["zHD"].values[idx_sn].astype(np.float64),
        "zhel": df["zHEL"].values[idx_sn].astype(np.float64),
        "mb": df["m_b_corr"].values[idx_sn].astype(np.float64),
        "cov": cov_sn,
        "inv_cov": Cinv,
        "Cinv_proj": Cinv_proj,
        "e_sn": e_sn,
    }


def resolve_des_dir(des_dir=None):
    """Resolve the DES-SN5YR Dovekie directory via arg -> env -> repo data/des_sn5yr."""
    candidates = [
        des_dir,
        os.environ.get("DES_SN5YR_DIR"),
        os.path.join(DATA_DIR, "des_sn5yr"),
    ]
    for cand in candidates:
        if cand and os.path.isfile(os.path.join(cand, "DES-Dovekie_HD.csv")):
            return cand
    return os.path.join(DATA_DIR, "des_sn5yr")


def load_des_sn5yr(des_dir=None):
    """
    Load the official DES-SN5YR Dovekie recalibrated supernova dataset (N = 1,820)
    and upper-triangular packed STAT+SYS inverse covariance matrix.
    """
    ddir = resolve_des_dir(des_dir)
    hd_path = os.path.join(ddir, "DES-Dovekie_HD.csv")
    cov_path = os.path.join(ddir, "STAT+SYS.npz")
    if not os.path.isfile(hd_path) or not os.path.isfile(cov_path):
        raise FileNotFoundError(
            f"DES-SN5YR Dovekie data not found in '{ddir}'. Set DES_SN5YR_DIR or place "
            "DES-Dovekie_HD.csv and STAT+SYS.npz in data/des_sn5yr/."
        )

    # Parse SNANA ASCII Hubble Diagram (# comments, VARNAMES: header, SN: data rows)
    with open(hd_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]

    header_tokens = lines[0].split()
    cols = header_tokens[1:] if header_tokens[0].startswith("VARNAMES") else header_tokens
    rows = []
    for ln in lines[1:]:
        parts = ln.split()
        rows.append(parts[1:] if parts[0].startswith("SN") else parts)
    df = pd.DataFrame(rows, columns=cols)
    for col in ("zHD", "zHEL", "MU", "MUERR"):
        df[col] = df[col].astype(np.float64)

    # Unpack inverse covariance strictly per official DES-Dovekie-SN_Likelihood.py
    d = np.load(cov_path)
    n = int(d[d.files[0]][0])
    if n != len(df):
        raise ValueError(f"DES-Dovekie length mismatch: HD={len(df)}, npz nsn={n}")
    inv_cov = np.zeros((n, n), dtype=np.float64)
    inv_cov[np.triu_indices(n)] = d[d.files[1]].astype(np.float64)
    i_lower = np.tril_indices(n, -1)
    inv_cov[i_lower] = inv_cov.T[i_lower]

    ww = df["zHD"].values > 0.0
    if not np.all(ww):
        cov_full = np.linalg.inv(inv_cov)
        cov_sub = cov_full[np.ix_(ww, ww)]
        inv_cov = np.linalg.inv(cov_sub)
        df = df.loc[ww].reset_index(drop=True)
        cov_sn = cov_sub
    else:
        cov_sn = None

    Cinv_proj, e_sn = _rank1_project_precision(inv_cov)

    return {
        "name": "des_sn5yr",
        "n_sn": int(len(df)),
        "zhd": df["zHD"].values.astype(np.float64),
        "zhel": df["zHEL"].values.astype(np.float64),
        "mb": df["MU"].values.astype(np.float64),
        "cov": cov_sn,
        "inv_cov": inv_cov,
        "Cinv_proj": Cinv_proj,
        "e_sn": e_sn,
    }


def load_desi_bao(release="dr2", data_dir=None):
    """
    Load DESI BAO measurements and exact covariance matrix.

    Returns:
      specs: list of (z_eff, quantity_str) aligned row-by-row with y_bao and C_bao
      y_bao: 1D numpy float64 array of observed BAO ratios
      C_bao: 2D numpy float64 covariance matrix
      Cinv_bao: 2D numpy float64 inverse covariance matrix
    """
    rel = str(release).lower().strip()
    if rel in ("dr1", "y1", "desi_dr1", "2024"):
        # DESI 2024 VI (arXiv:2404.03002, JCAP 02 (2025) 021, Table 1)
        desi_y1_spec = [
            (0.295, "DV", 7.93, 0.15, 0.0, 0.0, 0.0),
            (0.510, "DM_DH", 13.62, 0.25, 20.98, 0.61, -0.445),
            (0.706, "DM_DH", 16.85, 0.32, 20.08, 0.60, -0.420),
            (0.930, "DM_DH", 21.71, 0.28, 17.88, 0.35, -0.389),
            (1.317, "DM_DH", 27.79, 0.69, 13.82, 0.42, -0.444),
            (1.491, "DV", 26.07, 0.67, 0.0, 0.0, 0.0),
            (2.330, "DM_DH", 39.71, 0.94, 8.52, 0.17, -0.477),
        ]
        specs = []
        y_b = []
        blocks = []
        for z_b, kind, v1, s1, v2, s2, r_c in desi_y1_spec:
            if kind == "DV":
                specs.append((float(z_b), "DV_over_rs"))
                y_b.append(v1)
                blocks.append(np.array([[s1 ** 2]], dtype=np.float64))
            else:
                specs.append((float(z_b), "DM_over_rs"))
                specs.append((float(z_b), "DH_over_rs"))
                y_b.extend([v1, v2])
                blocks.append(
                    np.array([[s1 ** 2, r_c * s1 * s2], [r_c * s1 * s2, s2 ** 2]], dtype=np.float64)
                )
        y_bao = np.array(y_b, dtype=np.float64)
        C_bao = block_diag(*blocks)
        Cinv_bao = np.linalg.inv(C_bao)
        return specs, y_bao, C_bao, Cinv_bao

    if rel in ("dr2", "desi_dr2", "2025"):
        bdir = data_dir or os.path.join(DATA_DIR, "desi_dr2")
        mean_path = os.path.join(bdir, "desi_gaussian_bao_ALL_GCcomb_mean.txt")
        cov_path = os.path.join(bdir, "desi_gaussian_bao_ALL_GCcomb_cov.txt")
        if not os.path.isfile(mean_path) or not os.path.isfile(cov_path):
            raise FileNotFoundError(
                f"DESI DR2 files not found in '{bdir}'. Expected CobayaSampler/bao_data/desi_bao_dr2."
            )

        specs = []
        y_b = []
        with open(mean_path, "r", encoding="utf-8") as f:
            for ln in f:
                s = ln.strip()
                if not s or s.startswith("#"):
                    continue
                parts = s.split()
                specs.append((float(parts[0]), parts[2]))
                y_b.append(float(parts[1]))
        y_bao = np.array(y_b, dtype=np.float64)
        C_bao = np.loadtxt(cov_path, dtype=np.float64)
        if C_bao.shape != (len(y_bao), len(y_bao)):
            raise ValueError(f"DESI DR2 covariance shape {C_bao.shape} != {len(y_bao)}")
        Cinv_bao = np.linalg.inv(C_bao)
        return specs, y_bao, C_bao, Cinv_bao

    raise ValueError(f"Unknown DESI BAO release '{release}'. Use 'dr1' or 'dr2'.")
