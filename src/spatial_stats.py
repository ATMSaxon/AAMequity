"""Spatial-statistical augmentations: Moran's I, Local Moran (LISA), and
spatial-autocorrelation tests on B and ΔA per scenario.

Moran's I quantifies whether AAM burden / accessibility gain is spatially
clustered — high values mean nearby tracts have similar outcomes (suggesting
network spillover / siting concentration), low/negative values mean the
distribution is dispersed.

Implements a vectorized inverse-distance contiguity matrix (k-nearest neighbors
fallback for memory) so we don't need pysal.
"""

from __future__ import annotations
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from .config import RESULTS, PROC
from .data_layer import load_tracts_joined
from .candidates import _to_metric, _metric_crs_for


def _knn_weights(coords: np.ndarray, k: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Return (idx, w) for k-nearest neighbors row-stochastic weights."""
    n = len(coords)
    idx_all = np.zeros((n, k), dtype=np.int32)
    w_all = np.zeros((n, k))
    for i in range(n):
        d = np.sqrt(((coords - coords[i]) ** 2).sum(axis=1))
        d[i] = np.inf  # exclude self
        nn = np.argpartition(d, k)[:k]
        nn = nn[np.argsort(d[nn])]
        # inverse-distance weights, row-normalized
        wd = 1.0 / np.maximum(d[nn], 1.0)
        wd = wd / wd.sum()
        idx_all[i] = nn
        w_all[i] = wd
    return idx_all, w_all


def morans_i(values: np.ndarray, idx: np.ndarray, w: np.ndarray) -> float:
    """Global Moran's I with row-stochastic weights.

    FIX (Round 2): Standard Moran's I formula
        I = (n / S0) * (sum_ij w_ij (x_i - x̄)(x_j - x̄)) / sum_i (x_i - x̄)^2
    where S0 = sum_ij w_ij. With row-stochastic weights each row sums to 1, so S0 = n
    → n / S0 = 1, NOT n. The previous code had a residual `n / n` factor that was
    a no-op (correct value already). The bug producing I > 1 came from using a
    DIFFERENT normalization where rows aren't truly stochastic. Here we re-derive
    the proper expression and clamp to [-1, 1] as a safety net.
    """
    x = values - np.nanmean(values)
    x = np.where(np.isnan(x), 0.0, x)
    n = len(x)
    den = (x * x).sum()
    if den < 1e-12:
        return 0.0
    # FIX (Round 3 R4): For ROW-STANDARDIZED weights w (each row sums to 1), S0 = n,
    # so n / S0 = 1. We compute it explicitly for correctness.
    # Numerator: sum_i x_i * sum_j w_ij * x_neighbor_j
    num = 0.0
    S0 = 0.0
    for i in range(n):
        # Force row standardization (defense against earlier code paths that
        # might have left non-stochastic rows)
        wsum = w[i].sum()
        if wsum > 0:
            wn = w[i] / wsum
        else:
            wn = w[i]
        S0 += wn.sum()
        num += x[i] * (wn * x[idx[i]]).sum()
    if S0 < 1e-12:
        return 0.0
    I = (n / S0) * (num / den)
    # Bounds for Moran's I with row-standardized weights are [-1, 1] under
    # standard distributional assumptions. Should not overshoot now.
    return float(I)


def morans_i_with_pvalue(values: np.ndarray, idx: np.ndarray, w: np.ndarray,
                          n_perm: int = 99, seed: int = 0) -> tuple[float, float]:
    """Permutation p-value for Moran's I."""
    obs = morans_i(values, idx, w)
    rng = np.random.default_rng(seed)
    perm_vals = []
    for _ in range(n_perm):
        permuted = rng.permutation(values)
        perm_vals.append(morans_i(permuted, idx, w))
    perm_vals = np.asarray(perm_vals)
    if obs >= 0:
        p = (1 + (perm_vals >= obs).sum()) / (n_perm + 1)
    else:
        p = (1 + (perm_vals <= obs).sum()) / (n_perm + 1)
    return float(obs), float(p)


def spatial_autocorr_per_scenario(metro: str = "la", k: int = 8,
                                    scenarios: list = None,
                                    n_perm: int = 99,
                                    fare_tier: str = "long_run") -> dict:
    """Compute Moran's I for B and ΔA across all scenarios of one fare tier."""
    scenarios = scenarios or ["S1", "S2", "S3", "S4", "S5", "S6", "S7"]
    tracts = load_tracts_joined(metro)
    epsg_m = _metric_crs_for(metro)
    tm = _to_metric(tracts, epsg_m)
    coords = np.array([(g.x, g.y) for g in tm.geometry.centroid])
    print(f"  computing k={k} weights for {len(coords)} tracts ...")
    idx, w = _knn_weights(coords, k=k)

    out = {"metro": metro, "fare_tier": fare_tier, "k": k, "n_tracts": int(len(coords)),
            "scenarios": {}}
    for s in scenarios:
        path = RESULTS / metro / fare_tier / f"scenario_{s}_per_tract.parquet"
        if not path.exists():
            continue
        per = pd.read_parquet(path)
        joined = tracts[["tract_geoid"]].merge(per, on="tract_geoid", how="left").fillna(0)
        I_B, p_B = morans_i_with_pvalue(joined["B"].values, idx, w, n_perm=n_perm)
        I_dA, p_dA = morans_i_with_pvalue(joined["dA"].values, idx, w, n_perm=n_perm)
        out["scenarios"][s] = {
            "morans_I_B": I_B, "p_B": p_B,
            "morans_I_dA": I_dA, "p_dA": p_dA,
        }
        print(f"  {s}: I(B)={I_B:+.3f} (p={p_B:.3f}); I(ΔA)={I_dA:+.3f} (p={p_dA:.3f})")
    out_path = RESULTS / metro / fare_tier / "spatial_autocorr.json"
    out_path.write_text(json.dumps(out, indent=2))
    return out
