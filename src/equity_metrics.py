"""Equity / inequality metrics — full set per RESEARCH_BRIEF §8.5.

All metrics accept a per-unit value vector x and an optional weight (population),
and an optional grouping vector for group-conditioned variants.

Implemented:
    gini(x, w)
    theil(x, w)
    theil_decomposition(x, w, group)        # within / between
    palma(x, w)                              # top-10% / bottom-40%
    concentration_index(x, w, rank_var)     # CI relative to socioeconomic ranking
    exposure_disparity_ratio(x, group)      # group-mean / overall-mean
    minimum_gain(dA, group)                  # min over groups of mean gain
    benefit_burden_ratio(dA, B)              # tract-level EBR + group means
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import rankdata


def _ensure_weights(x, w):
    x = np.asarray(x, dtype=float)
    if w is None:
        w = np.ones_like(x)
    else:
        w = np.asarray(w, dtype=float)
    mask = np.isfinite(x) & np.isfinite(w) & (w > 0)
    return x[mask], w[mask]


# ----------------------------------------------------------------------------
# Gini
# ----------------------------------------------------------------------------
def gini(x, w=None) -> float:
    """Population-weighted Gini coefficient of x."""
    x, w = _ensure_weights(x, w)
    if len(x) == 0:
        return float("nan")
    order = np.argsort(x)
    x = x[order]; w = w[order]
    cum_w = np.cumsum(w)
    cum_xw = np.cumsum(x * w)
    if cum_xw[-1] == 0:
        return 0.0
    L = cum_xw / cum_xw[-1]
    F = cum_w / cum_w[-1]
    return float(1.0 - 2.0 * np.trapezoid(L, F))


# ----------------------------------------------------------------------------
# Theil L (= MLD) — natural for accessibility (ratio scale)
# ----------------------------------------------------------------------------
def theil(x, w=None) -> float:
    """Theil L (mean log deviation) — pop-weighted."""
    x, w = _ensure_weights(x, w)
    if len(x) == 0:
        return float("nan")
    x = np.where(x <= 0, 1e-9, x)
    mu = np.average(x, weights=w)
    return float(np.average(np.log(mu / x), weights=w))


def theil_decomposition(x, group, w=None) -> dict:
    """Within / between Theil L decomposition by group label."""
    x, w = _ensure_weights(x, w)
    g = np.asarray(group)[: len(x)]
    if len(x) == 0:
        return {"total": float("nan"), "within": float("nan"), "between": float("nan")}
    mu = np.average(x, weights=w)
    total = theil(x, w)
    within = 0.0
    between = 0.0
    for gv in pd.unique(g):
        m = (g == gv)
        if not m.any():
            continue
        xg = x[m]; wg = w[m]
        if wg.sum() == 0 or xg.sum() == 0:
            continue
        wt = wg.sum() / w.sum()
        mug = np.average(xg, weights=wg)
        within += wt * theil(xg, wg)
        between += wt * np.log(mu / max(mug, 1e-9))
    return {"total": float(total), "within": float(within), "between": float(between)}


# ----------------------------------------------------------------------------
# Palma ratio (top 10% / bottom 40%)
# ----------------------------------------------------------------------------
def palma(x, w=None) -> float:
    x, w = _ensure_weights(x, w)
    if len(x) == 0:
        return float("nan")
    order = np.argsort(x)
    x = x[order]; w = w[order]
    cum_w = np.cumsum(w) / w.sum()
    top10 = (x * w)[cum_w >= 0.9].sum()
    bot40 = (x * w)[cum_w <= 0.4].sum()
    if bot40 <= 0:
        return float("inf")
    return float(top10 / bot40)


# ----------------------------------------------------------------------------
# Concentration index — wrt a socioeconomic ranking (e.g., income)
# Wagstaff form: 2 * cov(x, rank(s)/N) / mean(x)
# Range: -1..+1. Positive = pro-rich; negative = pro-poor.
# ----------------------------------------------------------------------------
def concentration_index(x, rank_var, w=None) -> float:
    x = np.asarray(x, dtype=float)
    s = np.asarray(rank_var, dtype=float)
    if w is None:
        w = np.ones_like(x)
    w = np.asarray(w, dtype=float)
    mask = np.isfinite(x) & np.isfinite(s) & np.isfinite(w) & (w > 0)
    x, s, w = x[mask], s[mask], w[mask]
    if len(x) < 3:
        return float("nan")
    # Weighted rank of s
    order = np.argsort(s)
    cum_w = np.cumsum(w[order])
    rank = np.empty_like(cum_w)
    rank[order] = (cum_w - 0.5 * w[order]) / w.sum()
    mu = np.average(x, weights=w)
    if mu == 0:
        return 0.0
    cov = np.average((x - mu) * (rank - 0.5), weights=w)
    return float(2.0 * cov / mu)


# ----------------------------------------------------------------------------
# Exposure disparity ratio (group / overall)
# ----------------------------------------------------------------------------
def exposure_disparity_ratio(x, group, w=None) -> dict:
    x, w = _ensure_weights(x, w)
    g = np.asarray(group)[: len(x)]
    overall = np.average(x, weights=w)
    out = {}
    for gv in pd.unique(g):
        m = (g == gv)
        if not m.any():
            continue
        out[str(gv)] = float(np.average(x[m], weights=w[m]) / max(overall, 1e-9))
    return out


# ----------------------------------------------------------------------------
# Minimum gain across groups (sufficientarian floor)
# ----------------------------------------------------------------------------
def minimum_gain(dA, group, w=None) -> dict:
    x, w = _ensure_weights(dA, w)
    g = np.asarray(group)[: len(x)]
    means = {}
    for gv in pd.unique(g):
        m = (g == gv)
        if not m.any():
            continue
        means[str(gv)] = float(np.average(x[m], weights=w[m]))
    if not means:
        return {"min_group": None, "min_value": float("nan"), "by_group": {}}
    min_g = min(means, key=means.get)
    return {"min_group": min_g, "min_value": means[min_g], "by_group": means}


# ----------------------------------------------------------------------------
# Benefit-burden ratio EBR
# ----------------------------------------------------------------------------
def benefit_burden_ratio(dA, B, eps: float = 1e-3) -> np.ndarray:
    dA = np.asarray(dA, dtype=float)
    B = np.asarray(B, dtype=float)
    return dA / (B + eps)


def ebr_by_group(dA, B, group, w=None, eps: float = 1e-3) -> dict:
    ebr = benefit_burden_ratio(dA, B, eps)
    return {k: v for k, v in exposure_disparity_ratio(ebr, group, w).items()}


# ----------------------------------------------------------------------------
# Convenience: full equity panel for a given scenario result
# ----------------------------------------------------------------------------
def bootstrap_metric(metric_fn, *args, n_boot: int = 200, seed: int = 0,
                       sample_frac: float = 1.0) -> tuple[float, float, float]:
    """Bootstrap (median, low, high) 95% CI for any metric using resampled tracts."""
    arrs = [np.asarray(a) for a in args]
    n = len(arrs[0])
    rng = np.random.default_rng(seed)
    boot_size = max(int(n * sample_frac), 50)
    samples = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=boot_size)
        try:
            v = metric_fn(*[a[idx] for a in arrs])
            if isinstance(v, dict):
                # min_gain returns dict — extract min_value
                v = v.get("min_value", float("nan"))
            if np.isfinite(v):
                samples.append(float(v))
        except Exception:
            continue
    if not samples:
        return (float("nan"), float("nan"), float("nan"))
    arr = np.asarray(samples)
    return (float(np.median(arr)),
            float(np.percentile(arr, 2.5)),
            float(np.percentile(arr, 97.5)))


def equity_panel_with_ci(df: pd.DataFrame, *, n_boot: int = 200, seed: int = 0) -> dict:
    """Bootstrap-augmented equity panel — returns (point, low, high) for each metric."""
    out = {}
    out["gini_dA"]  = bootstrap_metric(gini, df["dA"].values, df["pop"].values,
                                         n_boot=n_boot, seed=seed)
    out["theil_dA"] = bootstrap_metric(theil, df["dA"].values, df["pop"].values,
                                         n_boot=n_boot, seed=seed)
    out["palma_dA"] = bootstrap_metric(palma, df["dA"].values, df["pop"].values,
                                         n_boot=n_boot, seed=seed)
    if "median_hh_income" in df.columns:
        out["ci_dA_vs_income"] = bootstrap_metric(
            concentration_index, df["dA"].values,
            df["median_hh_income"].values, df["pop"].values,
            n_boot=n_boot, seed=seed)
        out["ci_B_vs_income"] = bootstrap_metric(
            concentration_index, df["B"].values,
            df["median_hh_income"].values, df["pop"].values,
            n_boot=n_boot, seed=seed)
    return out


def compute_full_equity_panel(scenario_df: pd.DataFrame, tracts_df: pd.DataFrame,
                               group_var: str = "income_group") -> dict:
    """All headline metrics + group-stratified breakdowns.

    scenario_df  : must contain columns tract_geoid, dA, B
    tracts_df    : must contain columns tract_geoid, pop, group_var, median_hh_income
    """
    df = scenario_df.merge(tracts_df, on="tract_geoid", how="inner").dropna(subset=["dA", "B", "pop"])
    if df.empty:
        return {}
    out = {
        "gini_dA":     gini(df["dA"], df["pop"]),
        "theil_dA":    theil(df["dA"], df["pop"]),
        "palma_dA":    palma(df["dA"], df["pop"]),
        "ci_dA_vs_income": concentration_index(df["dA"], df["median_hh_income"], df["pop"]),
        "ci_B_vs_income":  concentration_index(df["B"], df["median_hh_income"], df["pop"]),
        "edr_dA":          exposure_disparity_ratio(df["dA"], df[group_var], df["pop"]),
        "edr_B":           exposure_disparity_ratio(df["B"], df[group_var], df["pop"]),
        "min_gain":        minimum_gain(df["dA"], df[group_var], df["pop"]),
        "ebr_by_group":    ebr_by_group(df["dA"], df["B"], df[group_var], df["pop"]),
        "theil_decomp":    theil_decomposition(df["dA"], df[group_var], df["pop"]),
        "mean_dA":         float(np.average(df["dA"], weights=df["pop"])),
        "mean_B":          float(np.average(df["B"], weights=df["pop"])),
        "n_tracts":        int(len(df)),
    }
    return out
