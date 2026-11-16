"""Efficiency–equity frontiers: the low-income floor and the burden cap swept separately."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CFG, RESULTS
from .accessibility import distance_fare
from .candidates import build_vertiport_candidates
from .data_layer import load_tracts_joined
from .equity_metrics import compute_full_equity_panel
from .optimizer import solve_scenario
from .scenarios import (
    _build_inputs, best_attainable_gain, group_mean_gain, precompute_contributions, realise,
)

FLOOR_SHARES = (0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0)     # of the best attainable low-income gain
CAP_SHARES = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3)   # of the unconstrained network's peak tract burden


def run_pareto_sweep_all_metros(metros: list = None,
                                n_vertiports: int = 25,
                                max_candidates: int = 50,
                                fare_tier: str = "long_run") -> pd.DataFrame:
    """Frontier sweeps across all metros, written to one cross-metro CSV."""
    metros = metros or ["la", "dfw", "sfbay", "nyc"]
    parts = []
    for m in metros:
        print(f"\n=== Pareto sweep: {m} ===")
        df = run_pareto_sweep(m, n_vertiports=n_vertiports, max_candidates=max_candidates,
                              fare_tier=fare_tier)
        df.insert(0, "metro", m)
        parts.append(df)
    full = pd.concat(parts, ignore_index=True)
    out = RESULTS / f"cross_metro_pareto_{fare_tier}.csv"
    full.to_csv(out, index=False)
    print(f"\n  -> {out}")
    return full


def run_pareto_sweep(metro: str = "la",
                     n_vertiports: int = 25,
                     max_candidates: int = 50,
                     fare_tier: str = "long_run") -> pd.DataFrame:
    """Sweep the low-income floor and the per-tract burden cap, one at a time.

    Each row reports whether the constraint is feasible, the realised mean
    accessibility gain and its income concentration, the low-income gain as a
    share of the best attainable, and the highest tract burden relative to that
    of the unconstrained network.
    """
    out_dir = RESULTS / metro / fare_tier / "pareto"
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates = build_vertiport_candidates(metro=metro, max_candidates=max_candidates)
    tracts = load_tracts_joined(metro)
    pre = precompute_contributions(metro, candidates, CFG.aam, CFG.burden,
                                   distance_fare(tracts, CFG.aam, fare_tier))
    inp = _build_inputs(pre)
    low = np.asarray(inp.group_i) == "low_income"
    best_low = best_attainable_gain(inp, low, n_vertiports)
    unconstrained = solve_scenario("unconstrained", inp, n_vertiports=n_vertiports, alpha=1.0)
    peak = float(pre["b_ik"][:, unconstrained.selected_idx].sum(axis=1).max())

    specs = [("floor", s, {"per_group_floor": {"low_income": s * best_low}} if s > 0 else {})
             for s in FLOOR_SHARES]
    specs += [("cap", c, {"per_tract_burden_cap": c * peak} if c < 1 else {}) for c in CAP_SHARES]

    rows = []
    for kind, level, kwargs in specs:
        res = solve_scenario(f"{kind}_{level}", inp, n_vertiports=n_vertiports, alpha=1.0, **kwargs)
        row = {"constraint": kind, "level": level, "status": res.status,
               "n_vertiports": int(res.selected.sum()), "objective": res.objective}
        if res.selected_idx.size:
            dA, B = realise(pre, res.selected_idx, metro)
            eq = compute_full_equity_panel(
                pd.DataFrame({"tract_geoid": tracts["tract_geoid"].values, "dA": dA, "B": B}), pre["groups"])
            row.update({
                "mean_dA": eq.get("mean_dA"), "mean_B": eq.get("mean_B"),
                "gini_dA": eq.get("gini_dA"),
                "ci_dA_vs_income": eq.get("ci_dA_vs_income"),
                "ci_B_vs_income": eq.get("ci_B_vs_income"),
                "low_income_share_of_best": group_mean_gain(inp, low, res.selected_idx) / max(best_low, 1e-12),
                "peak_burden_share_of_unconstrained": float(B.max() / peak),
            })
        rows.append(row)
        print(f"  {kind}={level}: {res.status}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "pareto_efficiency_equity.csv", index=False)
    return df
