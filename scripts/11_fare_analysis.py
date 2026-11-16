"""Fare analysis with distance-based fares and tract-specific value of time.

Part A  Per-km fare sweep on the full candidate network: accessibility gain, its
        income concentration, and gains by income tercile as the fare rises.
Part B  All scenarios (S0-S7) at each fare tier.
Part C  Floor and burden-cap frontiers at the long-run and public-service tiers.

Usage:  python scripts/11_fare_analysis.py <metro>
Output: results/revision/fare_<metro>.json, results/<metro>/<tier>/...
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.accessibility import (
    FareSpec, compute_aam_cost_matrix, compute_base_cost_matrix, gravity_accessibility,
    minutes_per_dollar_by_tract,
)
from src.candidates import build_vertiport_candidates
from src.config import CFG, RESULTS
from src.data_layer import load_lodes_wac, load_tracts_joined
from src.equity_metrics import concentration_index
from src.groups import assign_groups
from src.pareto import run_pareto_sweep
from src.scenarios import run_all_scenarios

N_VP = 25
MAX_CAND = 50
RATES_PER_KM = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.29, 0.35, 0.45, 0.61, 0.8, 1.0, 1.5, 1.85]


def fare_sweep(metro: str, candidates) -> dict:
    tracts = load_tracts_joined(metro)
    base, jobs = compute_base_cost_matrix(tracts, load_lodes_wac(metro), metro)
    A_base = gravity_accessibility(base, jobs)
    pop = tracts["pop"].fillna(0).values.astype(float)
    income = tracts["median_hh_income"].fillna(tracts["median_hh_income"].median()).values
    group = assign_groups(metro).set_index("tract_geoid")["income_group"].reindex(
        tracts["tract_geoid"].values).fillna("mid_income").values
    weight = minutes_per_dollar_by_tract(tracts, CFG.aam)
    all_idx = np.arange(len(candidates))

    rows = []
    for rate in RATES_PER_KM:
        aam = compute_aam_cost_matrix(tracts, candidates, all_idx, metro, CFG.aam,
                                      FareSpec(base=CFG.aam.fare_base, per_km=rate, weight=weight))
        dA = np.maximum(gravity_accessibility(np.minimum(base, aam), jobs) - A_base, 0.0)
        row = {
            "fare_per_km": rate,
            "mean_dA_pct_of_base": float(100 * np.average(dA, weights=pop) / np.average(A_base, weights=pop)),
            "share_tracts_dA_pos": float((dA > 0).mean()),
            "share_od_aam_cheaper": float((aam < base).sum() / base.size),
            "ci_dA_vs_income": concentration_index(dA, income, pop),
        }
        for g in ("low_income", "mid_income", "high_income"):
            m = (group == g) & (pop > 0)
            row[f"dA_pct_{g}"] = float(100 * np.average(dA[m], weights=pop[m]) / np.average(A_base[m], weights=pop[m]))
        rows.append(row)
        print(f"[A] {metro} ${rate:.2f}/km: dA={row['mean_dA_pct_of_base']:.3f}% CI={row['ci_dA_vs_income']:+.3f} "
              f"low/mid/high={row['dA_pct_low_income']:.3f}/{row['dA_pct_mid_income']:.3f}/{row['dA_pct_high_income']:.3f}",
              flush=True)
    return {"minutes_per_dollar_levels": sorted(set(np.round(weight, 3).tolist())), "rows": rows}


def main():
    metro = sys.argv[1]
    assert CFG.ground_time_source == "network"
    candidates = build_vertiport_candidates(metro=metro, max_candidates=MAX_CAND)
    out = {"metro": metro, "fare_per_km_tiers": CFG.aam.fare_per_km, "fare_sweep": fare_sweep(metro, candidates),
           "scenarios": {}}
    for tier in CFG.aam.fare_per_km:
        summary = run_all_scenarios(metro=metro, n_vertiports=N_VP, max_candidates=MAX_CAND, fare_tier=tier)
        out["scenarios"][tier] = summary.to_dict(orient="records")
        print(f"[B] {metro} {tier}\n" + summary[["scenario", "status", "mean_dA_pct_of_base",
              "share_tracts_dA_pos", "ci_dA_vs_income", "ci_B_vs_income"]].to_string(index=False), flush=True)
    out["pareto"] = {tier: run_pareto_sweep(metro, n_vertiports=N_VP, max_candidates=MAX_CAND,
                                            fare_tier=tier).to_dict(orient="records")
                     for tier in ("long_run", "public_service")}
    out_dir = RESULTS / "revision"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"fare_{metro}.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"{metro}: done", flush=True)


if __name__ == "__main__":
    main()
