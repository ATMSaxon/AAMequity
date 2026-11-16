"""Scenario outcomes and fare thresholds under network-calibrated ground times.

Part A  S1 and S2 of the submitted manuscript (flat fares, one value-of-time tier
        per scenario) re-solved with the calibrated street-network baseline.
Part B  Flat-fare sweep on the full 49-site network: for each value-of-time tier, the
        share of tract pairs on which AAM beats driving and the resulting mean
        accessibility gain, as the fare rises from zero.

Usage:  python scripts/10_network_baseline.py <metro>
Output: results/revision/network_<metro>.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.accessibility import FareSpec, compute_aam_cost_matrix, gravity_accessibility
from src.candidates import build_vertiport_candidates
from src.config import CFG, RESULTS
from src.equity_metrics import compute_full_equity_panel, concentration_index
from src.scenarios import precompute_contributions, run_scenario

N_VP = 25
MAX_CAND = 50
FARES = [0, 5, 10, 15, 20, 30, 45, 60, 80, 100]


def realise(pre: dict, sel_idx: np.ndarray, metro: str) -> dict:
    n = len(pre["tracts"])
    aam = compute_aam_cost_matrix(pre["tracts"], pre["candidates"], sel_idx, metro,
                                  pre["params_aam"], pre["fare"])
    dA = np.maximum(gravity_accessibility(np.minimum(pre["base_cost"], aam), pre["jobs"]) - pre["A_base"], 0.0)
    B = pre["b_ik"][:, sel_idx].sum(axis=1)
    df = pd.DataFrame({"tract_geoid": pre["tracts"]["tract_geoid"].values, "dA": dA, "B": B})
    eq = compute_full_equity_panel(df, pre["groups"])
    pop = pre["tracts"]["pop"].fillna(0).values.astype(float)
    return {
        "mean_dA": eq.get("mean_dA"),
        "mean_A_base": float(np.average(pre["A_base"], weights=pop)),
        "ci_dA_vs_income": eq.get("ci_dA_vs_income"),
        "ci_B_vs_income": eq.get("ci_B_vs_income"),
        "share_tracts_dA_pos": float((dA > 0).mean()),
        "share_od_aam_cheaper": float((aam < pre["base_cost"]).sum() / (n * n)),
    }


def main():
    metro = sys.argv[1]
    assert CFG.ground_time_source == "network"
    candidates = build_vertiport_candidates(metro=metro, max_candidates=MAX_CAND)
    out = {"metro": metro, "n_candidates": len(candidates), "scenarios": {}, "fare_sweep": {}}

    pre_mid = None
    submitted = {"S1": FareSpec.flat(CFG.aam.fare_high, CFG.aam.lambda_high_income),
                 "S2": FareSpec.flat(CFG.aam.fare_mid, CFG.aam.lambda_mid_income)}
    for scen, fare in submitted.items():
        pre = precompute_contributions(metro, candidates, CFG.aam, CFG.burden, fare)
        res = run_scenario(scen, pre, n_vertiports=N_VP)
        out["scenarios"][scen] = {"status": res.status, **realise(pre, res.selected_idx, metro)}
        pre_mid = pre
        print(f"[A] {metro} {scen}: {out['scenarios'][scen]}", flush=True)

    # Full-network AAM travel time excluding fare, then add lambda * fare analytically
    tracts, base, jobs, A_base = pre_mid["tracts"], pre_mid["base_cost"], pre_mid["jobs"], pre_mid["A_base"]
    pop = tracts["pop"].fillna(0).values.astype(float)
    income = tracts["median_hh_income"].fillna(np.nanmedian(tracts["median_hh_income"])).values
    time_only = compute_aam_cost_matrix(tracts, candidates, np.arange(len(candidates)), metro,
                                        CFG.aam, FareSpec.flat(0.0, 0.0))
    finite = np.isfinite(base) & np.isfinite(time_only)
    out["ground_minutes_p50_of_eligible_pairs"] = float(np.median(base[finite]))
    out["aam_time_minutes_p50_of_eligible_pairs"] = float(np.median(time_only[finite]))
    out["share_eligible_pairs_aam_faster_door_to_door"] = float((time_only[finite] < base[finite]).mean())
    saving = base - time_only                      # minutes saved before paying the fare
    out["time_saving_minutes_percentiles_eligible"] = {
        str(q): float(np.percentile(saving[finite], q)) for q in (50, 75, 90, 95, 99, 99.9)}

    lam = {"high": CFG.aam.lambda_high_income, "mid": CFG.aam.lambda_mid_income, "low": CFG.aam.lambda_low_income}
    for tier, l in lam.items():
        rows = []
        for fare in FARES:
            cost = time_only + l * fare
            dA = np.maximum(gravity_accessibility(np.minimum(base, cost), jobs) - A_base, 0.0)
            rows.append({
                "fare": fare,
                "share_od_aam_cheaper": float((cost < base).sum() / base.size),
                "mean_dA": float(np.average(dA, weights=pop)),
                "mean_dA_pct_of_base": float(100 * np.average(dA, weights=pop) / np.average(A_base, weights=pop)),
                "share_tracts_dA_pos": float((dA > 0).mean()),
                "ci_dA_vs_income": concentration_index(dA, income, pop),
            })
        out["fare_sweep"][tier] = rows
        print(f"[B] {metro} {tier}: " + " | ".join(
            f"${r['fare']}: {r['mean_dA_pct_of_base']:.3f}%" for r in rows), flush=True)

    out_dir = RESULTS / "revision"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"network_{metro}.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"{metro}: done", flush=True)


if __name__ == "__main__":
    main()
