"""Revision diagnostics for the first-round reviews.

Part A  Base ground-speed robustness (Reviewer 3, comment 1): re-run S1 and S2
        with the base-mode speed replaced by a common alternative or by the
        metro-specific speed implied by the OSMnx calibration.
Part B  Constraint decomposition: which of the low-income floor and the
        per-tract burden cap drives feasibility in S3/S6/S7 and the Pareto sweep.

Usage:  python scripts/9_revision_diagnostics.py <metro>
Output: results/revision/diag_<metro>.json
"""
from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.accessibility as acc
from src.accessibility import FareSpec, compute_aam_cost_matrix, gravity_accessibility
from src.candidates import build_vertiport_candidates
from src.config import CFG, RESULTS
from src.equity_metrics import compute_full_equity_panel
from src.optimizer import solve_scenario
from src.scenarios import _build_inputs, precompute_contributions, run_scenario

import pandas as pd

CFG.ground_time_source = "euclidean"   # these diagnostics vary the constant-speed baseline
BASE_SPEED = 22.0
# Flat fares and single value-of-time tiers of the submitted manuscript
SUBMITTED_FARES = {"S1": FareSpec.flat(CFG.aam.fare_high, CFG.aam.lambda_high_income),
                   "S2": FareSpec.flat(CFG.aam.fare_mid, CFG.aam.lambda_mid_income)}
COMMON_SPEEDS = [22.0, 30.0, 40.0]
N_VP = 25
MAX_CAND = 50


def calibrated_speed(metro: str) -> float:
    """Euclidean-equivalent speed implied by the OSMnx calibration medians."""
    c = json.loads((RESULTS / metro / "network_calibration.json").read_text())
    return BASE_SPEED * c["euclid_22kmh_min_p50"] / c["network_congested_min_p50"]


def set_base_speed(speed: float) -> None:
    acc.base_travel_minutes_drive = lambda dist_km, speed_kmh=speed: dist_km / speed_kmh * 60.0


def realise(pre: dict, sel_idx: np.ndarray, metro: str) -> dict:
    n = len(pre["tracts"])
    if sel_idx.size == 0:
        return {"n_vp": 0}
    aam = compute_aam_cost_matrix(pre["tracts"], pre["candidates"], sel_idx, metro,
                                  pre["params_aam"], pre["fare"])
    eff = np.minimum(pre["base_cost"], aam)
    dA = np.maximum(gravity_accessibility(eff, pre["jobs"]) - pre["A_base"], 0.0)
    B = pre["b_ik"][:, sel_idx].sum(axis=1)
    df = pd.DataFrame({"tract_geoid": pre["tracts"]["tract_geoid"].values, "dA": dA, "B": B})
    eq = compute_full_equity_panel(df, pre["groups"])
    pop = pre["tracts"]["pop"].fillna(0).values.astype(float)
    return {
        "n_vp": int(sel_idx.size),
        "selected": [int(i) for i in sel_idx],
        "mean_dA": eq.get("mean_dA"),
        "mean_A_base": float(np.average(pre["A_base"], weights=pop)),
        "ci_dA_vs_income": eq.get("ci_dA_vs_income"),
        "ci_B_vs_income": eq.get("ci_B_vs_income"),
        "gini_dA": eq.get("gini_dA"),
        "share_tracts_dA_pos": float((dA > 0).mean()),
        "share_od_aam_cheaper": float((aam < pre["base_cost"]).sum() / (n * n)),
    }


def part_a(metro: str, candidates) -> tuple[dict, dict]:
    out = {}
    pre_mid_base = None
    speeds = [(f"common_{int(s)}", s) for s in COMMON_SPEEDS]
    speeds.append(("calibrated", calibrated_speed(metro)))
    for label, speed in speeds:
        set_base_speed(speed)
        row = {"speed_kmh": speed}
        for scen, fare in SUBMITTED_FARES.items():
            pre = precompute_contributions(metro, candidates, CFG.aam, CFG.burden, fare)
            res = run_scenario(scen, pre, n_vertiports=N_VP)
            row[scen] = {"status": res.status, **realise(pre, res.selected_idx, metro)}
            if speed == BASE_SPEED and scen == "S2":
                pre_mid_base = pre
        out[label] = row
        print(f"[A] {metro} {label} {speed:.1f} km/h: "
              + " | ".join(f"{s} dA={row[s].get('mean_dA', 0):.1f} CI={row[s].get('ci_dA_vs_income', float('nan')):+.3f}"
                           for s in ("S1", "S2")), flush=True)
    set_base_speed(BASE_SPEED)
    return out, pre_mid_base


def part_b(metro: str, pre: dict) -> dict:
    inp = _build_inputs(pre)
    g, b, pop = pre["g_ik"], pre["b_ik"], inp.pop_i
    low = np.asarray(inp.group_i) == "low_income"
    city_mean = float(np.average(g.max(axis=1), weights=pop))
    full_b = b.sum(axis=1)
    cap95 = float(np.nanpercentile(full_b, 95))

    def low_floor_lhs(sel_idx):
        return float((pop[low, None] * g[low][:, sel_idx]).sum() / pop[low].sum())

    def solve(tag, **kw):
        r = solve_scenario(tag, inp, n_vertiports=N_VP, alpha=1.0, **kw)
        d = {"status": r.status, "objective": r.objective, "selected": [int(i) for i in r.selected_idx]}
        if r.selected_idx.size:
            d["low_income_theta_achieved"] = low_floor_lhs(r.selected_idx) / city_mean
            d["max_tract_B_over_cap95"] = float(b[:, r.selected_idx].sum(axis=1).max() / cap95)
        return d

    out = {"city_mean_surrogate": city_mean, "cap95": cap95}
    out["unconstrained"] = solve("none")
    for th in (0.05, 0.10, 0.40, 1.0, 2.0, 5.0):
        out[f"floor_only_{th}"] = solve(f"floor{th}", per_group_floor={"low_income": th * city_mean})
    out["cap_only_95"] = solve("cap95", per_tract_burden_cap=cap95)
    out["floor0.10_and_cap95"] = solve("both", per_group_floor={"low_income": 0.10 * city_mean},
                                       per_tract_burden_cap=cap95)
    for pct in (97, 99, 99.5, 100):
        out[f"cap_only_{pct}"] = solve(f"cap{pct}", per_tract_burden_cap=float(np.nanpercentile(full_b, pct)))

    # Largest low-income floor any 25-site network can meet
    inp_low = dataclasses.replace(inp, g_ik=g * low[:, None])
    r = solve_scenario("maxlow", inp_low, n_vertiports=N_VP, alpha=1.0)
    out["max_attainable_theta"] = low_floor_lhs(r.selected_idx) / city_mean

    # Tracts that exceed the cap under every 25-site network
    k_min = np.sort(b, axis=1)[:, :N_VP].sum(axis=1)
    out["n_tracts_violating_cap95_for_any_network"] = int((k_min > cap95).sum())
    out["n_tracts"] = int(len(pop))

    for scen in ("S3", "S7"):
        r = run_scenario(scen, pre, n_vertiports=N_VP)
        out[scen] = {"status": r.status, "selected": [int(i) for i in r.selected_idx]}
    return out


def main():
    metro = sys.argv[1]
    candidates = build_vertiport_candidates(metro=metro, max_candidates=MAX_CAND)
    print(f"{metro}: {len(candidates)} candidates {candidates['seed_type'].value_counts().to_dict()}", flush=True)
    a, pre_mid = part_a(metro, candidates)
    b = part_b(metro, pre_mid)
    out_dir = RESULTS / "revision"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"diag_{metro}.json").write_text(
        json.dumps({"metro": metro, "n_candidates": len(candidates), "speed": a, "constraints": b},
                   indent=2, default=float))
    print(f"{metro}: done", flush=True)


if __name__ == "__main__":
    main()
