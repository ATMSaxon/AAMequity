"""Robustness of the fare-subsidy results, per metro.

On the full candidate network, with tract-specific value of time, compares
  no subsidy,
  a means-tested subsidy (low-income tracts pay a reduced fare), and
  a universal fare cut of the same percentage,
by accessibility gain, gain of the low-income group and income concentration.
Reported with tract-bootstrap intervals for the changes in concentration, for
several subsidy shares and fare tiers, and under one-at-a-time changes to the
accessibility model and the calibrated ground times.

Usage:  python scripts/17_subsidy_robustness.py <metro>
Output: results/revision/subsidy_<metro>.json
"""
from __future__ import annotations

import dataclasses
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

MAX_CAND = 50
N_BOOT = 1000
SHARES = (0.25, 0.5, 0.75)
TIERS = ("public_service", "long_run", "near_term")
MODEL_VARIANTS = {
    "baseline": {},
    "terminal_-3min": {"cfg": {"terminal_time_shift_min": -3.0}},
    "terminal_+3min": {"cfg": {"terminal_time_shift_min": 3.0}},
    "congestion_-20%": {"cfg": {"congestion_multiplier_scale": 0.8}},
    "congestion_+20%": {"cfg": {"congestion_multiplier_scale": 1.2}},
    "security_4min": {"aam": {"t_security_min": 4.0}},
    "security_12min": {"aam": {"t_security_min": 12.0}},
    "wait_3min": {"aam": {"passenger_wait_min": 3.0}},
    "wait_6min": {"aam": {"passenger_wait_min": 6.0}},
    "vot_share_0.35": {"aam": {"vot_income_share": 0.35}},
    "vot_share_1.0": {"aam": {"vot_income_share": 1.0}},
}


def main():
    metro = sys.argv[1]
    assert CFG.ground_time_source == "network"
    rng = np.random.default_rng(0)
    tracts = load_tracts_joined(metro)
    wac = load_lodes_wac(metro)
    cand = build_vertiport_candidates(metro=metro, max_candidates=MAX_CAND)
    all_idx = np.arange(len(cand))
    pop = tracts["pop"].fillna(0).values.astype(float)
    inc = tracts["median_hh_income"].astype(float)
    income = inc.fillna(inc.median()).values
    low = (assign_groups(metro).set_index("tract_geoid")["income_group"]
           .reindex(tracts["tract_geoid"].values).values == "low_income")

    def evaluate(tier, share, params):
        """Gains without subsidy, with a means-tested subsidy and with a universal cut."""
        base, jobs = compute_base_cost_matrix(tracts, wac, metro)
        a0 = gravity_accessibility(base, jobs)
        weight = minutes_per_dollar_by_tract(tracts, params)
        out = {}
        for label, discount in (("none", 0.0), ("means_tested", share * low), ("universal", share)):
            fare = FareSpec(base=params.fare_base, per_km=params.fare_per_km[tier], weight=weight * (1.0 - discount))
            aam = compute_aam_cost_matrix(tracts, cand, all_idx, metro, params, fare)
            out[label] = np.maximum(gravity_accessibility(np.minimum(base, aam), jobs) - a0, 0.0)
        return out, a0

    def describe(gains, a0):
        rows = {}
        for label, dA in gains.items():
            rows[label] = {
                "mean_dA_pct_of_base": 100 * float(np.average(dA, weights=pop) / np.average(a0, weights=pop)),
                "low_income_dA_pct_of_base": 100 * float(np.average(dA[low], weights=pop[low])
                                                         / np.average(a0[low], weights=pop[low])),
                "ci_dA_vs_income": concentration_index(dA, income, pop),
            }
        return rows

    out = {"metro": metro, "tiers": {}, "model_variants": {}}
    for tier in TIERS:
        out["tiers"][tier] = {}
        for share in SHARES:
            gains, a0 = evaluate(tier, share, CFG.aam)
            row = describe(gains, a0)
            if tier == "public_service" and share == CFG.aam.subsidy_share:
                n = len(pop)
                diff = {"means_tested": [], "universal": []}
                for _ in range(N_BOOT):
                    idx = rng.integers(0, n, n)
                    ref = concentration_index(gains["none"][idx], income[idx], pop[idx])
                    for k in diff:
                        diff[k].append(concentration_index(gains[k][idx], income[idx], pop[idx]) - ref)
                row["change_in_ci_interval"] = {
                    k: {"estimate": row[k]["ci_dA_vs_income"] - row["none"]["ci_dA_vs_income"],
                        "low": float(np.percentile(v, 2.5)), "high": float(np.percentile(v, 97.5))}
                    for k, v in diff.items()}
            out["tiers"][tier][str(share)] = row
            print(f"[{metro}] {tier} share={share}: " + " | ".join(
                f"{k}: {v['mean_dA_pct_of_base']:.3f}% low {v['low_income_dA_pct_of_base']:.3f}% CI {v['ci_dA_vs_income']:+.3f}"
                for k, v in row.items() if k != "change_in_ci_interval"), flush=True)

    for label, v in MODEL_VARIANTS.items():
        for key, val in v.get("cfg", {}).items():
            setattr(CFG, key, val)
        params = dataclasses.replace(CFG.aam, **v.get("aam", {}))
        gains, a0 = evaluate("public_service", CFG.aam.subsidy_share, params)
        out["model_variants"][label] = describe(gains, a0)
        CFG.congestion_multiplier_scale, CFG.terminal_time_shift_min = 1.0, 0.0
        r = out["model_variants"][label]
        print(f"[{metro}] variant {label}: " + " | ".join(
            f"{k}: {x['mean_dA_pct_of_base']:.3f}%/{x['ci_dA_vs_income']:+.3f}" for k, x in r.items()), flush=True)

    out_dir = RESULTS / "revision"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"subsidy_{metro}.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"{metro}: done", flush=True)


if __name__ == "__main__":
    main()
