"""Vertiport operations simulation and its effect on accessibility and noise.

Part A  Discrete-event simulation of one vertiport: passengers arrive at random,
        a flight is dispatched when the cabin is full or the first passenger has
        waited the maximum hold time, and each flight occupies one of the pads for
        its turnaround. Reports passenger wait, load factor, pad utilisation and
        daily movements as demand rises towards pad capacity.
Part B  Accessibility gain of the full candidate network when the passenger wait
        is taken from the simulation instead of the fixed-headway assumption, and
        the shift in day-night average sound level implied by simulated movements.

Usage:  python scripts/14_operations.py <metro>
Output: results/revision/operations_<metro>.json
"""
from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.accessibility import (
    FareSpec, aam_wait_minutes, compute_aam_cost_matrix, compute_base_cost_matrix,
    gravity_accessibility, minutes_per_dollar_by_tract,
)
from src.candidates import build_vertiport_candidates
from src.config import CFG, RESULTS
from src.data_layer import load_lodes_wac, load_tracts_joined
from src.equity_metrics import concentration_index

MAX_CAND = 50
SEATS = 4
MAX_HOLD_MIN = 10.0
TURNAROUND_MIN = (5.0, 7.0, 12.0)        # triangular: minimum, mode, maximum
OPERATING_HOURS = 16.0
N_DAYS = 200
DEMAND_PAX_PER_HOUR = (5, 10, 20, 30, 40, 50, 60, 70)
ASSUMED_OPS_PER_DAY = 80.0
FARES_PER_KM = (0.0, CFG.aam.fare_per_km["public_service"], CFG.aam.fare_per_km["long_run"])


def simulate_vertiport(demand_per_hour: float, pads: int, rng) -> dict:
    waits, flights, boarded, busy = [], 0, 0, 0.0
    horizon = OPERATING_HOURS * 60.0
    for _ in range(N_DAYS):
        n = rng.poisson(demand_per_hour * OPERATING_HOURS)
        arr = np.sort(rng.uniform(0.0, horizon, n))
        pad_free = np.zeros(pads)
        i = 0
        while i < n:
            pad = int(np.argmin(pad_free))
            full_at = arr[i + SEATS - 1] if i + SEATS - 1 < n else np.inf
            start = max(pad_free[pad], min(full_at, arr[i] + MAX_HOLD_MIN))
            j = min(i + SEATS, int(np.searchsorted(arr, start, side="right")))
            j = max(j, i + 1)
            turnaround = rng.triangular(*TURNAROUND_MIN)
            pad_free[pad] = start + turnaround
            waits.extend((start - arr[i:j]).tolist())
            flights += 1
            boarded += j - i
            busy += turnaround
            i = j
    waits = np.asarray(waits)
    return {
        "demand_pax_per_hour": demand_per_hour,
        "mean_wait_min": float(waits.mean()),
        "p90_wait_min": float(np.percentile(waits, 90)),
        "flights_per_hour": flights / (N_DAYS * OPERATING_HOURS),
        "load_factor": boarded / (flights * SEATS),
        "pad_utilisation": busy / (N_DAYS * horizon * pads),
        # each dispatched flight is one departure and, in a balanced network, one arrival
        "movements_per_day": 2.0 * flights / N_DAYS,
    }


def main():
    metro = sys.argv[1]
    assert CFG.ground_time_source == "network"
    rng = np.random.default_rng(0)
    pads = CFG.aam.pads_per_vertiport
    sim = [simulate_vertiport(d, pads, rng) for d in DEMAND_PAX_PER_HOUR]
    for r in sim:
        r["dnl_shift_db_vs_assumed_ops"] = float(10 * np.log10(r["movements_per_day"] / ASSUMED_OPS_PER_DAY))
        print(f"[A] {r['demand_pax_per_hour']:>3} pax/h: wait {r['mean_wait_min']:.1f} min (p90 {r['p90_wait_min']:.1f}), "
              f"load {r['load_factor']:.2f}, pads {r['pad_utilisation']:.2f}, {r['movements_per_day']:.0f} movements/day",
              flush=True)

    tracts = load_tracts_joined(metro)
    candidates = build_vertiport_candidates(metro=metro, max_candidates=MAX_CAND)
    base, jobs = compute_base_cost_matrix(tracts, load_lodes_wac(metro), metro)
    A_base = gravity_accessibility(base, jobs)
    pop = tracts["pop"].fillna(0).values.astype(float)
    income = tracts["median_hh_income"].fillna(tracts["median_hh_income"].median()).values
    weight = minutes_per_dollar_by_tract(tracts, CFG.aam)
    all_idx = np.arange(len(candidates))

    waits = {"fixed_headway_assumption": aam_wait_minutes(CFG.aam)}
    waits.update({f"simulated_{r['demand_pax_per_hour']}_pax_per_hour": r["mean_wait_min"]
                  for r in sim if r["demand_pax_per_hour"] in (10, 30, 60)})
    rows = []
    for label, wait in waits.items():
        params = dataclasses.replace(CFG.aam, passenger_wait_min=wait)
        for rate in FARES_PER_KM:
            aam = compute_aam_cost_matrix(tracts, candidates, all_idx, metro, params,
                                          FareSpec(base=params.fare_base, per_km=rate, weight=weight))
            dA = np.maximum(gravity_accessibility(np.minimum(base, aam), jobs) - A_base, 0.0)
            rows.append({
                "wait": label, "wait_min": wait, "fare_per_km": rate,
                "mean_dA_pct_of_base": 100 * float(np.average(dA, weights=pop) / np.average(A_base, weights=pop)),
                "share_tracts_dA_pos": float((dA > 0).mean()),
                "ci_dA_vs_income": concentration_index(dA, income, pop),
            })
            print(f"[B] {metro} {label} ({wait:.1f} min) ${rate:.2f}/km: dA={rows[-1]['mean_dA_pct_of_base']:.3f}%",
                  flush=True)

    out = {"metro": metro,
           "simulation": {"seats": SEATS, "pads": pads, "max_hold_min": MAX_HOLD_MIN,
                          "turnaround_min_triangular": TURNAROUND_MIN,
                          "operating_hours": OPERATING_HOURS, "days": N_DAYS, "results": sim},
           "accessibility_by_wait": rows}
    out_dir = RESULTS / "revision"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"operations_{metro}.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"{metro}: done", flush=True)


if __name__ == "__main__":
    main()
