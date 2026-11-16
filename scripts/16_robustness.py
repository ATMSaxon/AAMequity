"""Robustness of the headline results, per metro.

calibration   out-of-sample fit of the congestion model, and outcomes when the
              calibrated ground times are perturbed
sign          income concentration of accessibility gains under a uniform value
              of time: optimised network with bootstrap interval, random networks
              from the same pool, and a greenfield candidate pool
parameters    one-at-a-time changes to the accessibility model
burden        income concentration of burden under alternative component
              weights and alternative noise parameters
optimisation  the surrogate-optimal network against a greedy network built on
              the realised objective

Usage:  python scripts/16_robustness.py <metro>
Output: results/revision/robustness_<metro>.json
"""
from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.accessibility import (
    FareSpec, compute_aam_cost_matrix, compute_base_cost_matrix, distance_fare,
    gravity_accessibility,
)
from src.burden import tract_vertiport_distance_m, vertiport_dnl_dba
from src.candidates import build_greenfield_candidates, build_vertiport_candidates
from src.config import CFG, RESULTS
from src.congestion import cross_validate
from src.data_layer import load_lodes_wac, load_tracts_joined
from src.equity_metrics import concentration_index
from src.optimizer import solve_scenario
from src.scenarios import _build_inputs, precompute_contributions

N_VP = 25
MAX_CAND = 50
N_GREENFIELD = 100
OPS_PER_DAY = 80.0
N_BOOT = 1000
N_RANDOM = 300
N_RANDOM_GREENFIELD = 100
N_WEIGHT_DRAWS = 500
TIERS = ("public_service", "long_run")
CALIBRATION_VARIANTS = {          # (congestion multiplier scale, terminal time shift in min)
    "calibrated": (1.0, 0.0), "congestion_-20%": (0.8, 0.0), "congestion_+20%": (1.2, 0.0),
    "free_flow": (0.0, 0.0), "terminal_-3min": (1.0, -3.0), "terminal_+3min": (1.0, 3.0),
}


class Metro:
    """Tract data and accessibility evaluation for one metro."""

    def __init__(self, metro: str):
        self.metro = metro
        self.tracts = load_tracts_joined(metro)
        self.wac = load_lodes_wac(metro)
        self.pop = self.tracts["pop"].fillna(0).values.astype(float)
        inc = self.tracts["median_hh_income"].astype(float)
        self.income = inc.fillna(inc.median()).values
        self.uniform_weight = 60.0 / (CFG.aam.vot_income_share * float(np.median(self.income))
                                      / CFG.aam.vot_hours_per_year)
        self.refresh_base()

    def refresh_base(self):
        self.base, self.jobs = compute_base_cost_matrix(self.tracts, self.wac, self.metro)

    def fare(self, vot: str, tier: str) -> FareSpec:
        if vot == "uniform":
            return FareSpec(base=CFG.aam.fare_base, per_km=CFG.aam.fare_per_km[tier], weight=self.uniform_weight)
        return distance_fare(self.tracts, CFG.aam, tier)

    def gain(self, candidates, sel, fare, params=None, beta=1.0, jobs=None) -> np.ndarray:
        jobs = self.jobs if jobs is None else jobs
        aam = compute_aam_cost_matrix(self.tracts, candidates, np.asarray(sel), self.metro,
                                      params or CFG.aam, fare)
        a0 = gravity_accessibility(self.base, jobs, beta_per_hour=beta)
        return np.maximum(gravity_accessibility(np.minimum(self.base, aam), jobs, beta_per_hour=beta) - a0, 0.0), a0

    def summary(self, dA, a0) -> dict:
        return {"mean_dA_pct_of_base": 100 * float(np.average(dA, weights=self.pop) / np.average(a0, weights=self.pop)),
                "share_tracts_dA_pos": float((dA > 0).mean()),
                "ci_dA_vs_income": concentration_index(dA, self.income, self.pop)}


def bootstrap_ci(dA, income, pop, rng) -> dict:
    n = len(dA)
    draws = [concentration_index(dA[idx], income[idx], pop[idx])
             for idx in (rng.integers(0, n, n) for _ in range(N_BOOT))]
    return {"estimate": concentration_index(dA, income, pop),
            "low": float(np.percentile(draws, 2.5)), "high": float(np.percentile(draws, 97.5))}


def spread(values) -> dict:
    v = np.asarray(values, dtype=float)
    return {"mean": float(np.nanmean(v)), "p2.5": float(np.nanpercentile(v, 2.5)),
            "p97.5": float(np.nanpercentile(v, 97.5)), "share_positive": float(np.nanmean(v > 0)),
            "n": int(len(v))}


def main():
    metro = sys.argv[1]
    assert CFG.ground_time_source == "network"
    rng = np.random.default_rng(0)
    M = Metro(metro)
    cand = build_vertiport_candidates(metro=metro, max_candidates=MAX_CAND)
    all_idx = np.arange(len(cand))
    out = {"metro": metro, "uniform_minutes_per_dollar": M.uniform_weight}

    # ---- calibration -------------------------------------------------------------------
    out["calibration_cross_validation"] = cross_validate(metro)
    rows = {}
    for label, (scale, shift) in CALIBRATION_VARIANTS.items():
        CFG.congestion_multiplier_scale, CFG.terminal_time_shift_min = scale, shift
        M.refresh_base()
        rows[label] = {vot: M.summary(*M.gain(cand, all_idx, M.fare(vot, "public_service")))
                       for vot in ("uniform", "income")}
        rows[label]["free_of_fare"] = M.summary(*M.gain(cand, all_idx, FareSpec.flat(0.0, 0.0)))
        print(f"[calibration] {metro} {label}: " + " ".join(
            f"{k}={v['mean_dA_pct_of_base']:.3f}%/{v['ci_dA_vs_income']:+.3f}" for k, v in rows[label].items()), flush=True)
    CFG.congestion_multiplier_scale, CFG.terminal_time_shift_min = 1.0, 0.0
    M.refresh_base()
    out["calibration_perturbation"] = rows

    # ---- sign of the income concentration under a uniform value of time -----------------
    sign = {}
    networks = {}
    for tier in TIERS:
        fare = M.fare("uniform", tier)
        pre = precompute_contributions(metro, cand, CFG.aam, CFG.burden, fare, ops_per_day_per_vertiport=OPS_PER_DAY)
        sel = solve_scenario("S2", _build_inputs(pre), n_vertiports=N_VP, alpha=1.0).selected_idx
        networks[tier] = (pre, sel)
        dA, a0 = M.gain(cand, sel, fare)
        sign[tier] = {"optimised": {**M.summary(dA, a0), "interval": bootstrap_ci(dA, M.income, M.pop, rng)},
                      "full_pool": M.summary(*M.gain(cand, all_idx, fare))}
        print(f"[sign] {metro} {tier}: optimised CI={sign[tier]['optimised']['interval']}", flush=True)
    fare = M.fare("uniform", "public_service")
    random_ci, random_gain = [], []
    for _ in range(N_RANDOM):
        dA, a0 = M.gain(cand, rng.choice(len(cand), N_VP, replace=False), fare)
        random_ci.append(concentration_index(dA, M.income, M.pop))
        random_gain.append(float(np.average(dA, weights=M.pop)))
    sign["public_service"]["random_networks_ci"] = spread(random_ci)

    green = build_greenfield_candidates(metro, N_GREENFIELD)
    green_ci = []
    for _ in range(N_RANDOM_GREENFIELD):
        dA, _ = M.gain(green, rng.choice(len(green), N_VP, replace=False), fare)
        green_ci.append(concentration_index(dA, M.income, M.pop))
    sign["public_service"]["greenfield_full_pool"] = M.summary(*M.gain(green, np.arange(len(green)), fare))
    sign["public_service"]["greenfield_random_networks_ci"] = spread(green_ci)
    out["sign_uniform_value_of_time"] = sign
    print(f"[sign] {metro} random={sign['public_service']['random_networks_ci']} "
          f"greenfield={sign['public_service']['greenfield_random_networks_ci']}", flush=True)

    # ---- one-at-a-time model parameters ----------------------------------------------------
    high_wage = (M.tracts[["tract_geoid"]].merge(M.wac[["dest_tract", "jobs_high_wage"]], left_on="tract_geoid",
                                                 right_on="dest_tract", how="left")["jobs_high_wage"].fillna(0).values)
    variants = {"baseline": {}}
    variants.update({f"beta_{b}": {"beta": b} for b in (0.5, 2.0)})
    variants.update({f"max_trip_{v}km": {"params": {"max_trip_km": v}} for v in (60.0, 150.0)})
    variants.update({f"min_trip_{v}km": {"params": {"min_trip_km": v}} for v in (8.0, 20.0)})
    variants.update({f"cruise_{v}kmh": {"params": {"cruise_speed_kmh": v}} for v in (240.0, 320.0)})
    variants.update({f"security_{v}min": {"params": {"t_security_min": v}} for v in (4.0, 12.0)})
    variants.update({f"wait_{v}min": {"params": {"passenger_wait_min": v}} for v in (3.0, 6.0)})
    variants["high_wage_jobs"] = {"jobs": high_wage}
    par = {}
    for label, v in variants.items():
        params = dataclasses.replace(CFG.aam, **v.get("params", {}))
        par[label] = {vot: M.summary(*M.gain(cand, all_idx, M.fare(vot, "public_service"), params=params,
                                             beta=v.get("beta", 1.0), jobs=v.get("jobs")))
                      for vot in ("uniform", "income")}
        print(f"[parameters] {metro} {label}: " + " ".join(
            f"{k}={r['mean_dA_pct_of_base']:.3f}%/{r['ci_dA_vs_income']:+.3f}" for k, r in par[label].items()), flush=True)
    out["parameter_sensitivity"] = par

    # ---- burden: weights and noise parameters ----------------------------------------------
    pre, sel = networks["public_service"]
    comp = {k: v[:, sel].sum(axis=1) for k, v in pre["burden_components"].items()}
    names = list(comp)
    stack = np.column_stack([comp[k] for k in names])
    base_w = np.array([CFG.burden.w_noise, CFG.burden.w_visual, CFG.burden.w_traffic,
                       CFG.burden.w_landuse, CFG.burden.w_safety])
    ci_b = lambda B: concentration_index(B, M.income, M.pop)
    draws = [ci_b(stack @ w) for w in rng.dirichlet(np.ones(len(names)), N_WEIGHT_DRAWS)]
    burden = {
        "component_order": names,
        "paper_weights": ci_b(stack @ base_w),
        "equal_weights": ci_b(stack.mean(axis=1)),
        "single_component": {k: ci_b(comp[k]) for k in names},
        "random_weights": {**spread(draws), "share_negative": float(np.mean(np.array(draws) < 0))},
    }
    d = tract_vertiport_distance_m(M.tracts, cand, sel, metro)
    noise = {}
    for label, change in {"baseline": {}, "duration_5s": {"event_duration_100m_s": 5.0},
                          "duration_20s": {"event_duration_100m_s": 20.0},
                          "duration_cap_60s": {"event_duration_max_s": 60.0},
                          "duration_cap_150s": {"event_duration_max_s": 150.0},
                          "absorption_2.0": {"atmospheric_absorption_db_per_km": 2.0},
                          "absorption_5.5": {"atmospheric_absorption_db_per_km": 5.5}}.items():
        p = dataclasses.replace(CFG.burden, **change)
        energy = (10 ** (vertiport_dnl_dba(d, OPS_PER_DAY, p) / 10)).sum(axis=1)
        dnl = 10 * np.log10(energy)
        noise[label] = {"ci_noise_energy_vs_income": ci_b(energy), "max_dnl": float(dnl.max()),
                        "population_share_dnl_ge_40": float(M.pop[dnl >= 40].sum() / M.pop.sum())}
    burden["noise_parameters"] = noise
    out["burden"] = burden
    print(f"[burden] {metro}: paper={burden['paper_weights']:+.3f} random={burden['random_weights']}", flush=True)

    # ---- optimisation: surrogate-optimal against greedy on the realised objective ----------
    realised = lambda s: float(np.average(M.gain(cand, s, fare)[0], weights=M.pop))
    chosen, remaining, value = [], list(range(len(cand))), 0.0
    for _ in range(N_VP):
        scores = [realised(chosen + [k]) for k in remaining]
        best = int(np.argmax(scores))
        value = scores[best]
        chosen.append(remaining.pop(best))
    surrogate_value = realised(networks["public_service"][1])
    out["optimisation"] = {
        "realised_gain_surrogate_network": surrogate_value,
        "realised_gain_greedy_network": value,
        "surrogate_over_greedy": surrogate_value / value if value > 0 else float("nan"),
        "sites_in_common": int(len(set(chosen) & set(networks["public_service"][1].tolist()))),
        "random_networks_gain": spread(random_gain),
        "ci_greedy_network": concentration_index(M.gain(cand, chosen, fare)[0], M.income, M.pop),
    }
    print(f"[optimisation] {metro}: {out['optimisation']}", flush=True)

    out_dir = RESULTS / "revision"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"robustness_{metro}.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"{metro}: done", flush=True)


if __name__ == "__main__":
    main()
