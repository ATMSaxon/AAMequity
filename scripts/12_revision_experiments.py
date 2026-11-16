"""Supplementary experiments for the revision, per metro.

For the accessibility-maximising network (S2) at the long-run and public-service
fare tiers:
  uncertainty   bootstrap intervals for the income concentration indices and the
                burden-HOLC rank correlation
  groups        concentration and exposure-disparity by minority share and by
                zero-vehicle households
  redlining     burden and noise by HOLC grade, benchmarked against random
                networks drawn from the same candidate pool, the full candidate
                pool, and a partial correlation controlling for candidate density
  noise         population by day-night average sound level band
  network size  outcomes as the number of vertiports varies
and, on the full candidate network, the fare sweep under alternative
value-of-time shares.

Usage:  python scripts/12_revision_experiments.py <metro>
Output: results/revision/experiments_<metro>.json
"""
from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.accessibility import (
    FareSpec, compute_aam_cost_matrix, distance_fare, gravity_accessibility,
    minutes_per_dollar_by_tract,
)
from src.burden import tract_vertiport_distance_m, vertiport_dnl_dba
from src.candidates import build_vertiport_candidates
from src.config import CFG, RESULTS
from src.data_layer import load_tracts_joined
from src.equity_metrics import concentration_index, exposure_disparity_ratio
from src.groups import assign_groups
from src.optimizer import solve_scenario
from src.redlining import assign_holc_grade_to_tracts
from src.scenarios import _build_inputs, precompute_contributions, realise

N_VP = 25
MAX_CAND = 50
OPS_PER_DAY = 80.0
TIERS = ("long_run", "public_service")
N_BOOT = 1000
N_RANDOM = 2000
NETWORK_SIZES = (5, 10, 15, 20, 25, 30, 35, 40, 45)
DNL_BANDS = (25, 30, 35, 40, 45, 50)
DENSITY_RADIUS_M = 3000.0
VOT_SHARES = (0.35, 0.5, 0.6, 1.0)
VOT_RATES = (0.05, 0.10, 0.29, 0.61, 1.85)


def interval(fn, arrays, rng) -> dict:
    """Point estimate and 95% tract-bootstrap interval of fn(*arrays)."""
    n = len(arrays[0])
    draws = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        v = fn(*[a[idx] for a in arrays])
        if np.isfinite(v):
            draws.append(v)
    lo, hi = (np.percentile(draws, [2.5, 97.5]) if draws else (np.nan, np.nan))
    return {"estimate": float(fn(*arrays)), "low": float(lo), "high": float(hi)}


def rho(x, y) -> float:
    return float(spearmanr(x, y)[0]) if np.ptp(x) > 0 and np.ptp(y) > 0 else float("nan")


def partial_rho(x, y, z) -> float:
    """Spearman correlation of x and y after removing the rank-linear effect of z."""
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    ex = rx - np.polyval(np.polyfit(rz, rx, 1), rz)
    ey = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
    return float(np.corrcoef(ex, ey)[0, 1])


def network_dnl(d_all: np.ndarray, sel: np.ndarray) -> np.ndarray:
    energy = (10.0 ** (vertiport_dnl_dba(d_all[:, sel], OPS_PER_DAY, CFG.burden) / 10.0)).sum(axis=1)
    return 10.0 * np.log10(energy)


def main():
    metro = sys.argv[1]
    assert CFG.ground_time_source == "network"
    rng = np.random.default_rng(0)
    candidates = build_vertiport_candidates(metro=metro, max_candidates=MAX_CAND)
    tracts = load_tracts_joined(metro)
    geoid = tracts["tract_geoid"].values
    pop = tracts["pop"].fillna(0).values.astype(float)
    income = tracts["median_hh_income"].fillna(tracts["median_hh_income"].median()).values
    groups = assign_groups(metro).set_index("tract_geoid").reindex(geoid)
    holc = assign_holc_grade_to_tracts(metro).set_index("tract_geoid").reindex(geoid)
    has_holc = holc["holc_rank"].notna().values
    holc_rank = holc["holc_rank"].values[has_holc].astype(float)
    holc_grade = holc["holc_grade"].values[has_holc]

    d_all = tract_vertiport_distance_m(tracts, candidates, np.arange(len(candidates)), metro)
    cand_density = (d_all <= DENSITY_RADIUS_M).sum(axis=1).astype(float)

    out = {"metro": metro, "n_tracts": int(len(geoid)), "n_tracts_with_holc": int(has_holc.sum()),
           "holc_is_optimisation_input": False, "tiers": {}}

    for tier in TIERS:
        pre = precompute_contributions(metro, candidates, CFG.aam, CFG.burden,
                                       distance_fare(tracts, CFG.aam, tier),
                                       ops_per_day_per_vertiport=OPS_PER_DAY)
        inp = _build_inputs(pre)
        b_ik = pre["b_ik"]
        A_base_mean = float(np.average(pre["A_base"], weights=pop))
        res = solve_scenario("S2", inp, n_vertiports=N_VP, alpha=1.0)
        sel = res.selected_idx
        dA, B = realise(pre, sel, metro)
        dnl = network_dnl(d_all, sel)
        t = {"fare_per_km": CFG.aam.fare_per_km[tier], "selected": sel.tolist(),
             "mean_dA_pct_of_base": 100 * float(np.average(dA, weights=pop)) / A_base_mean}

        # --- uncertainty -----------------------------------------------------------
        t["uncertainty"] = {
            "ci_dA_vs_income": interval(concentration_index, [dA, income, pop], rng),
            "ci_B_vs_income": interval(concentration_index, [B, income, pop], rng),
            "rho_B_vs_holc": interval(rho, [B[has_holc], holc_rank], rng),
            "rho_dA_vs_holc": interval(rho, [dA[has_holc], holc_rank], rng),
        }

        # --- other equity dimensions ---------------------------------------------
        minority = groups["pct_minority"].fillna(groups["pct_minority"].median()).values
        carless = groups["pct_zero_vehicle"].fillna(groups["pct_zero_vehicle"].median()).values
        t["groups"] = {
            "ci_dA_vs_minority_share": interval(concentration_index, [dA, minority, pop], rng),
            "ci_B_vs_minority_share": interval(concentration_index, [B, minority, pop], rng),
            "ci_dA_vs_zero_vehicle_share": interval(concentration_index, [dA, carless, pop], rng),
            "ci_B_vs_zero_vehicle_share": interval(concentration_index, [B, carless, pop], rng),
            "edr_dA_minority_group": exposure_disparity_ratio(dA, groups["minority_group"].values, pop),
            "edr_B_minority_group": exposure_disparity_ratio(B, groups["minority_group"].values, pop),
            "edr_dA_carless_group": exposure_disparity_ratio(dA, groups["carless_group"].values, pop),
            "edr_B_carless_group": exposure_disparity_ratio(B, groups["carless_group"].values, pop),
        }

        # --- burden and noise against HOLC grades, with counterfactual networks -----
        random_rho_B, random_rho_dnl = [], []
        for _ in range(N_RANDOM):
            pick = rng.choice(len(candidates), N_VP, replace=False)
            random_rho_B.append(rho(b_ik[:, pick].sum(axis=1)[has_holc], holc_rank))
            random_rho_dnl.append(rho(network_dnl(d_all, pick)[has_holc], holc_rank))
        random_rho_B, random_rho_dnl = np.array(random_rho_B), np.array(random_rho_dnl)
        rho_B, rho_dnl = rho(B[has_holc], holc_rank), rho(dnl[has_holc], holc_rank)
        all_idx = np.arange(len(candidates))
        t["redlining"] = {
            "rho_B_vs_holc": rho_B,
            "rho_dnl_vs_holc": rho_dnl,
            "rho_B_vs_holc_given_candidate_density": partial_rho(B[has_holc], holc_rank, cand_density[has_holc]),
            "rho_candidate_density_vs_holc": rho(cand_density[has_holc], holc_rank),
            "full_candidate_pool": {"rho_B_vs_holc": rho(b_ik.sum(axis=1)[has_holc], holc_rank),
                                    "rho_dnl_vs_holc": rho(network_dnl(d_all, all_idx)[has_holc], holc_rank)},
            "random_networks": {
                "n": N_RANDOM,
                "rho_B_mean": float(np.nanmean(random_rho_B)),
                "rho_B_p2.5": float(np.nanpercentile(random_rho_B, 2.5)),
                "rho_B_p97.5": float(np.nanpercentile(random_rho_B, 97.5)),
                "share_random_rho_B_below_selected": float(np.nanmean(random_rho_B < rho_B)),
                "rho_dnl_mean": float(np.nanmean(random_rho_dnl)),
                "rho_dnl_p2.5": float(np.nanpercentile(random_rho_dnl, 2.5)),
                "rho_dnl_p97.5": float(np.nanpercentile(random_rho_dnl, 97.5)),
                "share_random_rho_dnl_below_selected": float(np.nanmean(random_rho_dnl < rho_dnl)),
            },
            "by_grade": {
                g: {"n_tracts": int((holc_grade == g).sum()),
                    "mean_B": float(np.average(B[has_holc][holc_grade == g], weights=pop[has_holc][holc_grade == g])),
                    "dnl_of_mean_energy": float(10 * np.log10(np.average(
                        10 ** (dnl[has_holc][holc_grade == g] / 10), weights=pop[has_holc][holc_grade == g]))),
                    "mean_dA": float(np.average(dA[has_holc][holc_grade == g], weights=pop[has_holc][holc_grade == g]))}
                for g in "ABCD" if (holc_grade == g).any() and pop[has_holc][holc_grade == g].sum() > 0},
        }

        # --- noise exposure -----------------------------------------------------------
        t["noise"] = {
            "max_dnl": float(dnl.max()),
            "population_at_or_above_dnl": {str(b): float(pop[dnl >= b].sum()) for b in DNL_BANDS},
            "population_total": float(pop.sum()),
            "ci_dnl_energy_vs_income": concentration_index(10 ** (dnl / 10), income, pop),
        }

        # --- network size -------------------------------------------------------------
        rows = []
        for n_vp in NETWORK_SIZES:
            r = solve_scenario(f"S2_n{n_vp}", inp, n_vertiports=n_vp, alpha=1.0)
            dA_n, B_n = realise(pre, r.selected_idx, metro)
            dnl_n = network_dnl(d_all, r.selected_idx)
            rows.append({
                "n_vertiports": n_vp, "status": r.status,
                "mean_dA_pct_of_base": 100 * float(np.average(dA_n, weights=pop)) / A_base_mean,
                "share_tracts_dA_pos": float((dA_n > 0).mean()),
                "ci_dA_vs_income": concentration_index(dA_n, income, pop),
                "mean_B": float(np.average(B_n, weights=pop)),
                "ci_B_vs_income": concentration_index(B_n, income, pop),
                "population_dnl_ge_40": float(pop[dnl_n >= 40].sum()),
                "rho_B_vs_holc": rho(B_n[has_holc], holc_rank),
            })
        t["network_size"] = rows
        out["tiers"][tier] = t
        print(f"[{metro}] {tier}: dA={t['mean_dA_pct_of_base']:.4f}% "
              f"CI={t['uncertainty']['ci_dA_vs_income']} rhoB={rho_B:+.3f} "
              f"random=[{t['redlining']['random_networks']['rho_B_p2.5']:+.3f},"
              f"{t['redlining']['random_networks']['rho_B_p97.5']:+.3f}]", flush=True)

    # --- value-of-time share, full candidate network ---------------------------------
    base, jobs, A_base = pre["base_cost"], pre["jobs"], pre["A_base"]
    all_idx = np.arange(len(candidates))
    vot_rows = []
    for share in VOT_SHARES:
        weight = minutes_per_dollar_by_tract(tracts, dataclasses.replace(CFG.aam, vot_income_share=share))
        for rate in VOT_RATES:
            aam = compute_aam_cost_matrix(tracts, candidates, all_idx, metro, CFG.aam,
                                          FareSpec(base=CFG.aam.fare_base, per_km=rate, weight=weight))
            dA = np.maximum(gravity_accessibility(np.minimum(base, aam), jobs) - A_base, 0.0)
            vot_rows.append({
                "vot_income_share": share, "fare_per_km": rate,
                "mean_dA_pct_of_base": 100 * float(np.average(dA, weights=pop) / np.average(A_base, weights=pop)),
                "share_tracts_dA_pos": float((dA > 0).mean()),
                "ci_dA_vs_income": concentration_index(dA, income, pop),
            })
    out["value_of_time_sensitivity"] = vot_rows

    out_dir = RESULTS / "revision"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"experiments_{metro}.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"{metro}: done", flush=True)


if __name__ == "__main__":
    main()
