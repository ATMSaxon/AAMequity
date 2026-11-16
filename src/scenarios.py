"""Scenario orchestrator — runs all 7 policy scenarios end-to-end.

For each scenario:
  1. Configure objective weights + constraints
  2. Pre-compute per-vertiport (g_ik, b_ik) matrices (these don't depend on scenario)
  3. Solve MILP
  4. Compute realized accessibility A_i^AAM with the selected vertiport set
  5. Compute realized B_i
  6. Run full equity panel
  7. Save per-scenario CSV / JSON to results/
"""

from __future__ import annotations
import json
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from .config import CFG, RESULTS, AAMParams, BurdenParams, OptimizerParams
from .data_layer import load_tracts_joined, load_lodes_wac
from .candidates import build_vertiport_candidates, _to_metric, _metric_crs_for
from .accessibility import (
    FareSpec, compute_aam_cost_matrix, distance_fare, gravity_accessibility,
    compute_base_cost_matrix, euclid_dist_m,
)
from .burden import (
    noise_exposure_per_tract, visual_burden, traffic_burden, landuse_burden, safety_burden,
    _normalize, burden_validation_against_ces,
)
from .equity_metrics import compute_full_equity_panel
from .groups import assign_groups
from .optimizer import ScenarioInputs, ScenarioResult, solve_scenario


# ----------------------------------------------------------------------------
# Pre-compute per-vertiport contribution matrices (g_ik, b_ik)
# ----------------------------------------------------------------------------
def precompute_contributions(metro: str,
                              candidates: gpd.GeoDataFrame,
                              params_aam: AAMParams,
                              params_burden: BurdenParams,
                              fare: FareSpec,
                              ops_per_day_per_vertiport: float = 80.0) -> dict:
    """For each candidate k computed in isolation:
       - per-tract accessibility gain g_ik
       - per-tract burden contribution b_ik (each component normalized at metro level)
    Returns dict with arrays + tract metadata.
    """
    tracts = load_tracts_joined(metro)
    wac = load_lodes_wac(metro)
    n_tract = len(tracts)
    n_cand = len(candidates)

    base_cost, jobs = compute_base_cost_matrix(tracts, wac, metro)
    A_base = gravity_accessibility(base_cost, jobs)

    # Marginal-benefit (leave-one-out) g_ik formulation:
    #   Compute A_full   = accessibility with ALL candidates available
    #   For each k: A_minus_k = accessibility with all-but-k
    #   g_ik = A_full_i - A_minus_k_i  (≥ 0, marginal value of k)
    # Then ∑_k g_ik · x_k underestimates true joint benefit but gives a valid
    # additive lower bound usable in the MILP objective.
    print(f"  precomputing g_ik / b_ik for {n_cand} candidates × {n_tract} tracts ...")
    t0 = time.time()
    all_idx = np.arange(n_cand)
    aam_cost_full = compute_aam_cost_matrix(tracts, candidates, all_idx, metro, params_aam, fare)
    eff_full = np.minimum(base_cost, aam_cost_full)
    A_full = gravity_accessibility(eff_full, jobs)

    g_ik = np.zeros((n_tract, n_cand))
    raw_noise = np.zeros((n_tract, n_cand))
    raw_visual = np.zeros((n_tract, n_cand))
    raw_traffic = np.zeros((n_tract, n_cand))
    raw_landuse = np.zeros((n_tract, n_cand))
    raw_safety = np.zeros((n_tract, n_cand))  # FIX P3: 5th component restored

    for k in range(n_cand):
        keep = np.delete(all_idx, k)
        aam_cost_minus_k = compute_aam_cost_matrix(tracts, candidates, keep, metro, params_aam, fare)
        eff_minus_k = np.minimum(base_cost, aam_cost_minus_k)
        A_minus_k = gravity_accessibility(eff_minus_k, jobs)
        g_ik[:, k] = np.maximum(A_full - A_minus_k, 0.0)

        # Per-vertiport per-tract burden contribution (additive across selected vertiports)
        sel = np.array([k])
        raw_noise[:, k]   = noise_exposure_per_tract(tracts, candidates, sel,
                                                     ops_per_day_per_vertiport, params_burden, metro)
        raw_visual[:, k]  = visual_burden(tracts, candidates, sel, metro)
        raw_traffic[:, k] = traffic_burden(tracts, candidates, sel, metro)
        raw_landuse[:, k] = landuse_burden(tracts, candidates, sel, metro)
        # Safety must be approximated per-vertiport (single-vertiport contribution to corridor risk):
        # Use distance-from-vertiport as the per-vertiport safety contribution.
        from .accessibility import euclid_dist_m
        from .candidates import _to_metric, _metric_crs_for
        epsg_m = _metric_crs_for(metro)
        tm = _to_metric(tracts, epsg_m)
        sm = _to_metric(candidates.iloc[sel], epsg_m)
        t_xy = np.array([(g.x, g.y) for g in tm.geometry.centroid])
        s_xy = np.array([(g.x, g.y) for g in sm.geometry])
        dist = euclid_dist_m(t_xy, s_xy).flatten()
        raw_safety[:, k] = np.clip(1.0 - dist / 5000.0, 0.0, 1.0)

        if (k + 1) % 20 == 0 or k == n_cand - 1:
            print(f"    [{k+1}/{n_cand}]  elapsed {time.time()-t0:.1f}s  "
                  f"g_ik mean={g_ik[:, k].mean():.1f}, max={g_ik[:, k].max():.0f}")

    # Normalize each component's matrix by 99th percentile of its full-network sum
    full_noise   = raw_noise.sum(axis=1)
    full_visual  = raw_visual.sum(axis=1)
    full_traffic = raw_traffic.sum(axis=1)
    full_landuse = raw_landuse.sum(axis=1)

    def _ref(x): return max(np.nanpercentile(x[x > 0], 99) if np.any(x > 0) else 1.0, 1e-9)
    n99 = _ref(full_noise);  v99 = _ref(full_visual)
    t99 = _ref(full_traffic); l99 = _ref(full_landuse)
    full_safety = raw_safety.sum(axis=1); s99 = _ref(full_safety)

    # FIX P3: 5-component burden index now includes safety
    b_ik = (params_burden.w_noise   * raw_noise   / n99 +
            params_burden.w_visual  * raw_visual  / v99 +
            params_burden.w_traffic * raw_traffic / t99 +
            params_burden.w_landuse * raw_landuse / l99 +
            params_burden.w_safety  * raw_safety  / s99)

    # Tract attributes for groups + ranking
    grp_df = assign_groups(metro)
    grp_df = tracts[["tract_geoid", "pop", "median_hh_income"]].merge(
        grp_df.drop(columns=["pop", "median_hh_income"]), on="tract_geoid", how="left")
    grp_df = grp_df.set_index("tract_geoid").reindex(tracts["tract_geoid"]).reset_index()

    return {
        "tracts": tracts,
        "g_ik": g_ik,
        "b_ik": b_ik,
        "burden_components": {          # normalised, before weights
            "noise": raw_noise / n99, "visual": raw_visual / v99, "traffic": raw_traffic / t99,
            "landuse": raw_landuse / l99, "safety": raw_safety / s99},
        "A_base": A_base,
        "base_cost": base_cost,         # for FIX P1: post-solve recompute
        "jobs": jobs,                   # for FIX P1: post-solve recompute
        "params_aam": params_aam,       # for FIX P1: post-solve recompute
        "fare": fare,                   # for FIX P1: post-solve recompute
        "groups": grp_df,
        "candidates": candidates,
    }


# ----------------------------------------------------------------------------
# Scenario definitions
# ----------------------------------------------------------------------------
SCENARIOS = ["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7"]

SCENARIO_DESC = {
    "S0": "baseline (no AAM)",
    "S1": "market-led (income-weighted accessibility gain)",
    "S2": "accessibility-maximizing",
    "S3": "low-income floor (share of the best attainable low-income gain)",
    "S4": "transit-integrated",
    "S5": "environmental-justice burden cap (per-tract, share of the unconstrained peak)",
    "S6": "means-tested fare subsidy for low-income tracts",
    "S7": "procedural coverage (share of the best attainable gain of procedurally disadvantaged tracts)",
}


def _build_inputs(pre: dict) -> ScenarioInputs:
    tracts = pre["tracts"]
    cand = pre["candidates"]
    groups = pre["groups"]
    proc_label = (groups["procedural_disadvantage"]
                    .fillna("rest").values if "procedural_disadvantage" in groups.columns
                    else np.array(["rest"] * len(tracts)))
    proc_disadv = (proc_label == "high_proc_disadv").astype(int)
    return ScenarioInputs(
        g_ik=pre["g_ik"],
        b_ik=pre["b_ik"],
        pop_i=tracts["pop"].fillna(0).values.astype(float),
        income_i=tracts["median_hh_income"].fillna(np.nanmedian(tracts["median_hh_income"])).values,
        group_i=groups["income_group"].fillna("mid_income").values.astype(object),
        transit_int_k=cand["transit_integrated"].fillna(False).astype(int).values,
        ces_pctl_k=cand["ces_pctl"].fillna(50.0).values if "ces_pctl" in cand.columns else np.full(len(cand), 50.0),
        proc_disadv_i=proc_disadv,
    )


def group_mean_gain(inp: ScenarioInputs, mask: np.ndarray, selected_idx: np.ndarray) -> float:
    """Population-weighted mean surrogate gain of the tracts in `mask`."""
    pop = inp.pop_i[mask]
    return float((pop[:, None] * inp.g_ik[mask][:, selected_idx]).sum() / max(pop.sum(), 1e-12))


def best_attainable_gain(inp: ScenarioInputs, mask: np.ndarray, n_vertiports: int) -> float:
    """Largest mean surrogate gain any network of the given size can give the tracts in `mask`."""
    only = ScenarioInputs(
        g_ik=inp.g_ik * mask[:, None], b_ik=inp.b_ik, pop_i=inp.pop_i, income_i=inp.income_i,
        group_i=inp.group_i, transit_int_k=inp.transit_int_k, ces_pctl_k=inp.ces_pctl_k,
        proc_disadv_i=inp.proc_disadv_i,
    )
    res = solve_scenario("best_attainable", only, n_vertiports=n_vertiports, alpha=1.0)
    return group_mean_gain(inp, mask, res.selected_idx)


def _label_binding(res: ScenarioResult, reference: ScenarioResult) -> ScenarioResult:
    """Mark a constrained optimum that coincides with the unconstrained one."""
    if res.status.lower().startswith("optimal") and np.array_equal(res.selected_idx, reference.selected_idx):
        res.status = "Optimal (constraint not binding)"
    return res


def run_scenario(name: str, pre: dict, n_vertiports: int) -> ScenarioResult:
    inp = _build_inputs(pre)

    if name == "S0":
        return ScenarioResult(name="S0", selected=np.zeros(len(inp.transit_int_k), bool),
                              selected_idx=np.array([], dtype=int),
                              dA_i=np.zeros(len(inp.pop_i)),
                              B_i=np.zeros(len(inp.pop_i)),
                              objective=0.0, runtime_sec=0.0, status="N/A")

    if name == "S1":
        income = inp.income_i / max(np.nanmedian(inp.income_i), 1.0)
        inp_market = ScenarioInputs(
            g_ik=pre["g_ik"] * income.reshape(-1, 1),
            b_ik=pre["b_ik"], pop_i=inp.pop_i, income_i=inp.income_i,
            group_i=inp.group_i, transit_int_k=inp.transit_int_k,
            ces_pctl_k=inp.ces_pctl_k, proc_disadv_i=inp.proc_disadv_i,
        )
        return solve_scenario("S1", inp_market, n_vertiports=n_vertiports, alpha=1.0)

    unconstrained = solve_scenario(name if name in ("S2", "S6") else "S2", inp,
                                   n_vertiports=n_vertiports, alpha=1.0)
    if name in ("S2", "S6"):
        # S6 differs from S2 only through the subsidised fare in `pre`
        return unconstrained

    if name == "S3":
        low = np.asarray(inp.group_i) == "low_income"
        floor = CFG.opt.equity_floor_share * best_attainable_gain(inp, low, n_vertiports)
        res = solve_scenario("S3", inp, n_vertiports=n_vertiports, alpha=1.0,
                             per_group_floor={"low_income": floor})
        return _label_binding(res, unconstrained)

    if name == "S4":
        # Bonus scaled to the 99th-percentile gain so that it is material in the objective
        bonus_unit = float(np.average(inp.pop_i)) * float(np.nanpercentile(pre["g_ik"], 99))
        res = solve_scenario("S4", inp, n_vertiports=n_vertiports, alpha=1.0,
                             transit_bonus=bonus_unit, require_transit_share=0.8)
        return _label_binding(res, unconstrained)

    if name == "S5":
        peak = float(pre["b_ik"][:, unconstrained.selected_idx].sum(axis=1).max())
        cap = CFG.opt.burden_cap_share * peak
        res = solve_scenario("S5", inp, n_vertiports=n_vertiports, alpha=1.0,
                             per_tract_burden_cap=cap)
        return _label_binding(res, unconstrained)

    if name == "S7":
        proc = inp.proc_disadv_i.astype(bool)
        target = CFG.opt.procedural_coverage_share * best_attainable_gain(inp, proc, n_vertiports)
        res = solve_scenario("S7", inp, n_vertiports=n_vertiports, alpha=1.0,
                             procedural_min_coverage=target)
        return _label_binding(res, unconstrained)

    raise ValueError(f"Unknown scenario {name}")


def realise(pre: dict, selected_idx: np.ndarray, metro: str) -> tuple[np.ndarray, np.ndarray]:
    """Accessibility gain and burden of a selected network, recomputed from the cost model."""
    n = len(pre["tracts"])
    if selected_idx.size == 0:
        return np.zeros(n), np.zeros(n)
    aam_cost = compute_aam_cost_matrix(pre["tracts"], pre["candidates"], selected_idx, metro,
                                       pre["params_aam"], pre["fare"])
    A_aam = gravity_accessibility(np.minimum(pre["base_cost"], aam_cost), pre["jobs"])
    return np.maximum(A_aam - pre["A_base"], 0.0), pre["b_ik"][:, selected_idx].sum(axis=1)


# ----------------------------------------------------------------------------
# Run all
# ----------------------------------------------------------------------------
def run_all_scenarios(metro: str = "la",
                      n_vertiports: int = 25,
                      max_candidates: int = 80,
                      ops_per_day_per_vertiport: float = 80.0,
                      fare_tier: str = "long_run") -> pd.DataFrame:
    """Run S0–S7 at one fare tier and write results to results/<metro>/<fare_tier>/."""
    out_dir = RESULTS / metro / fare_tier
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates = build_vertiport_candidates(metro=metro, max_candidates=max_candidates)
    print(f"  {len(candidates)} candidate vertiports "
          f"(types: {candidates['seed_type'].value_counts().to_dict()})")

    # Two fare specifications: the tier's fare with tract-specific value of time, and
    # the same fare with the means-tested discount for low-income tracts (S6).
    tracts = load_tracts_joined(metro)
    low_income = assign_groups(metro).set_index("tract_geoid")["income_group"].reindex(
        tracts["tract_geoid"].values).values == "low_income"
    pre_main = precompute_contributions(
        metro, candidates, CFG.aam, CFG.burden, distance_fare(tracts, CFG.aam, fare_tier),
        ops_per_day_per_vertiport=ops_per_day_per_vertiport)
    pre_subsidy = precompute_contributions(
        metro, candidates, CFG.aam, CFG.burden,
        distance_fare(tracts, CFG.aam, fare_tier, discount=CFG.aam.subsidy_share * low_income),
        ops_per_day_per_vertiport=ops_per_day_per_vertiport)
    pre_for = {name: pre_main for name in SCENARIOS}
    pre_for["S6"] = pre_subsidy

    # Burden validation
    full_burden_default = pd.DataFrame({
        "tract_geoid": pre_main["tracts"]["tract_geoid"].values,
        "B": pre_main["b_ik"].sum(axis=1),
    })
    val = burden_validation_against_ces(full_burden_default, metro=metro)
    print(f"  burden×CES validation: {val}")
    (out_dir / "burden_validation.json").write_text(json.dumps(val, indent=2))

    summary_rows = []
    for name in SCENARIOS:
        print(f"\n— Scenario {name}: {SCENARIO_DESC[name]} —")
        pre = pre_for[name]
        res = run_scenario(name, pre, n_vertiports=n_vertiports)
        dA_real, B_real = realise(pre, res.selected_idx, metro)

        df = pd.DataFrame({
            "tract_geoid": pre["tracts"]["tract_geoid"].values,
            "dA": dA_real,         # POST-SOLVE realized accessibility gain
            "B":  B_real,
            "dA_surrogate": res.dA_i,   # keep surrogate for diagnostics / comparison
            "B_surrogate":  res.B_i,
        })
        equity = compute_full_equity_panel(df, pre["groups"])
        df.to_parquet(out_dir / f"scenario_{name}_per_tract.parquet")
        candidates.iloc[res.selected_idx].to_parquet(out_dir / f"scenario_{name}_selected.parquet")
        (out_dir / f"scenario_{name}_equity.json").write_text(json.dumps(equity, indent=2, default=float))

        summary_rows.append({
            "scenario": name,
            "fare_tier": fare_tier,
            "n_vertiports": int(res.selected.sum()),
            "objective": res.objective,
            "runtime_sec": res.runtime_sec,
            "status": res.status,
            "mean_dA":  equity.get("mean_dA"),
            "mean_dA_pct_of_base": 100.0 * float(np.average(dA_real, weights=np.maximum(
                pre["tracts"]["pop"].fillna(0).values, 0))) / float(np.average(
                pre["A_base"], weights=np.maximum(pre["tracts"]["pop"].fillna(0).values, 0))),
            "share_tracts_dA_pos": float((dA_real > 0).mean()),
            "mean_B":   equity.get("mean_B"),
            "gini_dA":  equity.get("gini_dA"),
            "theil_dA": equity.get("theil_dA"),
            "palma_dA": equity.get("palma_dA"),
            "ci_dA_vs_income": equity.get("ci_dA_vs_income"),
            "ci_B_vs_income":  equity.get("ci_B_vs_income"),
            "min_gain_value": (equity.get("min_gain") or {}).get("min_value"),
        })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "scenario_summary.csv", index=False)
    print(f"\nResults written to {out_dir}")
    return summary
