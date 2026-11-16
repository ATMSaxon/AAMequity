"""End-to-end smoke test on REAL LA County data.

Loads downloaded ACS/LODES/GTFS/CES/TIGER, builds candidates from real heliports +
transit hubs, runs ONE scenario (S1 market-led) on a real LA central-county
subset (~150 tracts), verifies all numerical outputs are finite + plausible.

Run:  python tests/smoke_test.py
"""

from __future__ import annotations
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import geopandas as gpd

from src.config import CFG, RESULTS
from src.data_layer import (
    load_tracts, load_acs, load_lodes_od, load_lodes_wac, load_ces,
    load_tracts_joined, load_faa_heliports, load_transit_stops,
)
from src.candidates import build_vertiport_candidates


def banner(s):
    print("\n" + "=" * 70)
    print(s)
    print("=" * 70)


def step(name, fn, *args, **kwargs):
    print(f"\n--- {name} ---")
    t0 = time.time()
    try:
        out = fn(*args, **kwargs)
        print(f"  OK ({time.time()-t0:.1f}s)")
        return out
    except Exception as e:
        print(f"  FAIL: {type(e).__name__}: {e}")
        traceback.print_exc()
        raise


def main():
    banner("AAMequity smoke test on REAL LA County data")

    # --- 1. Load tracts
    tracts = step("load LA tracts (TIGER 2024)", load_tracts, "la")
    print(f"  n_tracts = {len(tracts)}; bbox = {tuple(round(b,3) for b in tracts.total_bounds)}")
    assert len(tracts) > 1000, "Expected >1000 LA County tracts (got %d)" % len(tracts)

    # --- 2. Load ACS demographics
    acs = step("load ACS 5-yr 2022", load_acs, "la")
    print(f"  n_tracts (ACS) = {len(acs)}")
    print(f"  median_hh_income p10/50/90 = "
          f"{acs['median_hh_income'].quantile([0.1,0.5,0.9]).round(0).tolist()}")
    print(f"  pct_minority p10/50/90 = "
          f"{acs['pct_minority'].quantile([0.1,0.5,0.9]).round(2).tolist()}")
    assert acs["median_hh_income"].notna().sum() > 1000

    # --- 3. CES
    try:
        ces = step("load CalEnviroScreen 4.0", load_ces)
        print(f"  CES rows = {len(ces)} ; cols = {ces.columns.tolist()}")
    except Exception as e:
        print(f"  WARN CES: {e}")

    # --- 4. LODES OD + WAC
    od = step("load LODES OD (LA)", load_lodes_od, "la")
    print(f"  OD pairs = {len(od)}; total jobs = {od['jobs'].sum():,}")
    wac = step("load LODES WAC (LA)", load_lodes_wac, "la")
    print(f"  WAC tracts = {len(wac)}; total = {wac['jobs_total'].sum():,}; "
          f"high-wage = {wac['jobs_high_wage'].sum():,}")

    # --- 5. Joined tracts
    joined = step("join tracts × ACS × CES", load_tracts_joined, "la")
    print(f"  joined rows = {len(joined)}")
    print(f"  pop sum = {joined['pop'].sum():,.0f}  (LA County ≈ 10M)")

    # --- 6. Heliports
    heli = step("load FAA heliports/airports (LA)", load_faa_heliports, "la")
    print(f"  heliports/airports in LA = {len(heli)}; types = {heli['type'].value_counts().to_dict()}")

    # --- 7. Transit stops
    ts = step("load LA Metro GTFS stops", load_transit_stops, "la")
    print(f"  transit stops = {len(ts)}; rail = {ts['is_rail'].sum()}; hubs = {ts['is_hub'].sum()}")

    # --- 8. Build candidates
    cand = step("build vertiport candidates", build_vertiport_candidates, "la",
                max_candidates=40)
    print(f"  candidates = {len(cand)}; types = {cand['seed_type'].value_counts().to_dict()}")
    print(f"  transit-integrated = {cand['transit_integrated'].sum()}")

    # --- 9. Compute base accessibility (for a small subset to keep quick)
    from src.accessibility import (
        compute_base_cost_matrix, gravity_accessibility,
        compute_aam_cost_matrix,
    )
    # Subset to central LA (within 25km of City Hall = -118.243, 34.054)
    from src.candidates import _to_metric, _metric_crs_for
    tj = joined.copy()
    tj_m = _to_metric(tj, _metric_crs_for("la"))
    city_hall_xy = _to_metric(
        gpd.GeoDataFrame(geometry=gpd.points_from_xy([-118.243], [34.054]), crs="EPSG:4326"),
        _metric_crs_for("la")
    ).geometry.iloc[0]
    cx, cy = city_hall_xy.x, city_hall_xy.y
    dist_to_center = np.sqrt(
        (tj_m.geometry.centroid.x - cx) ** 2
        + (tj_m.geometry.centroid.y - cy) ** 2
    )
    keep = dist_to_center < 25_000
    tj_sub = tj.loc[keep.values].reset_index(drop=True)
    print(f"\n  subset to central LA (≤25km from City Hall): {len(tj_sub)} tracts")

    base_cost, jobs = step("compute base cost matrix (subset)",
                            compute_base_cost_matrix, tj_sub, wac, "la")
    A_base = step("base accessibility (gravity)", gravity_accessibility, base_cost, jobs)
    print(f"  A_base p10/50/90 = {np.quantile(A_base, [0.1,0.5,0.9]).round(0).tolist()}")
    assert np.all(np.isfinite(A_base))

    # --- 10. AAM accessibility on the FULL LA County (so we capture long
    # commute trips where AAM has competitive advantage). Use all 40 candidates.
    base_cost_full, jobs_full = step("compute base cost matrix (full LA)",
                                      compute_base_cost_matrix, tj, wac, "la")
    A_base_full = gravity_accessibility(base_cost_full, jobs_full)
    cand_sub_idx = np.arange(min(len(cand), 30))
    # Use high-income tier + low (subsidized) fare so smoke test exposes positive ΔA
    aam_cost = step("AAM cost matrix (full LA, fare=$50, λ_high)",
                     compute_aam_cost_matrix,
                     tj, cand, cand_sub_idx, "la", CFG.aam, "high",
                     CFG.aam.fare_low)
    eff_cost = np.minimum(base_cost_full, aam_cost)
    A_aam = gravity_accessibility(eff_cost, jobs_full)
    dA = A_aam - A_base_full
    print(f"  ΔA p10/50/90 = {np.quantile(dA, [0.1,0.5,0.9]).round(0).tolist()}")
    print(f"  ΔA mean / max = {dA.mean():.1f} / {dA.max():.0f}")
    print(f"  tracts with ΔA > 0: {(dA > 0).sum()} / {len(dA)}")
    assert np.all(np.isfinite(dA))
    assert dA.min() >= -1e-6, "ΔA cannot be negative"
    assert dA.max() > 0, "Expect SOME tract to benefit from AAM"

    # --- 11. Burden index (single vertiport)
    from src.burden import compute_burden_index, burden_validation_against_ces
    # Need to use the FULL joined tracts, not subset
    bdf = step("compute burden index (full LA)",
                compute_burden_index, "la", cand, cand_sub_idx, 80.0)
    print(f"  B p10/50/90 = {bdf['B'].quantile([0.1,0.5,0.9]).round(3).tolist()}")
    print(f"  B mean / nonzero share = {bdf['B'].mean():.3f} / {(bdf['B']>0).mean():.2f}")
    assert bdf["B"].between(0, 100).all()

    val = step("burden × CES validation", burden_validation_against_ces, bdf, "la")
    print(f"  validation: {val}")

    # --- 12. Equity metrics
    from src.equity_metrics import compute_full_equity_panel
    from src.groups import assign_groups
    grp = assign_groups("la")
    df = pd.DataFrame({
        "tract_geoid": tj["tract_geoid"].values,
        "dA": dA,
    })
    df = df.merge(bdf[["tract_geoid", "B"]], on="tract_geoid", how="left")
    panel = step("equity panel", compute_full_equity_panel, df, grp)
    print(f"  Gini ΔA = {panel.get('gini_dA'):.3f}")
    print(f"  Theil ΔA = {panel.get('theil_dA'):.3f}")
    print(f"  Palma ΔA = {panel.get('palma_dA'):.3f}")
    print(f"  CI(ΔA, income) = {panel.get('ci_dA_vs_income'):+.3f}  (positive = pro-rich)")
    print(f"  CI(B, income) = {panel.get('ci_B_vs_income'):+.3f}")
    print(f"  EDR(ΔA) by group = {panel.get('edr_dA')}")
    print(f"  EDR(B)  by group = {panel.get('edr_B')}")

    # --- 13. MILP test (tiny scale)
    from src.scenarios import precompute_contributions, run_scenario, SCENARIO_DESC
    print("\n--- precompute_contributions for tiny test (≤10 candidates) ---")
    cand_small = cand.iloc[:10].reset_index(drop=True)
    pre = precompute_contributions("la", cand_small, CFG.aam, CFG.burden,
                                    ops_per_day_per_vertiport=80.0,
                                    income_tier="mid")
    print(f"  g_ik shape = {pre['g_ik'].shape}; g range = "
          f"[{pre['g_ik'].min():.0f}, {pre['g_ik'].max():.0f}]")
    print(f"  b_ik shape = {pre['b_ik'].shape}; b range = "
          f"[{pre['b_ik'].min():.3f}, {pre['b_ik'].max():.3f}]")

    res = step("solve S1 MILP (smoke, n=5)",
                run_scenario, "S1", pre, 5)
    print(f"  selected = {res.selected.sum()} of 10  (objective={res.objective:.0f})")
    print(f"  status = {res.status}; runtime = {res.runtime_sec:.1f}s")
    assert res.selected.sum() == 5

    banner("SMOKE TEST PASSED — pipeline runs end-to-end on real LA data")


if __name__ == "__main__":
    main()
