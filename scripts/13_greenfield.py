"""Greenfield candidate pool: sites placed by population coverage, not at existing facilities.

Pools of K sites are placed at tract centroids chosen by population-weighted
k-means over the metro, so that candidate geography follows where people live
rather than where heliports, stations and hospitals already are. The
accessibility-maximising network (S2) is then solved from this pool and compared
with the network selected from the existing-facility pool.

Usage:  python scripts/13_greenfield.py <metro>
Output: results/revision/greenfield_<metro>.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.accessibility import distance_fare
from src.burden import tract_vertiport_distance_m, vertiport_dnl_dba
from src.candidates import build_greenfield_candidates, build_vertiport_candidates
from src.config import CFG, RESULTS
from src.data_layer import load_tracts_joined
from src.equity_metrics import concentration_index
from src.redlining import assign_holc_grade_to_tracts
from src.scenarios import _build_inputs, precompute_contributions, realise
from src.optimizer import solve_scenario

N_VP = 25
MAX_CAND = 50
GREENFIELD_SIZES = (49, 100)      # the size of the existing-facility pool, and a larger pool
OPS_PER_DAY = 80.0
TIERS = ("long_run", "public_service")


def evaluate(metro, tracts, candidates, tier, holc_rank, has_holc) -> dict:
    pop = tracts["pop"].fillna(0).values.astype(float)
    income = tracts["median_hh_income"].fillna(tracts["median_hh_income"].median()).values
    pre = precompute_contributions(metro, candidates, CFG.aam, CFG.burden,
                                   distance_fare(tracts, CFG.aam, tier), ops_per_day_per_vertiport=OPS_PER_DAY)
    res = solve_scenario("S2", _build_inputs(pre), n_vertiports=N_VP, alpha=1.0)
    dA, B = realise(pre, res.selected_idx, metro)
    d = tract_vertiport_distance_m(tracts, candidates, res.selected_idx, metro)
    dnl = 10 * np.log10((10 ** (vertiport_dnl_dba(d, OPS_PER_DAY, CFG.burden) / 10)).sum(axis=1))
    host = candidates.iloc[res.selected_idx]["tract_geoid"].astype(str)
    site_income = tracts.set_index("tract_geoid")["median_hh_income"].astype(float).reindex(host)
    return {
        "n_candidates": int(len(candidates)), "status": res.status,
        "mean_dA_pct_of_base": 100 * float(np.average(dA, weights=pop) / np.average(pre["A_base"], weights=pop)),
        "share_tracts_dA_pos": float((dA > 0).mean()),
        "ci_dA_vs_income": concentration_index(dA, income, pop),
        "mean_B": float(np.average(B, weights=pop)),
        "ci_B_vs_income": concentration_index(B, income, pop),
        "ci_dnl_energy_vs_income": concentration_index(10 ** (dnl / 10), income, pop),
        "population_dnl_ge_40": float(pop[dnl >= 40].sum()),
        "rho_dnl_vs_holc": float(spearmanr(dnl[has_holc], holc_rank)[0]),
        "median_income_of_host_tracts": float(np.nanmedian(site_income)),
        "metro_median_tract_income": float(np.nanmedian(income)),
    }


def main():
    metro = sys.argv[1]
    assert CFG.ground_time_source == "network"
    tracts = load_tracts_joined(metro)
    holc = assign_holc_grade_to_tracts(metro).set_index("tract_geoid").reindex(tracts["tract_geoid"].values)
    has_holc = holc["holc_rank"].notna().values
    holc_rank = holc["holc_rank"].values[has_holc].astype(float)

    pools = {"existing_facilities": build_vertiport_candidates(metro=metro, max_candidates=MAX_CAND)}
    pools.update({f"greenfield_{k}": build_greenfield_candidates(metro, k) for k in GREENFIELD_SIZES})
    out = {"metro": metro, "tiers": {}}
    for tier in TIERS:
        out["tiers"][tier] = {name: evaluate(metro, tracts, cand, tier, holc_rank, has_holc)
                              for name, cand in pools.items()}
        print(f"[{metro}] {tier}: " + " | ".join(
            f"{name}: dA={r['mean_dA_pct_of_base']:.4f}% CI={r['ci_dA_vs_income']:+.3f} CI_B={r['ci_B_vs_income']:+.3f}"
            for name, r in out["tiers"][tier].items()), flush=True)

    out_dir = RESULTS / "revision"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"greenfield_{metro}.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"{metro}: done", flush=True)


if __name__ == "__main__":
    main()
