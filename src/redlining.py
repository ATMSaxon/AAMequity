"""Acoustic-redlining analysis — does AAM burden compound historical HOLC disparities?

Method (Casey/Nardone 2024 follow-up):
  1. Spatial-join 1930s HOLC polygons (A/B/C/D) against current LA tracts
  2. Assign each tract a primary HOLC grade by majority area overlap
  3. For each scenario's per-tract burden B_i, compute:
       (a) Mean B by HOLC grade (A < B < C < D expected if redlining compounds)
       (b) Fair Share Ratio (per Huang et al. 2023): share of HA / share of pop
       (c) Spearman ρ(B, holc_disadvantage_score) where D=4, C=3, B=2, A=1
  4. Repeat for ΔA — do D-graded tracts also lose out on accessibility benefit?
  5. EBR by HOLC grade — is the dual benefit-burden ratio worst in D tracts?
"""

from __future__ import annotations
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .config import RAW, PROC, RESULTS
from .data_layer import load_tracts_joined


HOLC_RANK = {"A": 1, "B": 2, "C": 3, "D": 4}


def load_holc(metro: str = "la") -> gpd.GeoDataFrame:
    path = RAW / "holc" / f"holc_{metro}.geojson"
    if not path.exists():
        raise FileNotFoundError(f"Run scripts/3_fetch_holc_redlining.py {metro}")
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    # Mapping-Inequality polygons have property 'holc_grade' or sometimes nested in 'properties.holc_grade'
    if "holc_grade" not in gdf.columns:
        # Try nested
        for col in ["holcGrade", "grade", "fillColor"]:
            if col in gdf.columns:
                gdf["holc_grade"] = gdf[col]
                break
    if "holc_grade" not in gdf.columns:
        raise KeyError(f"Could not find holc_grade in {path}; columns: {gdf.columns.tolist()[:10]}")
    gdf["holc_grade"] = gdf["holc_grade"].astype(str).str.upper().str[0]
    gdf = gdf[gdf["holc_grade"].isin(["A", "B", "C", "D"])].copy()
    gdf["holc_rank"] = gdf["holc_grade"].map(HOLC_RANK).astype(int)
    return gdf


def assign_holc_grade_to_tracts(metro: str = "la") -> pd.DataFrame:
    """Each tract gets its primary HOLC grade by majority area overlap."""
    out = PROC / f"tract_holc_{metro}.parquet"
    if out.exists():
        return pd.read_parquet(out)
    tracts = load_tracts_joined(metro)
    holc = load_holc(metro)
    # Reproject to a common metric CRS for area-weighted overlay
    tracts_m = tracts.to_crs("EPSG:3310")
    holc_m = holc.to_crs("EPSG:3310")
    # Use overlay to get per-(tract, holc_polygon) intersection areas
    inter = gpd.overlay(tracts_m[["tract_geoid", "geometry"]], holc_m[["holc_grade", "holc_rank", "geometry"]],
                         how="intersection", keep_geom_type=False)
    inter["area_m2"] = inter.geometry.area
    # Pick the HOLC grade with the largest overlap area per tract
    idx = inter.groupby("tract_geoid")["area_m2"].idxmax()
    pick = inter.loc[idx, ["tract_geoid", "holc_grade", "holc_rank", "area_m2"]].reset_index(drop=True)
    pick.to_parquet(out)
    return pick


def acoustic_redlining_test(metro: str = "la", scenarios: list = None,
                            fare_tier: str = "long_run") -> dict:
    """For each scenario of one fare tier, compute B and ΔA stratified by HOLC grade."""
    scenarios = scenarios or ["S1", "S2", "S3", "S4", "S5", "S6", "S7"]
    holc_tracts = assign_holc_grade_to_tracts(metro)
    tracts = load_tracts_joined(metro)
    base = tracts[["tract_geoid", "pop", "median_hh_income"]].merge(
        holc_tracts[["tract_geoid", "holc_grade", "holc_rank"]],
        on="tract_geoid", how="inner")
    print(f"  HOLC overlap: {len(base)} tracts of {len(tracts)} have HOLC grades")

    out = {"metro": metro, "fare_tier": fare_tier, "n_tracts_with_holc": int(len(base)),
            "by_grade_pop": base.groupby("holc_grade")["pop"].sum().to_dict(),
            "scenarios": {}}

    for s in scenarios:
        path = RESULTS / metro / fare_tier / f"scenario_{s}_per_tract.parquet"
        if not path.exists():
            continue
        per = pd.read_parquet(path)
        join = base.merge(per, on="tract_geoid", how="left").fillna(0)
        # Mean B and ΔA by HOLC grade (population-weighted)
        results = {}
        for grade, sub in join.groupby("holc_grade"):
            wsum = sub["pop"].sum()
            if wsum <= 0:
                continue
            results[grade] = {
                "mean_dA": float(np.average(sub["dA"], weights=sub["pop"])),
                "mean_B":  float(np.average(sub["B"],  weights=sub["pop"])),
                "n_tracts": int(len(sub)),
                "pop_share": float(wsum / join["pop"].sum()),
            }
        # Fair Share Ratio (Huang 2023): share of (population × B) / share of population
        total_burden_pop = (join["B"] * join["pop"]).sum()
        if total_burden_pop > 0:
            for grade in results:
                grade_burden_pop = ((join["holc_grade"] == grade) * join["B"] * join["pop"]).sum()
                results[grade]["fsr_burden"] = float(
                    (grade_burden_pop / total_burden_pop) / (results[grade]["pop_share"] + 1e-9))
        # FIX P10: Spearman over ALL HOLC tracts (not B>0 only) so we capture
        # the full geographic story including tracts unaffected by AAM. Restricting
        # to B>0 biased earlier results toward only-near-vertiport tracts.
        if len(join) > 50:
            rho_b, p_b = spearmanr(join["B"], join["holc_rank"])
            rho_da, p_da = spearmanr(join["dA"], join["holc_rank"])
            # Also keep B>0 subset as a separate diagnostic
            sub = join[join["B"] > 0]
            if len(sub) > 50:
                rho_b_pos, p_b_pos = spearmanr(sub["B"], sub["holc_rank"])
            else:
                rho_b_pos, p_b_pos = float("nan"), float("nan")
        else:
            rho_b = rho_da = float("nan"); p_b = p_da = float("nan")
            rho_b_pos, p_b_pos = float("nan"), float("nan")
        out["scenarios"][s] = {
            "by_grade": results,
            "spearman_B_vs_rank":  float(rho_b),       # FIX P10: all HOLC tracts
            "spearman_dA_vs_rank": float(rho_da),
            "p_B_vs_rank":  float(p_b),
            "p_dA_vs_rank": float(p_da),
            "spearman_B_vs_rank_Bpos_only": float(rho_b_pos),  # diagnostic
            "p_B_vs_rank_Bpos_only": float(p_b_pos),
        }
        print(f"  {s}: by-grade B = " + ", ".join(
            f"{g}={results[g]['mean_B']:.4f}" for g in sorted(results)))
    out_path = RESULTS / metro / fare_tier / "acoustic_redlining.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"  -> {out_path}")
    return out
