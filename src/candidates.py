"""Build vertiport candidate set from real infrastructure layers.

Candidates are derived from:
  - Existing FAA heliports & small airports (most likely first vertiports)
  - Major transit hubs (rail stations, high-frequency bus interchanges)
  - Major hospitals (medical-mission AAM use case)
  - Centroids of large parcels meeting land-use compatibility (when LA parcel data available)

Each candidate is annotated with airspace / land-use / infrastructure compatibility
and with proximity-based attributes used downstream by the optimizer.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from .config import RAW, PROC
from .data_layer import (
    load_tracts_joined, load_faa_heliports, load_hospitals, load_transit_stops,
)


def _to_metric(gdf: gpd.GeoDataFrame, epsg: int = 3310) -> gpd.GeoDataFrame:
    """California Albers (EPSG:3310) for accurate distance/area in CA;
    use 32614 (UTM 14N) for TX, 32618 (UTM 18N) for NY."""
    return gdf.to_crs(f"EPSG:{epsg}")


def _metric_crs_for(metro: str) -> int:
    return {
        "la": 3310, "sfbay": 3310,
        "dfw": 32614,
        "nyc": 32618,
    }[metro]


def build_vertiport_candidates(metro: str = "la",
                               max_candidates: int | None = None,
                               min_pad_separation_m: float = 1500.0,
                               refresh: bool = False) -> gpd.GeoDataFrame:
    """Build candidate vertiport locations from real infrastructure layers.

    Steps:
      1. Seed candidates: heliports + transit hubs + hospitals + airports
      2. Cluster nearby seeds (1.5 km min separation) to remove redundancy
      3. Annotate each candidate with: type, nearest_transit_dist, nearest_hospital_dist,
         containing_tract, county, CES_pctl, area land-use class (when available)
      4. Apply hard exclusion filters (airspace stub: outside major airport approach cones)
    """
    out = PROC / f"candidates_{metro}.parquet"
    if out.exists() and not refresh:
        return gpd.read_parquet(out)

    epsg_m = _metric_crs_for(metro)

    # ---- Seed pool ----
    seeds = []
    try:
        heli = load_faa_heliports(metro)
        heli["seed_type"] = heli["type"].fillna("heliport")
        seeds.append(heli[["seed_type", "tract_geoid", "county_fips", "geometry"]])
    except FileNotFoundError as e:
        print(f"  WARN: heliports unavailable: {e}")

    try:
        hosp = load_hospitals(metro)
        hosp["seed_type"] = "hospital"
        seeds.append(hosp[["seed_type", "tract_geoid", "county_fips", "geometry"]])
    except FileNotFoundError as e:
        print(f"  WARN: hospitals unavailable: {e}")

    try:
        ts = load_transit_stops(metro)
        # Major transit hubs only (rail or high-frequency bus)
        hubs = ts[ts["is_hub"]].copy()
        hubs["seed_type"] = np.where(hubs["is_rail"], "rail_hub", "bus_hub")
        seeds.append(hubs[["seed_type", "tract_geoid", "county_fips", "geometry"]])
    except FileNotFoundError as e:
        print(f"  WARN: transit stops unavailable: {e}")

    if not seeds:
        raise FileNotFoundError(
            "No infrastructure data available for candidate generation — "
            "run scripts/1_download_data.py")

    seed = pd.concat(seeds, ignore_index=True)
    seed = gpd.GeoDataFrame(seed, geometry="geometry", crs="EPSG:4326")

    # ---- Cluster within min_pad_separation_m ----
    seed_m = _to_metric(seed, epsg_m)
    coords = np.array([(g.x, g.y) for g in seed_m.geometry])
    keep = np.ones(len(seed_m), dtype=bool)
    # Greedy clustering: rail hubs > heliports > hospitals > bus hubs
    priority = seed["seed_type"].map(
        {"rail_hub": 0, "heliport": 1, "small_airport": 1, "hospital": 2, "bus_hub": 3, "seaplane_base": 1}
    ).fillna(4).values
    order = np.argsort(priority)
    selected_coords = []
    selected_idx = []
    for idx in order:
        if not keep[idx]:
            continue
        c = coords[idx]
        if any(np.hypot(c[0] - sc[0], c[1] - sc[1]) < min_pad_separation_m for sc in selected_coords):
            keep[idx] = False
            continue
        selected_coords.append(c)
        selected_idx.append(idx)

    cand = seed.iloc[selected_idx].copy().reset_index(drop=True)
    cand["candidate_id"] = "V" + cand.index.astype(str).str.zfill(4)

    # ---- Annotate: nearest transit, nearest hospital, containing tract / CES ----
    tracts = load_tracts_joined(metro)
    cand_m = _to_metric(cand, epsg_m)
    tracts_m = _to_metric(tracts, epsg_m)

    try:
        ts_all = load_transit_stops(metro)
        ts_m = _to_metric(ts_all, epsg_m)
        nearest = (gpd.sjoin_nearest(cand_m, ts_m[["geometry", "is_rail"]],
                                      how="left", distance_col="dist_to_transit_m")
                     .drop_duplicates(subset="candidate_id"))
        # Re-align by candidate_id to guarantee 1:1
        nearest = nearest.set_index("candidate_id").reindex(cand["candidate_id"])
        cand["dist_to_transit_m"] = nearest["dist_to_transit_m"].values
    except Exception as e:
        print(f"  WARN: nearest transit failed: {e}")
        cand["dist_to_transit_m"] = np.nan

    try:
        hosp_m = _to_metric(load_hospitals(metro), epsg_m)
        nearest = (gpd.sjoin_nearest(cand_m, hosp_m[["geometry"]],
                                      how="left", distance_col="dist_to_hospital_m")
                     .drop_duplicates(subset="candidate_id"))
        nearest = nearest.set_index("candidate_id").reindex(cand["candidate_id"])
        cand["dist_to_hospital_m"] = nearest["dist_to_hospital_m"].values
    except Exception as e:
        print(f"  WARN: nearest hospital failed: {e}")
        cand["dist_to_hospital_m"] = np.nan

    # CES of containing tract
    if "ces_pctl" in tracts.columns:
        cand = cand.merge(tracts[["tract_geoid", "ces_pctl", "median_hh_income",
                                  "pct_minority", "pct_zero_vehicle"]],
                          on="tract_geoid", how="left")

    # ---- Soft attribute: transit-integration bonus -----------------
    cand["transit_integrated"] = cand["dist_to_transit_m"].fillna(1e9) <= 800.0

    # ---- Hard exclusion filters (placeholder for airspace) ---------
    # Without FAA NASR airspace polygons, we exclude only seaplane bases and require
    # candidate to be inside a tract polygon (inland CA).
    cand = cand[cand["seed_type"] != "seaplane_base"].copy()
    cand = cand.dropna(subset=["tract_geoid"]).reset_index(drop=True)

    if max_candidates is not None and len(cand) > max_candidates:
        # FIX P9: HARMONIZED type quotas across all metros. If a metro lacks a type
        # (e.g. DFW has no rail_hub), the slot is automatically reallocated to the
        # next-priority type so total stays at max_candidates and pool comparability holds.
        type_quota_target = {
            "small_airport": int(max_candidates * 0.10),
            "heliport":      int(max_candidates * 0.30),
            "rail_hub":      int(max_candidates * 0.30),
            "hospital":      int(max_candidates * 0.15),
            "bus_hub":       int(max_candidates * 0.15),
        }
        # Reallocate missing-type slots to neighboring types for harmonization
        type_quota = {}
        deficit = 0
        for t, q in type_quota_target.items():
            available = (cand["seed_type"] == t).sum()
            if available == 0:
                deficit += q
                type_quota[t] = 0
            else:
                type_quota[t] = min(q, available)
        # Distribute deficit to types that have spare capacity, preferring transit hubs
        priority_order = ["bus_hub", "hospital", "heliport", "small_airport", "rail_hub"]
        for t in priority_order:
            if deficit <= 0: break
            avail = (cand["seed_type"] == t).sum() - type_quota.get(t, 0)
            add = min(avail, deficit)
            type_quota[t] = type_quota.get(t, 0) + add
            deficit -= add
        # Within a seed type, sites in lower-income tracts are kept first
        inc = cand["median_hh_income"] if "median_hh_income" in cand.columns else None
        cand["_sort_key"] = inc.fillna(inc.median()) if inc is not None else 0
        ascending = True
        chosen = []
        for t, q in type_quota.items():
            sub = cand[cand["seed_type"] == t]
            if len(sub) == 0 or q == 0:
                continue
            sub = sub.sort_values("_sort_key", ascending=ascending).head(q)
            chosen.append(sub)
        cand = pd.concat(chosen, ignore_index=True) if chosen else cand.head(max_candidates)
        if "_sort_key" in cand.columns:
            cand = cand.drop(columns=["_sort_key"])
        cand = cand.head(max_candidates).reset_index(drop=True)

    # Pre-compute lat/lon for downstream
    cand["lat"] = cand.geometry.y
    cand["lon"] = cand.geometry.x

    cand.to_parquet(out)
    print(f"  built {len(cand)} vertiport candidates  -> {out}")
    return cand


def build_greenfield_candidates(metro: str, n_sites: int = 100, seed: int = 0) -> gpd.GeoDataFrame:
    """Candidate sites placed by population coverage instead of at existing facilities.

    Sites are the tract centroids nearest to the centres of a population-weighted
    k-means partition of the metro. Their ids are "T<tract_geoid>".
    """
    from scipy.cluster.vq import kmeans2
    from scipy.spatial import cKDTree

    tracts = load_tracts_joined(metro)
    epsg_m = _metric_crs_for(metro)
    centroid = _to_metric(tracts, epsg_m).geometry.centroid
    xy = np.column_stack([centroid.x.values, centroid.y.values])
    pop = tracts["pop"].fillna(0).values.astype(float)
    rng = np.random.default_rng(seed)
    sample = xy[rng.choice(len(xy), size=20000, p=pop / pop.sum())]   # population-weighted points
    centres, _ = kmeans2(sample, n_sites, minit="++", seed=seed)
    _, nearest = cKDTree(xy).query(centres)
    idx = np.unique(nearest)
    cand = tracts.iloc[idx][["tract_geoid", "median_hh_income"]].copy()
    cand["geometry"] = gpd.GeoSeries(centroid.iloc[idx].values, crs=f"EPSG:{epsg_m}").to_crs(tracts.crs).values
    cand = gpd.GeoDataFrame(cand, geometry="geometry", crs=tracts.crs).reset_index(drop=True)
    cand["candidate_id"] = "T" + cand["tract_geoid"].astype(str)
    cand["seed_type"] = "greenfield"
    cand["transit_integrated"] = False
    cand["ces_pctl"] = (tracts["ces_pctl"].iloc[idx].values if "ces_pctl" in tracts.columns else np.nan)
    return cand
