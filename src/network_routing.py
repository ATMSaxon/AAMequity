"""Network-based driving / walking travel times via OSMnx.

Replaces Euclidean × constant-speed approximation with real street-network
shortest-path times. Per-metro graph is downloaded once, cached, and re-used.

Computes:
  - tract-centroid → vertiport-candidate driving times (for AAM access leg)
  - tract-centroid → tract-centroid driving times (for base-mode O-D matrix)
  - Saves to data/processed/network_*.parquet

Computational cost: O(N_origins × N_destinations) Dijkstra calls. For 2476 LA
tracts that is ~6M shortest-path queries — slow but tractable in a one-time
batch (~30-60 min per metro). After caching, downstream analysis is instant.
"""

from __future__ import annotations
import time
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd

from .config import RAW, PROC, RESULTS, METRO_COUNTY_FIPS
from .data_layer import load_tracts_joined
from .candidates import _to_metric, _metric_crs_for


# Per-metro driving graph polygon (use the bounding box of joined tracts)
def metro_bbox(metro: str) -> tuple[float, float, float, float]:
    """Returns OSMnx bbox tuple (left, bottom, right, top) = (west, south, east, north)."""
    tracts = load_tracts_joined(metro)
    minx, miny, maxx, maxy = tracts.total_bounds
    return (float(minx), float(miny), float(maxx), float(maxy))


def download_or_load_drive_graph(metro: str) -> nx.MultiDiGraph:
    """Download (cached) the OSMnx driving graph for a SAMPLE of central metro.

    Full county is too slow; we use a 30km box around the metro centroid for
    calibration purposes. Sample-based calibration is sufficient to validate
    the Euclidean × 22 km/h × 1.6 congestion approximation.
    """
    cache_path = PROC / f"osmnx_drive_{metro}.graphml"
    if cache_path.exists():
        print(f"  loading cached graph: {cache_path}")
        return ox.load_graphml(cache_path)

    # Use a smaller central bbox for calibration (saves 10× time)
    tracts = load_tracts_joined(metro)
    minx, miny, maxx, maxy = tracts.total_bounds
    cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
    half = 0.30  # ~33 km half-width at LA latitude
    bbox = (cx - half, cy - half, cx + half, cy + half)  # (W, S, E, N)
    print(f"  downloading OSMnx drive graph for {metro} central bbox={bbox} ...")
    t0 = time.time()
    G = ox.graph_from_bbox(bbox=bbox, network_type="drive", simplify=True)
    print(f"    nodes={len(G.nodes)}, edges={len(G.edges)}, fetched in {time.time()-t0:.0f}s")
    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)
    print(f"  saving graph -> {cache_path}")
    ox.save_graphml(G, cache_path)
    return G


def network_drive_minutes(metro: str,
                            origins_xy: np.ndarray,
                            dests_xy: np.ndarray,
                            congestion_factor: float = 1.6) -> np.ndarray:
    """Returns (n_origin, n_dest) drive-time matrix in minutes via OSMnx graph.

    congestion_factor multiplies free-flow travel-time to model congested LA peak;
    1.6 ≈ TTI Urban Mobility Report congestion factor for LA.
    """
    G = download_or_load_drive_graph(metro)
    # Project tract coords (Albers EPSG:3310 / UTM) to graph CRS (lat/lon EPSG:4326)
    # Inputs are in metric CRS — convert back to lat/lon for nearest-node lookup
    epsg_m = _metric_crs_for(metro)
    origins_gdf = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(origins_xy[:, 0], origins_xy[:, 1]),
        crs=f"EPSG:{epsg_m}").to_crs("EPSG:4326")
    dests_gdf = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(dests_xy[:, 0], dests_xy[:, 1]),
        crs=f"EPSG:{epsg_m}").to_crs("EPSG:4326")
    o_lats, o_lons = origins_gdf.geometry.y.values, origins_gdf.geometry.x.values
    d_lats, d_lons = dests_gdf.geometry.y.values, dests_gdf.geometry.x.values
    print(f"  finding nearest network nodes ({len(o_lons)} origins, {len(d_lons)} dests)")
    o_nodes = ox.distance.nearest_nodes(G, X=o_lons, Y=o_lats)
    d_nodes = ox.distance.nearest_nodes(G, X=d_lons, Y=d_lats)
    # Unique destination nodes for batched Dijkstra
    n_o, n_d = len(o_nodes), len(d_nodes)
    out = np.full((n_o, n_d), np.inf)
    print(f"  running {n_o} single-source-shortest-paths Dijkstras ...")
    t0 = time.time()
    for i, src in enumerate(o_nodes):
        try:
            lengths = nx.single_source_dijkstra_path_length(
                G, src, weight="travel_time")
        except (nx.NetworkXError, KeyError):
            continue
        for j, dst in enumerate(d_nodes):
            t = lengths.get(dst)
            if t is not None:
                out[i, j] = t / 60.0  # seconds → minutes
        if (i + 1) % 200 == 0 or i + 1 == n_o:
            print(f"    [{i+1}/{n_o}]  elapsed {time.time()-t0:.0f}s")
    out = out * congestion_factor
    return out


def precompute_network_base_costs(metro: str = "la") -> Path:
    """Compute and cache tract-centroid × tract-centroid network drive times.
    WARNING: ~30-60 min per metro for full N×N matrix. Use sample-based calibration
    via `calibrate_speed_assumption` for faster validation.
    """
    out_path = PROC / f"network_base_cost_{metro}.npy"
    if out_path.exists():
        print(f"  cached: {out_path}")
        return out_path
    tracts = load_tracts_joined(metro)
    epsg_m = _metric_crs_for(metro)
    tm = _to_metric(tracts, epsg_m)
    coords = np.array([(g.x, g.y) for g in tm.geometry.centroid])
    cost = network_drive_minutes(metro, coords, coords, congestion_factor=1.6)
    np.save(out_path, cost)
    print(f"  saved {cost.shape} -> {out_path}")
    return out_path


def calibrate_speed_assumption(metro: str = "la", n_sample_pairs: int = 200,
                                 seed: int = 0) -> dict:
    """Sample-based calibration: compare OSMnx network drive times against our
    Euclidean × 22 km/h × congestion_factor approximation for a random subset of
    OD pairs. Reports OLS slope / intercept / RMSE so we can defend the
    Euclidean assumption.
    """
    import json
    from .accessibility import euclid_dist_m, base_travel_minutes_drive
    from scipy.stats import pearsonr, spearmanr
    out_path = RESULTS / metro / "network_calibration.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tracts = load_tracts_joined(metro)
    epsg_m = _metric_crs_for(metro)
    tm = _to_metric(tracts, epsg_m)
    coords = np.array([(g.x, g.y) for g in tm.geometry.centroid])
    n = len(coords)
    rng = np.random.default_rng(seed)
    pair_origins = rng.integers(0, n, size=n_sample_pairs)
    pair_dests = rng.integers(0, n, size=n_sample_pairs)
    # Restrict to non-self pairs with realistic distances 5-50 km
    dist_km = np.sqrt(((coords[pair_origins] - coords[pair_dests]) ** 2).sum(axis=1)) / 1000.0
    keep = (dist_km > 5.0) & (dist_km < 50.0)
    pair_origins, pair_dests, dist_km = pair_origins[keep], pair_dests[keep], dist_km[keep]
    if len(dist_km) < 30:
        return {"error": "too few valid sample pairs"}

    # Compute Euclidean approximation
    euclid_min = base_travel_minutes_drive(dist_km, speed_kmh=22.0)

    # Compute OSMnx network for the SAMPLE
    G = download_or_load_drive_graph(metro)
    o_xy = coords[pair_origins]
    d_xy = coords[pair_dests]
    o_gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(o_xy[:, 0], o_xy[:, 1]),
                              crs=f"EPSG:{epsg_m}").to_crs("EPSG:4326")
    d_gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(d_xy[:, 0], d_xy[:, 1]),
                              crs=f"EPSG:{epsg_m}").to_crs("EPSG:4326")
    o_nodes = ox.distance.nearest_nodes(G, X=o_gdf.geometry.x, Y=o_gdf.geometry.y)
    d_nodes = ox.distance.nearest_nodes(G, X=d_gdf.geometry.x, Y=d_gdf.geometry.y)

    network_min = []
    for src, dst in zip(o_nodes, d_nodes):
        try:
            sec = nx.shortest_path_length(G, src, dst, weight="travel_time")
            network_min.append(sec / 60.0)
        except (nx.NetworkXNoPath, nx.NodeNotFound, KeyError):
            network_min.append(np.nan)
    network_min = np.asarray(network_min)
    # Apply same congestion factor (1.6) for fairness
    network_min_congested = network_min * 1.6

    valid = np.isfinite(network_min_congested) & np.isfinite(euclid_min)
    if valid.sum() < 30:
        return {"error": "too few converged routes"}

    rho_p, _ = pearsonr(network_min_congested[valid], euclid_min[valid])
    rho_s, _ = spearmanr(network_min_congested[valid], euclid_min[valid])
    # OLS slope: network = a + b * euclid
    A = np.vstack([np.ones(valid.sum()), euclid_min[valid]]).T
    a, b = np.linalg.lstsq(A, network_min_congested[valid], rcond=None)[0]
    rmse = float(np.sqrt(np.mean((network_min_congested[valid] - euclid_min[valid]) ** 2)))
    bias = float(np.mean(network_min_congested[valid] - euclid_min[valid]))

    out = {
        "metro": metro,
        "n_pairs_attempted": int(len(dist_km)),
        "n_pairs_converged": int(valid.sum()),
        "euclid_22kmh_min_p50": float(np.median(euclid_min[valid])),
        "network_congested_min_p50": float(np.median(network_min_congested[valid])),
        "pearson_r": float(rho_p),
        "spearman_r": float(rho_s),
        "ols_intercept_a_min": float(a),
        "ols_slope_b": float(b),
        "rmse_min": rmse,
        "mean_bias_min_network_minus_euclid": bias,
        "interpretation": (
            "If pearson r > 0.85 and ols_slope_b in [0.8, 1.2] and rmse < 10 min, "
            "the Euclidean × 22 km/h × 1.6 approximation is defensible relative to "
            "OSMnx network shortest paths for the metro's central-trip range (5-50 km)."
        ),
    }
    out_path.write_text(json.dumps(out, indent=2))
    print(f"  calibration: pearson r={rho_p:.3f}, slope={b:.3f}, "
          f"rmse={rmse:.1f} min, bias={bias:+.1f} min")
    print(f"  -> {out_path}")
    return out
