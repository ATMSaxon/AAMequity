"""Multimodal accessibility — base (no AAM) and AAM-augmented.

Base accessibility A_i^base for each origin tract i:
    A_i^base = sum_j O_j * f(C_ij^base)
where C_ij^base is the generalized cost of the cheapest existing mode (drive or transit),
and f is a negative-exponential deterrence f(C) = exp(-beta * C / 60).

AAM accessibility A_i^AAM at each candidate vertiport network V (set of selected k):
    For each (i, j), AAM cost is the minimum across all vertiport pairs (k_o, k_d):
        C_ij^AAM(k_o, k_d) = T_access(i,k_o) + T_wait + T_security + T_flight(k_o,k_d)
                              + T_egress(k_d, j) + lambda * Fare
    The AAM-augmented effective cost is min(C_ij^base, min over (k_o,k_d) C_ij^AAM).

Outputs:
    A_i_base, A_i_aam, dA_i = A_i_aam - A_i_base, all per group.
"""

from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from .config import CFG, AAMParams, PROC
from .data_layer import load_tracts_joined, load_lodes_wac
from .candidates import _to_metric, _metric_crs_for


# ----------------------------------------------------------------------------
# Fares and value of time
# ----------------------------------------------------------------------------
@dataclass
class FareSpec:
    """Fare schedule and its conversion to generalised minutes.

    A flight of d km costs `base + per_km * d` dollars; a traveller from tract i
    converts dollars to minutes with `weight[i]` (min/$), which already includes
    any fare discount the tract receives.
    """
    base: float
    per_km: float
    weight: np.ndarray | float

    @classmethod
    def flat(cls, fare: float, minutes_per_dollar: float) -> "FareSpec":
        return cls(base=fare, per_km=0.0, weight=minutes_per_dollar)


def minutes_per_dollar_by_tract(tracts: gpd.GeoDataFrame, params: AAMParams) -> np.ndarray:
    """Fare-to-time weight of each tract from its median household income.

    Value of time is `vot_income_share` of hourly household income; tracts take
    the value of their income decile, with missing incomes set to the median.
    """
    income = tracts["median_hh_income"].astype(float)
    income = income.fillna(income.median()).values
    edges = np.quantile(income, np.linspace(0, 1, params.vot_income_bins + 1)[1:-1])
    decile = np.digitize(income, edges)
    rep = np.array([np.median(income[decile == d]) for d in range(params.vot_income_bins)])
    vot_per_hour = params.vot_income_share * rep[decile] / params.vot_hours_per_year
    return 60.0 / vot_per_hour


def distance_fare(tracts: gpd.GeoDataFrame, params: AAMParams, tier: str,
                  discount: np.ndarray | None = None) -> FareSpec:
    """Distance-based fare of a tier with tract-specific value of time.

    `discount` is the share of the fare each tract's residents do not pay.
    """
    weight = minutes_per_dollar_by_tract(tracts, params)
    if discount is not None:
        weight = weight * (1.0 - discount)
    return FareSpec(base=params.fare_base, per_km=params.fare_per_km[tier], weight=weight)


# ----------------------------------------------------------------------------
# Cost functions
# ----------------------------------------------------------------------------
def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    R = 6371.0088
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def euclid_dist_m(coords_a: np.ndarray, coords_b: np.ndarray) -> np.ndarray:
    """Pairwise distance matrix (m) between two arrays of (x, y)."""
    a = coords_a[:, None, :]
    b = coords_b[None, :, :]
    return np.sqrt(((a - b) ** 2).sum(axis=2))


def base_travel_minutes_drive(dist_km: np.ndarray, speed_kmh: float = 22.0) -> np.ndarray:
    """LA congested driving avg ~22 km/h (INRIX 2024 Global Traffic Scorecard for LA;
    Texas Transportation Institute Urban Mobility Report puts LA worst-in-US peak at ~18 mph
    = 29 km/h on uncongested freeways but ~13-14 mph = 21-22 km/h on congested arterials).
    Uses a single congested-period speed to model peak commute conditions.
    """
    return dist_km / speed_kmh * 60.0


def aam_flight_minutes(dist_km: np.ndarray, params: AAMParams) -> np.ndarray:
    """Flight time = climb + cruise + descent. We lump in pad/taxi separately."""
    flight = dist_km / params.cruise_speed_kmh * 60.0
    return flight + params.t_takeoff_min + params.t_landing_min + params.t_taxi_pad_min


def aam_access_minutes(dist_to_vertiport_m: np.ndarray,
                       walk_speed_kmh: float = 4.5,
                       drive_speed_kmh: float = 30.0) -> np.ndarray:
    """Access time: walk if ≤ 1 km, else drive (TNC) at congested-urban speed."""
    dist_km = dist_to_vertiport_m / 1000.0
    walk_min = dist_km / walk_speed_kmh * 60.0
    drive_min = dist_km / drive_speed_kmh * 60.0 + 4.0  # 4 min wait/staging
    return np.where(dist_km <= 1.0, walk_min, drive_min)


def aam_wait_minutes(params: AAMParams) -> float:
    if params.passenger_wait_min is not None:
        return params.passenger_wait_min
    rate_per_hr = params.ops_per_pad_per_hour * params.pads_per_vertiport
    return 30.0 / max(rate_per_hr, 1e-3)  # mean wait for Poisson with rate λ


# ----------------------------------------------------------------------------
# Core: build cost matrices over tracts (origins) and vertiport candidates
# ----------------------------------------------------------------------------
def compute_base_cost_matrix(tracts: gpd.GeoDataFrame,
                              dest_jobs: pd.DataFrame,
                              metro: str) -> tuple[np.ndarray, np.ndarray]:
    """Pairwise base generalized travel time (min) tracts → tracts.

    Returns:
        cost_min: (n_tracts, n_tracts) ground-mode minutes
        jobs:     (n_tracts,) job opportunities at each destination
    """
    tracts_m = _to_metric(tracts, _metric_crs_for(metro))
    coords = np.array([(g.x, g.y) for g in tracts_m.geometry.centroid])
    dist_m = euclid_dist_m(coords, coords)
    dist_km = dist_m / 1000.0

    if CFG.ground_time_source == "network":
        from .congestion import congested_minutes
        net = congested_minutes(metro)
        if not np.array_equal(net["tract_geoid"], tracts["tract_geoid"].values.astype(str)):
            raise ValueError(f"network OD matrix for {metro} does not match the tract layer")
        cost = net["tract_tract"].copy()
    else:
        cost = base_travel_minutes_drive(dist_km)
    # Self-cost = 0; also clip distances above max_trip_km to "infeasible"
    np.fill_diagonal(cost, 0.0)
    cost[dist_km > 200.0] = np.inf

    # Job opportunities: merge LODES WAC onto tracts
    j = tracts[["tract_geoid"]].merge(dest_jobs, left_on="tract_geoid",
                                       right_on="dest_tract", how="left").fillna(0.0)
    jobs = j["jobs_total"].values
    return cost, jobs


def compute_aam_cost_matrix(tracts: gpd.GeoDataFrame,
                             candidates: gpd.GeoDataFrame,
                             selected_idx: np.ndarray,
                             metro: str,
                             params: AAMParams,
                             fare: FareSpec) -> np.ndarray:
    """Per OD pair, minimum AAM generalised cost across all (k_o, k_d) pairs in the selected set.

    Returns:
        aam_cost_min: (n_tracts, n_tracts) generalized cost (min) of best AAM routing.
    """
    if len(selected_idx) == 0:
        return np.full((len(tracts), len(tracts)), np.inf)

    sel = candidates.iloc[selected_idx]
    epsg_m = _metric_crs_for(metro)
    tracts_m = _to_metric(tracts, epsg_m)
    sel_m = _to_metric(sel, epsg_m)
    t_xy = np.array([(g.x, g.y) for g in tracts_m.geometry.centroid])
    s_xy = np.array([(g.x, g.y) for g in sel_m.geometry])

    # Tract → vertiport distance (m)
    d_tv = euclid_dist_m(t_xy, s_xy)               # (n_tracts, k)
    d_vv_km = euclid_dist_m(s_xy, s_xy) / 1000.0   # (k, k)

    t_access = aam_access_minutes(d_tv)            # (n_tracts, k)
    if CFG.ground_time_source == "network":
        from .congestion import congested_minutes
        net = congested_minutes(metro)
        # Candidates at existing facilities are routed to directly; a candidate sited at a
        # tract centroid carries the id "T<tract_geoid>" and uses the tract-to-tract times.
        ids = sel["candidate_id"].astype(str).values
        col = pd.Index(net["candidate_id"]).get_indexer(ids)
        at_tract = pd.Index(net["tract_geoid"]).get_indexer([i[1:] for i in ids])
        if ((col < 0) & (at_tract < 0)).any():
            raise ValueError(f"candidates missing from the network OD matrix for {metro}")
        drive_min = np.where(col >= 0, net["tract_candidate"][:, np.maximum(col, 0)],
                             net["tract_tract_drive"][:, np.maximum(at_tract, 0)]) + 4.0   # 4 min wait/staging
        t_access = np.where(d_tv <= 1000.0, t_access, drive_min)
    t_egress = t_access                            # symmetric model
    time_kk = aam_flight_minutes(d_vv_km, params) + aam_wait_minutes(params) + params.t_security_min
    fare_kk = fare.base + fare.per_km * d_vv_km    # ($) by vertiport pair

    n = len(tracts)
    k = len(selected_idx)
    # C(i, j) = min_{ko, kd} t_access[i, ko] + time_kk[ko, kd] + w_i * fare_kk[ko, kd] + t_egress[j, kd]
    # Origins sharing a fare weight w share the best destination vertiport for each ko.
    weight = np.broadcast_to(np.asarray(fare.weight, dtype=float), (n,))
    levels, level_of = np.unique(weight, return_inverse=True)
    rows_of = [np.where(level_of == b)[0] for b in range(len(levels))]

    aam_cost = np.full((n, n), np.inf)
    for ko in range(k):
        for w, rows in zip(levels, rows_of):
            best_kd = (t_egress + time_kk[ko, :] + w * fare_kk[ko, :]).min(axis=1)   # (n_dest,)
            aam_cost[rows] = np.minimum(aam_cost[rows], t_access[rows, ko][:, None] + best_kd[None, :])

    # Filter trips too short (AAM not worth it) or too long (infeasible)
    dist_km = euclid_dist_m(t_xy, t_xy) / 1000.0
    invalid = (dist_km < params.min_trip_km) | (dist_km > params.max_trip_km)
    aam_cost[invalid] = np.inf
    return aam_cost


# ----------------------------------------------------------------------------
# Accessibility computation
# ----------------------------------------------------------------------------
def gravity_accessibility(cost_min: np.ndarray, opportunities: np.ndarray,
                          beta_per_hour: float = 1.0) -> np.ndarray:
    """A_i = sum_j O_j * exp(-beta_per_hour * C_ij/60)."""
    decay = np.exp(-beta_per_hour * cost_min / 60.0)
    decay[~np.isfinite(decay)] = 0.0
    return decay @ opportunities


def compute_base_accessibility(metro: str = "la") -> pd.DataFrame:
    """A_i^base for all tracts in the metro."""
    tracts = load_tracts_joined(metro)
    wac = load_lodes_wac(metro)
    cost, jobs = compute_base_cost_matrix(tracts, wac, metro)
    A_total = gravity_accessibility(cost, jobs)

    # Also compute by job-tier for income-segmented analyses
    j_high = (tracts[["tract_geoid"]].merge(wac[["dest_tract", "jobs_high_wage"]],
              left_on="tract_geoid", right_on="dest_tract", how="left")
              .fillna(0)["jobs_high_wage"].values)
    A_high = gravity_accessibility(cost, j_high)

    out = pd.DataFrame({
        "tract_geoid": tracts["tract_geoid"].values,
        "A_base_total": A_total,
        "A_base_high_wage": A_high,
    })
    return out


def compute_aam_accessibility(metro: str,
                              candidates: gpd.GeoDataFrame,
                              selected_idx: np.ndarray,
                              fare_tier: str,
                              params: AAMParams = None) -> pd.DataFrame:
    """For a given selection of vertiports, return per-tract A^AAM and ΔA."""
    params = params or CFG.aam
    tracts = load_tracts_joined(metro)
    fare = distance_fare(tracts, params, fare_tier)
    wac = load_lodes_wac(metro)

    base_cost, jobs = compute_base_cost_matrix(tracts, wac, metro)
    aam_cost = compute_aam_cost_matrix(tracts, candidates, selected_idx, metro, params, fare)

    # Effective cost = min of base and AAM
    eff_cost = np.minimum(base_cost, aam_cost)
    A_base = gravity_accessibility(base_cost, jobs)
    A_aam = gravity_accessibility(eff_cost, jobs)

    out = pd.DataFrame({
        "tract_geoid": tracts["tract_geoid"].values,
        "A_base_total": A_base,
        "A_aam_total": A_aam,
        "dA_total": A_aam - A_base,
        "dA_pct": (A_aam - A_base) / np.maximum(A_base, 1.0),
    })
    return out
