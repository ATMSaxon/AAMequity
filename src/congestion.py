"""Congested door-to-door driving times calibrated to observed commutes.

Free-flow network times from `network_od` are mapped to door-to-door peak times by

    T_ij = tau + sum_c kappa_c * m_ijc,

where m_ijc is the free-flow time the path from i to j spends on roads of
congestion class c (area type x freeway/surface street), kappa_c is the
travel-time multiplier of that class relative to posted speeds, and tau is a
terminal time. For each metro
the multipliers, tau and a lognormal dispersion sigma are estimated by maximum
likelihood: car-commute flows assign every home tract a distribution of T_ij,
and that distribution is fitted to the ACS travel-time bins reported by car
commuters living in the tract (table B08134). Car-commute flows are LODES
home-work flows rebalanced by iterative proportional fitting to the car share of
commuters at each home tract (B08134) and at each workplace county (B08406).
Paths are then re-routed on congested edge times and the fit is repeated until
the multipliers settle.

Run:  python -m src.congestion <metro> [n_workers]
Output: results/<metro>/congestion_calibration.json
"""
from __future__ import annotations

import json
import sys
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import ndtr

from .config import CFG, METRO_COUNTY_FIPS, RAW, RESULTS
from .network_od import N_CLASS, build_od, od_path

STATE_TO_LODES = {"06": "ca", "48": "tx", "36": "ny", "34": "nj", "09": "ct"}
BIN_EDGES = np.array([10.0, 15, 20, 25, 30, 35, 45, 60])      # upper edges; last bin is 60+
BIN_MIDPOINTS = np.array([5, 12.5, 17.5, 22.5, 27.5, 32.5, 40, 52.5, 75.0])
CAR_TOTAL_COL = "B08134_E011"
CAR_BIN_COLS = [f"B08134_E{c:03d}" for c in range(12, 21)]
ALL_TOTAL_COL = "B08134_E001"
KAPPA_BOUNDS = (0.7, 8.0)      # below 1: traffic moves faster than the posted or imputed speed
MIN_CLASS_SHARE = 0.02         # classes carrying less commute time are pooled with a neighbour
MAX_REROUTES = 4
REROUTE_TOL = 0.05


def calibration_path(metro: str):
    return RESULTS / metro / "congestion_calibration.json"


def load_car_commute_bins(tract_geoid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Car-commuter counts by travel-time bin and the car share of commuters."""
    cols = ["GEO_ID", ALL_TOTAL_COL, CAR_TOTAL_COL] + CAR_BIN_COLS
    df = pd.read_csv(RAW / "acs_commute" / "acs5_2022_b08134_tract_5states.psv",
                     sep="|", usecols=cols)
    df["tract_geoid"] = df["GEO_ID"].str[-11:]
    df = df.set_index("tract_geoid").reindex(tract_geoid)
    bins = df[CAR_BIN_COLS].clip(lower=0).fillna(0).values.astype(float)
    share = (df[CAR_TOTAL_COL].clip(lower=0) / df[ALL_TOTAL_COL].clip(lower=1)).fillna(0).values
    return bins, share


def load_flows(metro: str, tract_geoid: np.ndarray) -> np.ndarray:
    """Home-tract by work-tract job flows, including cross-state commutes."""
    counties = set(METRO_COUNTY_FIPS[metro])
    index = {t: i for i, t in enumerate(tract_geoid)}
    n = len(tract_geoid)
    flows = np.zeros((n, n))
    for st in {STATE_TO_LODES[c[:2]] for c in counties}:
        for part in ("main", "aux"):
            path = RAW / "lodes" / f"{st}_od_{part}_JT00_2021.csv.gz"
            for chunk in pd.read_csv(path, usecols=["w_geocode", "h_geocode", "S000"],
                                     dtype={"w_geocode": str, "h_geocode": str},
                                     chunksize=2_000_000):
                keep = chunk["h_geocode"].str[:5].isin(counties) & chunk["w_geocode"].str[:5].isin(counties)
                chunk = chunk[keep]
                h = chunk["h_geocode"].str[:11].map(index)
                w = chunk["w_geocode"].str[:11].map(index)
                ok = h.notna() & w.notna()
                np.add.at(flows, (h[ok].astype(int).values, w[ok].astype(int).values),
                          chunk.loc[ok, "S000"].values)
    return flows


def car_flows(flows: np.ndarray, tract_geoid: np.ndarray, home_car_share: np.ndarray,
              n_iter: int = 100) -> np.ndarray:
    """Rebalance all-mode flows to car commutes by home tract and workplace county."""
    county = np.array([g[:5] for g in tract_geoid])
    wp = pd.read_csv(RAW / "acs_commute" / "acs5_2022_b08406_county_5states.psv", sep="|",
                     usecols=["GEO_ID", "B08406_E001", "B08406_E002"])
    wp["county"] = wp["GEO_ID"].str[-5:]
    work_share = (wp.set_index("county")["B08406_E002"] / wp.set_index("county")["B08406_E001"])
    codes, col_group = np.unique(county, return_inverse=True)
    group = np.zeros((len(county), len(codes)))
    group[np.arange(len(county)), col_group] = 1.0

    row_target = flows.sum(axis=1) * home_car_share
    col_target = (flows.sum(axis=0) @ group) * work_share.reindex(codes).values
    col_target *= row_target.sum() / col_target.sum()

    out = flows.copy()
    for _ in range(n_iter):
        out *= (row_target / np.maximum(out.sum(axis=1), 1e-12))[:, None]
        col_factor = col_target / np.maximum(out.sum(axis=0) @ group, 1e-12)
        out *= col_factor[col_group][None, :]
    return out


def class_minutes(metro: str) -> dict:
    """Free-flow minutes by congestion class, with intrazonal trips filled in."""
    od = np.load(od_path(metro), allow_pickle=False)
    tt = od["tt_class_min"].astype(np.float64)
    n = tt.shape[0]
    total = tt.sum(axis=2)
    off = np.where(np.eye(n, dtype=bool), np.inf, total)
    nearest = off.min(axis=1)
    # Intrazonal trips: half the time to the nearest other tract, on local surface streets
    tt[np.arange(n), np.arange(n), :] = 0.0
    tt[np.arange(n), np.arange(n), 2 * od["tract_area_type"].astype(int)] = 0.5 * nearest
    return {"tract_geoid": od["tract_geoid"], "candidate_id": od["candidate_id"],
            "tt": tt, "tc": od["tc_class_min"].astype(np.float64), "tt_km": od["tt_km"].astype(np.float64)}


def pooled_classes(exposure: np.ndarray) -> np.ndarray:
    """Parameter index of each class after pooling thinly observed classes.

    A class carrying less than MIN_CLASS_SHARE of commute minutes shares its
    multiplier with the same road type in the next lower-density area type (or
    the next higher one when it is the lowest).
    """
    share = exposure / exposure.sum()
    owner = np.arange(N_CLASS)
    for c in range(N_CLASS - 1, -1, -1):
        if share[c] >= MIN_CLASS_SHARE:
            continue
        lower = [k for k in range(c - 2, -1, -2) if share[k] >= MIN_CLASS_SHARE]
        upper = [k for k in range(c + 2, N_CLASS, 2) if share[k] >= MIN_CLASS_SHARE]
        if lower or upper:
            owner[c] = (lower or upper)[0]
    return owner


def _fit(row, weight, minutes, bins, owner):
    """Maximum-likelihood tau, kappa and sigma for car commutes on given paths.

    row, weight, minutes describe the non-zero car flows: home-tract index, the
    flow's share of that tract's car commuters, and free-flow minutes by class.
    `owner` maps every class to the class whose multiplier it uses.
    """
    n_home = bins.shape[0]
    log_edges = np.log(BIN_EDGES)

    def probabilities(theta):
        tau, kappa, sigma = theta[0], theta[1:1 + N_CLASS][owner], theta[-1]
        logT = np.log(np.maximum(tau + minutes @ kappa, 1e-3))
        cdf = np.column_stack([np.bincount(row, weight * ndtr((e - logT) / sigma), minlength=n_home)
                               for e in log_edges])
        cdf = np.column_stack([np.zeros(n_home), cdf, np.ones(n_home)])
        return np.clip(np.diff(cdf, axis=1), 1e-9, None)

    def nll(theta):
        return -(bins * np.log(probabilities(theta))).sum()

    bounds = [(0.0, 30.0)] + [KAPPA_BOUNDS] * N_CLASS + [(0.05, 1.5)]
    start = np.array([5.0] + [1.5] * N_CLASS + [0.4])
    res = minimize(nll, start, method="L-BFGS-B", bounds=bounds)
    return res, probabilities(res.x), bounds


def calibrate(metro: str, n_workers: int = 16) -> dict:
    build_od(metro, n_workers)
    geoid = np.load(od_path(metro), allow_pickle=False)["tract_geoid"]
    bins, car_share = load_car_commute_bins(geoid)
    flows = car_flows(load_flows(metro, geoid), geoid, car_share)

    history, kappa_prev = [], np.ones(N_CLASS)
    for reroute in range(MAX_REROUTES + 1):
        comp = class_minutes(metro)
        reachable = np.isfinite(comp["tt"]).all(axis=2)
        fl = np.where(reachable, flows, 0.0)
        use = (bins.sum(axis=1) > 0) & (fl.sum(axis=1) > 0)
        home = np.cumsum(use) - 1
        i, j = np.nonzero(fl * use[:, None])
        weight = fl[i, j] / fl.sum(axis=1)[i]
        minutes = comp["tt"][i, j]
        b = bins[use]

        exposure = (weight[:, None] * minutes * b.sum(axis=1)[home[i], None]).sum(axis=0)
        owner = pooled_classes(exposure)
        res, p, bounds = _fit(home[i], weight, minutes, b, owner)
        tau, kappa, sigma = res.x[0], res.x[1:1 + N_CLASS][owner], res.x[-1]
        history.append({"tau_min": float(tau), "kappa": kappa.tolist(), "neg_loglik": float(res.fun)})
        change = float(np.max(np.abs(kappa - kappa_prev) / kappa_prev))
        print(f"  {metro} pass {reroute}: tau={tau:.1f} kappa={np.round(kappa, 2).tolist()} "
              f"sigma={sigma:.2f} max change={change:.3f}", flush=True)
        if reroute > 0 and change < REROUTE_TOL:
            break
        if reroute < MAX_REROUTES:
            kappa_prev = kappa
            build_od(metro, n_workers, kappa=kappa)

    n_car = b.sum(axis=1)
    obs_mean = (b * BIN_MIDPOINTS).sum(axis=1) / n_car
    pred_mean = (p * BIN_MIDPOINTS).sum(axis=1)
    T = tau + minutes @ kappa
    km = comp["tt_km"][i, j]
    moving = np.isfinite(km) & (km > 0)
    out = {
        "metro": metro,
        "tau_min": float(tau), "kappa": kappa.tolist(), "sigma_log": float(sigma),
        "class_order": "index = 2 * area_type + freeway; area types split at activity density "
                       "1000 / 5000 / 20000 per km2",
        "class_share_of_commute_minutes": (exposure / exposure.sum()).tolist(),
        "class_pooled_with": owner.tolist(),
        "neg_loglik": float(res.fun), "converged": bool(res.success),
        "kappa_at_bound": [bool(np.isclose(k, KAPPA_BOUNDS[0]) or np.isclose(k, KAPPA_BOUNDS[1])) for k in kappa],
        "reroute_passes": len(history) - 1, "history": history,
        "n_tracts_used": int(use.sum()), "n_car_commuters": float(n_car.sum()),
        "bin_upper_edges_min": BIN_EDGES.tolist() + [None],
        "observed_bin_share": (b.sum(axis=0) / b.sum()).tolist(),
        "predicted_bin_share": ((p * n_car[:, None]).sum(axis=0) / n_car.sum()).tolist(),
        "tract_mean_time_pearson_r": float(np.corrcoef(obs_mean, pred_mean)[0, 1]),
        "observed_mean_min": float(np.average(obs_mean, weights=n_car)),
        "predicted_mean_min": float(np.average(pred_mean, weights=n_car)),
        "flow_weighted_speed_kmh": float((weight[moving] * km[moving]).sum()
                                         / (weight[moving] * T[moving]).sum() * 60.0),
    }
    path = calibration_path(metro)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    return out


def congested_minutes(metro: str) -> dict:
    """Calibrated tract->tract minutes (door-to-door, and driving only) and tract->candidate drive minutes."""
    return _congested_minutes(metro, CFG.congestion_multiplier_scale, CFG.terminal_time_shift_min)


@lru_cache(maxsize=4)
def _congested_minutes(metro: str, multiplier_scale: float, terminal_shift: float) -> dict:
    c = json.loads(calibration_path(metro).read_text())
    comp = class_minutes(metro)
    kappa = 1.0 + multiplier_scale * (np.array(c["kappa"]) - 1.0)
    drive = comp["tt"] @ kappa
    np.fill_diagonal(drive, 0.0)
    tt = max(c["tau_min"] + terminal_shift, 0.0) + drive
    np.fill_diagonal(tt, 0.0)
    return {"tract_geoid": comp["tract_geoid"], "candidate_id": comp["candidate_id"],
            "tract_tract": tt, "tract_tract_drive": drive, "tract_candidate": comp["tc"] @ kappa}


def cross_validate(metro: str, n_folds: int = 5, seed: int = 0) -> dict:
    """Out-of-sample fit of the congestion model on the final paths.

    Home tracts are split into folds; the model is fitted on the other folds and
    evaluated on the held-out tracts by the correlation and mean absolute error
    of predicted against observed mean commute times, and by the largest gap
    between predicted and observed travel-time-bin shares.
    """
    geoid = np.load(od_path(metro), allow_pickle=False)["tract_geoid"]
    bins, car_share = load_car_commute_bins(geoid)
    flows = car_flows(load_flows(metro, geoid), geoid, car_share)
    comp = class_minutes(metro)
    fl = np.where(np.isfinite(comp["tt"]).all(axis=2), flows, 0.0)
    use = (bins.sum(axis=1) > 0) & (fl.sum(axis=1) > 0)
    home = np.cumsum(use) - 1
    i, j = np.nonzero(fl * use[:, None])
    weight = fl[i, j] / fl.sum(axis=1)[i]
    minutes = comp["tt"][i, j]
    b = bins[use]
    n_home = b.shape[0]
    fold = np.random.default_rng(seed).integers(0, n_folds, n_home)
    exposure = (weight[:, None] * minutes * b.sum(axis=1)[home[i], None]).sum(axis=0)
    owner = pooled_classes(exposure)

    rows = []
    for f in range(n_folds):
        train = fold != f
        tr_pair = train[home[i]]
        tr_index = np.cumsum(train) - 1
        res, _, _ = _fit(tr_index[home[i][tr_pair]], weight[tr_pair], minutes[tr_pair], b[train], owner)
        tau, kappa, sigma = res.x[0], res.x[1:1 + N_CLASS][owner], res.x[-1]
        te_pair = ~tr_pair
        te_index = np.cumsum(~train) - 1
        logT = np.log(np.maximum(tau + minutes[te_pair] @ kappa, 1e-3))
        n_te = int((~train).sum())
        cdf = np.column_stack([np.bincount(te_index[home[i][te_pair]],
                                           weight[te_pair] * ndtr((np.log(e) - logT) / sigma), minlength=n_te)
                               for e in BIN_EDGES])
        p = np.clip(np.diff(np.column_stack([np.zeros(n_te), cdf, np.ones(n_te)]), axis=1), 1e-9, None)
        obs, n_car = b[~train], b[~train].sum(axis=1)
        obs_mean = (obs * BIN_MIDPOINTS).sum(axis=1) / n_car
        pred_mean = (p * BIN_MIDPOINTS).sum(axis=1)
        rows.append({
            "fold": f, "tau_min": float(tau), "kappa": kappa.tolist(),
            "pearson_r": float(np.corrcoef(obs_mean, pred_mean)[0, 1]),
            "mean_abs_error_min": float(np.average(np.abs(obs_mean - pred_mean), weights=n_car)),
            "bias_min": float(np.average(pred_mean - obs_mean, weights=n_car)),
            "max_bin_share_gap": float(np.abs(obs.sum(axis=0) / obs.sum()
                                              - (p * n_car[:, None]).sum(axis=0) / n_car.sum()).max()),
        })
    return {"metro": metro, "n_folds": n_folds, "folds": rows}


if __name__ == "__main__":
    workers_ = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    result = calibrate(sys.argv[1], workers_)
    print(json.dumps({k: v for k, v in result.items() if k != "history"}, indent=2))
