"""Five sensitivity sweeps (per RESEARCH_BRIEF §15).

  1. AAM fare:           {high, mid, low, subsidized}
  2. Wait time:          ops_per_pad_per_hour ∈ {3, 6, 12}
  3. Vertiport capacity: pads_per_vertiport ∈ {1, 2, 4}
  4. Noise weight:       (w_noise, w_landuse) tuples ∈ 3 settings
  5. Equity weight:      equity_floor_share ∈ {0.4, 0.6, 0.8}

Each sweep re-runs ALL 7 scenarios under the perturbed parameter and stores results.
Outputs are saved per sweep so paper figures can compare frontiers.
"""

from __future__ import annotations
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CFG, AAMParams, BurdenParams, OptimizerParams, RESULTS
from .scenarios import run_all_scenarios


def _save_sweep(name: str, results: dict[str, pd.DataFrame], metro: str):
    out = RESULTS / metro / "sensitivity" / name
    out.mkdir(parents=True, exist_ok=True)
    for k, df in results.items():
        df.to_csv(out / f"sweep_{k}.csv", index=False)


def fare_sweep(metro: str = "la", n_vertiports: int = 25, max_candidates: int = 50):
    """Run all scenarios at each fare tier."""
    out = {}
    for tier in CFG.aam.fare_per_km:
        df = run_all_scenarios(metro=metro, n_vertiports=n_vertiports,
                               max_candidates=max_candidates, fare_tier=tier)
        df["sweep_setting"] = tier
        out[tier] = df
    _save_sweep("fare", out, metro)
    return out


def wait_sweep(metro: str = "la", n_vertiports: int = 25, max_candidates: int = 50):
    settings = {"freq_4ph": 4.0, "freq_8ph": 8.0, "freq_16ph": 16.0}
    base = CFG.aam.ops_per_pad_per_hour
    out = {}
    for tag, val in settings.items():
        CFG.aam.ops_per_pad_per_hour = val
        df = run_all_scenarios(metro=metro, n_vertiports=n_vertiports, max_candidates=max_candidates)
        df["sweep_setting"] = tag
        out[tag] = df
    CFG.aam.ops_per_pad_per_hour = base
    _save_sweep("wait", out, metro)
    return out


def capacity_sweep(metro: str = "la", n_vertiports: int = 25, max_candidates: int = 50):
    settings = {"single_pad": 1, "dual_pad": 2, "hub_pad": 4}
    base = CFG.aam.pads_per_vertiport
    out = {}
    for tag, val in settings.items():
        CFG.aam.pads_per_vertiport = val
        df = run_all_scenarios(metro=metro, n_vertiports=n_vertiports, max_candidates=max_candidates)
        df["sweep_setting"] = tag
        out[tag] = df
    CFG.aam.pads_per_vertiport = base
    _save_sweep("capacity", out, metro)
    return out


def noise_weight_sweep(metro: str = "la", n_vertiports: int = 25, max_candidates: int = 50):
    settings = {
        "noise_low":  (0.20, 0.30),
        "noise_mid":  (0.40, 0.20),
        "noise_high": (0.60, 0.10),
    }
    base = (CFG.burden.w_noise, CFG.burden.w_landuse)
    out = {}
    for tag, (wn, wl) in settings.items():
        CFG.burden.w_noise = wn
        CFG.burden.w_landuse = wl
        # Re-balance other components to keep weights summing to 1 (excluding the two we set)
        rest = 1.0 - (wn + wl)
        # Distribute proportionally to original visual / traffic / safety
        orig_rest = 0.10 + 0.15 + 0.15  # default w_visual + w_traffic + w_safety
        CFG.burden.w_visual  = rest * (0.10 / orig_rest)
        CFG.burden.w_traffic = rest * (0.15 / orig_rest)
        CFG.burden.w_safety  = rest * (0.15 / orig_rest)
        df = run_all_scenarios(metro=metro, n_vertiports=n_vertiports, max_candidates=max_candidates)
        out[tag] = df
    CFG.burden.w_noise, CFG.burden.w_landuse = base
    CFG.burden.w_visual, CFG.burden.w_traffic, CFG.burden.w_safety = 0.10, 0.15, 0.15
    _save_sweep("noise_weight", out, metro)
    return out


def equity_weight_sweep(metro: str = "la", n_vertiports: int = 25, max_candidates: int = 50):
    settings = {"floor_05": 0.05, "floor_10": 0.10, "floor_30": 0.30}
    base = CFG.opt.equity_floor_share
    out = {}
    for tag, val in settings.items():
        CFG.opt.equity_floor_share = val
        df = run_all_scenarios(metro=metro, n_vertiports=n_vertiports, max_candidates=max_candidates)
        out[tag] = df
    CFG.opt.equity_floor_share = base
    _save_sweep("equity_weight", out, metro)
    return out


def run_all_sweeps(metro: str = "la", n_vertiports: int = 25, max_candidates: int = 60):
    """Run all five sensitivity sweeps. NOTE: 5 sweeps × 4 settings × 7 scenarios = 140 MILPs.
    Default uses smaller candidate budget to keep runtime manageable."""
    print("Sensitivity sweeps starting (5 dimensions × ~3-4 settings × 7 scenarios)")
    fare_sweep(metro, n_vertiports, max_candidates)
    wait_sweep(metro, n_vertiports, max_candidates)
    capacity_sweep(metro, n_vertiports, max_candidates)
    noise_weight_sweep(metro, n_vertiports, max_candidates)
    equity_weight_sweep(metro, n_vertiports, max_candidates)
    print("All sensitivity sweeps complete.")
