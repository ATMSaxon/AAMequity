"""Demographic group definitions for equity analysis.

Builds tract-level group labels from real ACS columns:
  income_group     : low / mid / high (terciles within metro on median_hh_income)
  minority_group   : majority_minority (≥50% non-white-NH) vs not
  carless_group    : high (top quartile pct_zero_vehicle) vs not
  transit_dep_group: high (top quartile pct_transit_commute) vs not
  elderly_group    : high (top quartile pct_65_plus) vs not
  disability_group : high (top quartile pct_disab_sample) vs not
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import geopandas as gpd

from .data_layer import load_tracts_joined


def assign_groups(metro: str = "la") -> pd.DataFrame:
    """Add group columns to tracts; returns plain DataFrame keyed by tract_geoid."""
    tracts = load_tracts_joined(metro)
    df = tracts[["tract_geoid", "pop", "median_hh_income", "pct_minority",
                 "pct_zero_vehicle", "pct_transit_commute",
                 "pct_65_plus", "pct_disab_sample", "pct_renter",
                 "pct_lang_isolated"]].copy()

    # Income terciles (population-weighted)
    inc = df["median_hh_income"].dropna()
    if len(inc) >= 3:
        # population-weighted-ish quantiles (use unweighted on tract values; OK at tract resolution)
        q1, q2 = np.nanpercentile(inc, [33.33, 66.67])
    else:
        q1, q2 = np.nanmedian(inc), np.nanmedian(inc)
    df["income_group"] = pd.cut(
        df["median_hh_income"],
        bins=[-np.inf, q1, q2, np.inf],
        labels=["low_income", "mid_income", "high_income"],
    ).astype(object)

    df["minority_group"] = np.where(df["pct_minority"] >= 0.5,
                                     "majority_minority", "majority_white")

    def _top_quartile_label(col, label_high, label_low):
        thresh = np.nanpercentile(df[col].fillna(0), 75)
        return np.where(df[col].fillna(0) >= thresh, label_high, label_low)

    df["carless_group"]    = _top_quartile_label("pct_zero_vehicle",     "high_carless",     "rest")
    df["transit_dep_group"] = _top_quartile_label("pct_transit_commute", "high_transit_dep", "rest")
    df["elderly_group"]    = _top_quartile_label("pct_65_plus",          "high_elderly",     "rest")
    df["disability_group"] = _top_quartile_label("pct_disab_sample",     "high_disab",       "rest")

    # Procedural-equity proxy: composite of language isolation + renter share + low income
    proc = (df["pct_lang_isolated"].fillna(0) +
            df["pct_renter"].fillna(0) +
            (df["income_group"] == "low_income").astype(float))
    proc_q = np.nanpercentile(proc, 75)
    df["procedural_disadvantage"] = np.where(proc >= proc_q,
                                              "high_proc_disadv", "rest")
    return df
