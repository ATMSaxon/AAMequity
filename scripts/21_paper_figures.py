"""Manuscript figures in the shared paper style.

map2_demographics       LA County median income, minority share and zero-vehicle share
fig_fare_direction      accessibility gain against fare, and the income concentration of the gain
fig_policy              fare subsidy and burden cap
map_policy_change       LA: gain and burden under S2, and the changes made by S6 and S5
fig_burden              distribution of burden, HOLC grades, existing facilities and greenfield sites
fig_operations_sim      simulated vertiport operations: waits, accessibility and noise

Usage:  python scripts/21_paper_figures.py [name ...]
Output: figures/<name>.pdf (+ .svg, .png)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm, Normalize
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import paper_figstyle as S
from src.config import CFG

REV = ROOT / "results" / "revision"
METROS = ["la", "dfw", "sfbay", "nyc"]
TIER = "public_service"
TIER_LINES = [("public_service", "Public\nservice"), ("long_run", "Long\nrun"), ("near_term", "Near\nterm")]
GAIN_THRESHOLD_PCT = 0.1


def load(kind: str, metro: str) -> dict:
    return json.loads((REV / f"{kind}_{metro}.json").read_text())


def _row_of_maps(n: int, base: S.MetroBase, gap: float = 0.12, head: float = 0.22,
                 foot: float = 0.0, width: float = S.FULL_WIDTH):
    w = (width - (n - 1) * gap) / n
    h = w / base.aspect
    height = head + h + foot
    fig = plt.figure(figsize=(width, height))
    axes = [fig.add_axes([(i * (w + gap)) / width, foot / height, w / width, h / height])
            for i in range(n)]
    return fig, axes, height


# ---------------------------------------------------------------- demographics
def demographics():
    base = S.MetroBase("la")
    t = base.tracts
    layers = [
        ("median_hh_income", "Median household income", LogNorm(25_000, 250_000),
         [25_000, 50_000, 100_000, 250_000], ["$25k", "$50k", "$100k", "$250k"], "US dollars"),
        ("pct_minority", "Minority population share", Normalize(0, 1),
         [0, 0.25, 0.5, 0.75, 1], ["0", "0.25", "0.50", "0.75", "1"], "Share of residents"),
        ("pct_zero_vehicle", "Zero-vehicle household share", Normalize(0, 0.3),
         [0, 0.1, 0.2, 0.3], ["0", "0.10", "0.20", "0.30"], "Share of households"),
    ]
    fig, axes, height = _row_of_maps(3, base, foot=0.42)
    for ax, letter, (col, title, norm, ticks, labels, unit) in zip(axes, "abc", layers):
        base.draw(ax, column=col, cmap=S.BLUES, norm=norm, scalebar=(letter == "a"))
        S.panel_label(ax, letter, title)
        pos = ax.get_position()
        cax = S.horizontal_colorbar(fig, [pos.x0 + 0.15 * pos.width, 0.14 / height, 0.7 * pos.width,
                                          0.07 / height], S.BLUES, norm, unit, ticks, labels,
                                    extend="both" if col != "pct_minority" else "neither")
        cax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
    return S.save(fig, "map2_demographics")


# ---------------------------------------------------------------- helpers
def _metro_rows(ax, labels=True):
    """Metros as horizontal bands, LA at the top."""
    for y in range(len(METROS)):
        if y % 2 == 0:
            ax.axhspan(y - 0.5, y + 0.5, color="#F6F6F6", zorder=0, lw=0)
    ax.set_ylim(-0.5, len(METROS) - 0.5)
    ax.set_yticks(range(len(METROS)))
    ax.set_yticklabels([S.METRO_SHORT[m] for m in reversed(METROS)] if labels else [])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)


def _row(m: str) -> int:
    return len(METROS) - 1 - METROS.index(m)


def _sign_notes(ax, low: str, high: str, pad: float = 0.01):
    ax.annotate(low, xy=(-pad, 1.0), xycoords=("data", "axes fraction"), xytext=(0, 3),
                textcoords="offset points", ha="right", va="bottom", fontsize=5.5, color=S.MUTED)
    ax.annotate(high, xy=(pad, 1.0), xycoords=("data", "axes fraction"), xytext=(0, 3),
                textcoords="offset points", ha="left", va="bottom", fontsize=5.5, color=S.MUTED)


def _ci_err(ax, x, y, lo, hi, **kw):
    ax.errorbar(x, y, xerr=[[x - lo], [hi - x]], capsize=1.4, elinewidth=0.8, capthick=0.8, **kw)


# ---------------------------------------------------------------- 5.1 fare and direction
def fare_direction():
    width, height = S.FULL_WIDTH, 4.3
    fig = plt.figure(figsize=(width, height))
    top_axes = []
    left, gap, w = 0.075, 0.035, (1 - 0.075 - 0.01 - 3 * 0.035) / 4
    for i, m in enumerate(METROS):
        ax = fig.add_axes([left + i * (w + gap), 0.615, w, 0.335])
        top_axes.append(ax)
        rows = sorted(load("fare", m)["fare_sweep"]["rows"], key=lambda r: r["fare_per_km"])
        rows = [r for r in rows if r["fare_per_km"] <= 0.8 + 1e-9]
        x = np.array([r["fare_per_km"] for r in rows])
        for tier, _ in TIER_LINES:
            ax.axvline(CFG.aam.fare_per_km[tier], color=S.GRID, lw=0.8, zorder=0)
        ax.axhline(GAIN_THRESHOLD_PCT, color=S.MUTED, lw=0.6, ls=(0, (3, 2)), zorder=1)
        series = [("high_income", f"dA_pct_high_income", 1.1), ("mid_income", "dA_pct_mid_income", 1.1),
                  ("low_income", "dA_pct_low_income", 1.1), (None, "mean_dA_pct_of_base", 1.6)]
        for key, col, lw in series:
            y = np.array([r[col] for r in rows], dtype=float)
            y[y < 0.01] = np.nan
            ax.plot(x, y, color=S.INK if key is None else S.INCOME_COLORS[key], lw=lw, zorder=3)
        ax.set_yscale("log")
        ax.set_ylim(0.01, 15)
        ax.set_yticks([0.01, 0.1, 1, 10])
        ax.set_yticklabels(["0.01", "0.1", "1", "10"] if i == 0 else [])
        ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())
        ax.set_xlim(0, 0.7)
        ax.set_xticks([0, 0.2, 0.4, 0.6])
        ax.set_xticklabels(["0", "0.2", "0.4", "0.6"])
        ax.set_xlabel("Fare ($ per km)")
        S.panel_label(ax, "abcd"[i], S.METRO_SHORT[m])
    ax = top_axes[0]
    ax.set_ylabel("Accessibility gain\n(% of baseline, log scale)")
    for tier, label in TIER_LINES:
        rate = CFG.aam.fare_per_km[tier]
        right = tier == "near_term"
        ax.text(rate + (-0.012 if right else 0.012), 11, label, fontsize=5.5, va="top",
                ha="right" if right else "left", color=S.MUTED, linespacing=1.2)
    top_axes[1].text(0.57, GAIN_THRESHOLD_PCT * 1.15, "0.1% threshold", fontsize=5.5, ha="right", va="bottom",
                     color=S.MUTED)
    handles = [Line2D([], [], color=S.INK, lw=1.6, label="All residents"),
               Line2D([], [], color=S.RED, lw=1.1, label="Low income"),
               Line2D([], [], color=S.GREY, lw=1.1, label="Middle income"),
               Line2D([], [], color=S.BLUE, lw=1.1, label="High income")]
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(0.53, 0.47), ncol=4, columnspacing=2.2)

    ax = fig.add_axes([0.075, 0.085, 0.56, 0.31])
    series = [
        ("Optimised network, uniform value of time", "o", S.BLUE, True, 0.27),
        ("Random networks, existing facilities", "s", S.GREY, False, 0.09),
        ("Random networks, greenfield sites", "^", S.TEAL, False, -0.09),
        ("Optimised network, income-specific value of time", "D", S.RED, True, -0.27),
    ]
    _metro_rows(ax)
    for m in METROS:
        y = _row(m)
        s = load("robustness", m)["sign_uniform_value_of_time"][TIER]
        inc = load("experiments", m)["tiers"][TIER]["uncertainty"]["ci_dA_vs_income"]
        opt, rnd, grn = s["optimised"]["interval"], s["random_networks_ci"], s["greenfield_random_networks_ci"]
        values = [(opt["estimate"], opt["low"], opt["high"]), (rnd["mean"], rnd["p2.5"], rnd["p97.5"]),
                  (grn["mean"], grn["p2.5"], grn["p97.5"]), (inc["estimate"], inc["low"], inc["high"])]
        for (_, marker, color, filled, dy), (mid, lo, hi) in zip(series, values):
            _ci_err(ax, mid, y + dy, lo, hi, fmt=marker, ms=3.8, color=color,
                    mfc=color if filled else "white", mew=0.8, zorder=3)
    ax.axvline(0, color=S.INK, lw=0.6, zorder=1)
    ax.set_xlabel("Income concentration index of accessibility gain")
    ax.set_ylim(-0.5, len(METROS) - 0.5 + 0.45)
    ax.text(-0.012, len(METROS) - 0.5 + 0.22, "Gains concentrated in lower-income tracts", ha="right", va="center",
            fontsize=5.5, color=S.MUTED)
    ax.text(0.012, len(METROS) - 0.5 + 0.22, "Gains concentrated in higher-income tracts", ha="left", va="center",
            fontsize=5.5, color=S.MUTED)
    S.panel_label(ax, "e", "Income concentration of the gain, public-service fare")
    handles = [Line2D([], [], marker=mk, ls="", ms=3.8, color=c, mfc=c if f else "white", mew=0.8, label=l)
               for l, mk, c, f, _ in series]
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.665, 0.23), ncol=1, labelspacing=1.0)
    return S.save(fig, "fig_fare_direction")


# ---------------------------------------------------------------- 5.2 policy instruments
def policy():
    width, height = S.FULL_WIDTH, 2.35
    fig, axes = plt.subplots(1, 3, figsize=(width, height))
    fig.subplots_adjust(left=0.075, right=0.92, bottom=0.30, top=0.88, wspace=0.42)
    shares = [0.0, 0.25, 0.5, 0.75]
    for m in METROS:
        sub = load("subsidy", m)["tiers"][TIER]
        c = S.METRO_COLORS[m]
        for kind, ls, mfc in (("means_tested", "-", c), ("universal", (0, (3, 1.5)), "white")):
            ci = [sub["0.5"]["none"]["ci_dA_vs_income"]] + [sub[f"{s}"][kind]["ci_dA_vs_income"] for s in shares[1:]]
            axes[0].plot(shares, ci, color=c, ls=ls, lw=1.0, marker="o", ms=2.8, mfc=mfc, mec=c, mew=0.7)
        low = [sub["0.5"]["none"]["low_income_dA_pct_of_base"]] + [sub[f"{s}"]["means_tested"]["low_income_dA_pct_of_base"]
                                                                   for s in shares[1:]]
        axes[1].plot(shares, low, color=c, lw=1.0, marker="o", ms=2.8, mec="white", mew=0.4)
        axes[1].annotate(S.METRO_SHORT[m], xy=(shares[-1], low[-1]), xytext=(4, 0), textcoords="offset points",
                         ha="left", va="center", fontsize=6, color=c)
    ax = axes[0]
    ax.axhline(0, color=S.INK, lw=0.6)
    ax.set_xticks(shares)
    ax.set_xticklabels(["0", "25", "50", "75"])
    ax.set_xlabel("Fare subsidy (%)")
    ax.set_ylabel("Income concentration index of gain")
    ax.legend(handles=[Line2D([], [], color=S.INK, lw=1.0, marker="o", ms=2.8, label="Means-tested"),
                       Line2D([], [], color=S.INK, lw=1.0, ls=(0, (3, 1.5)), marker="o", ms=2.8, mfc="white",
                              label="Universal")], loc="lower left")
    S.panel_label(ax, "a", "Concentration of the gain")
    ax = axes[1]
    ax.set_xticks(shares)
    ax.set_xticklabels(["0", "25", "50", "75"])
    ax.set_xlabel("Fare subsidy (%)")
    ax.set_ylabel("Low-income gain (% of baseline)")
    ax.set_ylim(bottom=0)
    S.panel_label(ax, "b", "Gain of the low-income tercile")
    ax = axes[2]
    for m in METROS:
        p = [r for r in load("fare", m)["pareto"][TIER] if r["constraint"] == "cap"]
        ref = next(r for r in p if r["level"] == 1.0)["mean_dA"]
        ok = sorted((r for r in p if r["status"] == "Optimal"), key=lambda r: r["level"])
        x = [r["level"] for r in ok]
        y = [100 * r["mean_dA"] / ref for r in ok]
        ax.plot(x, y, marker="o", ms=2.8, lw=1.0, color=S.METRO_COLORS[m], mec="white", mew=0.4)
        below = m == "la"
        ax.annotate(S.METRO_SHORT[m], xy=(x[0], y[0]), xytext=(0, -5) if below else (4, 0),
                    textcoords="offset points", ha="center" if below else "left",
                    va="top" if below else "center", fontsize=6, color=S.METRO_COLORS[m])
    ax.set_xlabel("Burden cap (share of unconstrained peak)")
    ax.set_ylabel("Gain retained (%)")
    ax.set_xlim(1.02, 0.28)
    ax.set_xticks([1.0, 0.8, 0.6, 0.4])
    ax.set_ylim(0, 108)
    ax.grid(axis="y", color=S.GRID, lw=0.5)
    ax.set_axisbelow(True)
    S.panel_label(ax, "c", "Burden cap")
    return S.save(fig, "fig_policy")


def map_policy_change():
    base = S.MetroBase("la")
    per = {}
    for scen in ("S2", "S5", "S6"):
        df = pd.read_parquet(ROOT / "results" / "la" / TIER / f"scenario_{scen}_per_tract.parquet")
        per[scen] = base.tracts[["tract_geoid"]].merge(df, on="tract_geoid", how="left").fillna({"dA": 0.0, "B": 0.0})
    gain = per["S2"]["dA"]
    burden = per["S2"]["B"]
    d_gain = per["S6"]["dA"] - per["S2"]["dA"]
    d_burden = per["S5"]["B"] - per["S2"]["B"]
    lim_g = np.nanpercentile(np.abs(d_gain), 97.5)
    lim_b = np.nanpercentile(np.abs(d_burden), 97.5)
    panels = [
        ("a", "S2: accessibility gain", gain, S.BLUES, Normalize(0, np.nanpercentile(gain, 95)), "Gravity-weighted jobs", "max"),
        ("b", "S6 minus S2: gain", d_gain, S.DIV_GAIN, Normalize(-lim_g, lim_g), "Change in gravity-weighted jobs", "both"),
        ("c", "S2: Burden Index", burden, S.REDS, Normalize(0, np.nanpercentile(burden, 95)), "Burden Index", "max"),
        ("d", "S5 minus S2: burden", d_burden, S.DIV_BURDEN, Normalize(-lim_b, lim_b), "Change in Burden Index", "both"),
    ]
    fig, axes, height = _row_of_maps(4, base, gap=0.08, foot=0.5)
    for ax, (letter, title, data, cmap, norm, label, ext) in zip(axes, panels):
        base.draw(ax, column="v", data=data, cmap=cmap, norm=norm, scalebar=(letter == "a"))
        S.panel_label(ax, letter, title)
        pos = ax.get_position()
        cax = S.horizontal_colorbar(fig, [pos.x0 + 0.1 * pos.width, 0.15 / height, 0.8 * pos.width, 0.07 / height],
                                    cmap, norm, label, extend=ext)
        cax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
        if "jobs" in label.lower():
            signed = label.startswith("Change")
            cax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(
                lambda v, _, signed=signed: "0" if abs(v) < 1e-9 else (f"{v / 1000:+.0f}k" if signed else f"{v / 1000:.0f}k")))
    return S.save(fig, "map_policy_change")


# ---------------------------------------------------------------- 5.3 burden
def burden():
    width, height = S.FULL_WIDTH, 4.7
    fig = plt.figure(figsize=(width, height))
    W = [0.075, 0.405, 0.735]
    wide = 0.24
    ax_a = fig.add_axes([W[0], 0.645, wide, 0.30])
    ax_b = fig.add_axes([W[1], 0.645, wide, 0.30])
    ax_c = fig.add_axes([W[2], 0.645, wide, 0.30])
    ax_d = fig.add_axes([W[0], 0.115, wide, 0.27])
    ax_e = fig.add_axes([W[1], 0.115, wide, 0.27])
    ax_f = fig.add_axes([W[2], 0.115, wide, 0.27])
    exp = {m: load("experiments", m)["tiers"][TIER] for m in METROS}
    green = {m: load("greenfield", m)["tiers"][TIER] for m in METROS}

    # a: concentration of burden against three attributes
    _metro_rows(ax_a)
    attrs = [("Income", lambda t: t["uncertainty"]["ci_B_vs_income"], "o", S.INK, 0.22),
             ("Minority share", lambda t: t["groups"]["ci_B_vs_minority_share"], "s", S.VIOLET, 0.0),
             ("Zero-vehicle share", lambda t: t["groups"]["ci_B_vs_zero_vehicle_share"], "^", S.TEAL, -0.22)]
    for m in METROS:
        for _, get, mk, c, dy in attrs:
            d = get(exp[m])
            _ci_err(ax_a, d["estimate"], _row(m) + dy, d["low"], d["high"], fmt=mk, ms=3.4, color=c, zorder=3)
    ax_a.axvline(0, color=S.INK, lw=0.6)
    ax_a.set_xlabel("Concentration index of burden")
    ax_a.legend(handles=[Line2D([], [], marker=mk, ls="", ms=3.4, color=c, label=l) for l, _, mk, c, _ in attrs],
                loc="upper left", bbox_to_anchor=(-0.32, -0.27), ncol=3, columnspacing=0.8, handletextpad=0.2)
    S.panel_label(ax_a, "a", "Who bears the burden")

    # b: mean burden by HOLC grade
    for m in METROS:
        g = exp[m]["redlining"]["by_grade"]
        ks = [k for k in "ABCD" if k in g]
        ax_b.plot(range(len(ks)), [g[k]["mean_B"] for k in ks], color=S.METRO_COLORS[m], lw=1.0, marker="o",
                  ms=2.8, mec="white", mew=0.4)
        ax_b.annotate(S.METRO_SHORT[m], xy=(len(ks) - 1, g[ks[-1]]["mean_B"]), xytext=(4, 0),
                      textcoords="offset points", ha="left", va="center", fontsize=6, color=S.METRO_COLORS[m])
    ax_b.set_xticks(range(4))
    ax_b.set_xticklabels(["A", "B", "C", "D"])
    ax_b.set_xlim(-0.2, 3.2)
    ax_b.set_ylim(bottom=0)
    ax_b.set_xlabel("1939 HOLC grade")
    ax_b.set_ylabel("Mean Burden Index")
    S.panel_label(ax_b, "b", "Burden by HOLC grade")

    # c: rank correlation against the random-network range
    _metro_rows(ax_c)
    for m in METROS:
        rn = exp[m]["redlining"]["random_networks"]
        rho = exp[m]["uncertainty"]["rho_B_vs_holc"]
        y = _row(m)
        ax_c.add_patch(mpl.patches.Rectangle((rn["rho_B_p2.5"], y - 0.16), rn["rho_B_p97.5"] - rn["rho_B_p2.5"], 0.32,
                                             facecolor="#D9D9D9", edgecolor="none", zorder=1))
        _ci_err(ax_c, rho["estimate"], y, rho["low"], rho["high"], fmt="o", ms=3.4, color=S.METRO_COLORS[m], zorder=3)
    ax_c.axvline(0, color=S.INK, lw=0.6)
    ax_c.set_xlabel("Spearman correlation, burden and HOLC rank")
    ax_c.legend(handles=[Line2D([], [], marker="o", ls="", ms=3.4, color=S.INK, label="Optimised (95% interval)"),
                         mpl.patches.Patch(facecolor="#D9D9D9", label="Random networks (95% range)")],
                loc="upper left", bbox_to_anchor=(-0.32, -0.27), ncol=2, columnspacing=0.8, handletextpad=0.3)
    S.panel_label(ax_c, "c", "Association with HOLC grade")

    # d-f: existing facilities against greenfield sites
    pools = [("existing_facilities", "Existing facilities", S.GREY), ("greenfield_49", "Greenfield sites", S.BLUE)]
    _metro_rows(ax_d)
    for m in METROS:
        y = _row(m)
        a, b = (green[m][k]["ci_dnl_energy_vs_income"] for k, _, _ in pools)
        ax_d.annotate("", xy=(b, y), xytext=(a, y), arrowprops=dict(arrowstyle="-|>", color=S.MUTED, lw=0.7,
                                                                     shrinkA=2.5, shrinkB=2.5, mutation_scale=6))
        for (k, _, c), v in zip(pools, (a, b)):
            ax_d.plot(v, y, "o", ms=3.6, color=c, zorder=3)
    ax_d.axvline(0, color=S.INK, lw=0.6)
    ax_d.set_xlabel("Concentration index of noise")
    ax_d.set_ylim(-0.5, len(METROS) - 0.5 + 0.5)
    ax_d.text(-0.02, len(METROS) - 0.5 + 0.25, "Lower-income", ha="right", va="center", fontsize=5.5, color=S.MUTED)
    ax_d.text(0.02, len(METROS) - 0.5 + 0.25, "Higher-income", ha="left", va="center", fontsize=5.5, color=S.MUTED)
    S.panel_label(ax_d, "d", "Concentration of noise")
    for ax, key, scale, xlabel, letter, title in (
            (ax_e, "population_dnl_ge_40", 1e-3, "Residents (thousand)", "e", "Residents at or above 40 dB"),
            (ax_f, "mean_dA_pct_of_base", 1.0, "% of baseline accessibility", "f", "Accessibility gain")):
        _metro_rows(ax, labels=(letter == "e"))
        for m in METROS:
            y = _row(m)
            for (k, _, c), dy in zip(pools, (0.17, -0.17)):
                ax.barh(y + dy, green[m][k][key] * scale, height=0.3, color=c, zorder=2)
        ax.set_xlabel(xlabel)
        ax.set_xlim(left=0)
        S.panel_label(ax, letter, title)
    fig.legend(handles=[mpl.patches.Patch(facecolor=c, label=l) for _, l, c in pools], loc="lower center",
               bbox_to_anchor=(0.53, 0.0), ncol=2, columnspacing=2.0)
    fig.text(0.5, 0.455, "Accessibility-maximising networks from existing facilities and from greenfield sites",
             fontsize=7, color=S.INK, ha="center", va="bottom")
    return S.save(fig, "fig_burden")


# ---------------------------------------------------------------- 5.4 operations
def operations_sim():
    ops = {m: load("operations", m) for m in METROS}
    sim = ops["la"]["simulation"]["results"]
    dem = np.array([r["demand_pax_per_hour"] for r in sim])
    width, height = S.FULL_WIDTH, 2.3
    fig, axes = plt.subplots(1, 3, figsize=(width, height))
    fig.subplots_adjust(left=0.07, right=0.97, bottom=0.2, top=0.86, wspace=0.42)
    ax = axes[0]
    ax.plot(dem, [r["mean_wait_min"] for r in sim], color=S.INK, lw=1.1, marker="o", ms=2.8, label="Mean")
    ax.plot(dem, [r["p90_wait_min"] for r in sim], color=S.INK, lw=0.9, ls=(0, (3, 1.5)), marker="o", ms=2.8,
            mfc="white", label="90th percentile")
    ax.axhline(CFG.aam.ops_per_pad_per_hour and 60 / (2 * CFG.aam.ops_per_pad_per_hour * CFG.aam.pads_per_vertiport),
               color=S.RED, lw=0.8)
    ax.text(dem[0], 1.25 * 1.15, "Fixed-headway wait, 1.25 min", fontsize=5.5, color=S.RED, va="bottom")
    ax.set_yscale("log")
    ax.set_yticks([1, 2, 5, 10, 20, 50, 100, 200])
    ax.set_yticklabels(["1", "2", "5", "10", "20", "50", "100", "200"])
    ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax.set_xlabel("Demand (passengers per hour)")
    ax.set_ylabel("Passenger wait (min, log scale)")
    ax.legend(loc="upper left")
    S.panel_label(ax, "a", "Passenger wait")
    ax = axes[1]
    for m in METROS:
        rows = [r for r in ops[m]["accessibility_by_wait"] if abs(r["fare_per_km"] - CFG.aam.fare_per_km[TIER]) < 1e-9]
        rows.sort(key=lambda r: r["wait_min"])
        x = [r["wait_min"] for r in rows]
        y = [r["mean_dA_pct_of_base"] for r in rows]
        ax.plot(x, y, color=S.METRO_COLORS[m], lw=1.0, marker="o", ms=2.8, mec="white", mew=0.4,
                label=S.METRO_SHORT[m])
    ax.legend(loc="upper right", handlelength=1.2)
    ax.set_xscale("log")
    ax.set_xticks([1.25, 3, 6, 18])
    ax.set_xticklabels(["1.25", "3", "6", "18"])
    ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax.set_xlim(1.0, 22)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Passenger wait (min, log scale)")
    ax.set_ylabel("Accessibility gain (% of baseline)")
    S.panel_label(ax, "b", "Gain at the public-service fare")
    ax = axes[2]
    mv = np.array([r["movements_per_day"] for r in sim])
    ax.plot(dem, [r["dnl_shift_db_vs_assumed_ops"] for r in sim], color=S.INK, lw=1.1, marker="o", ms=2.8)
    for d_, m_, v_ in zip(dem, mv, [r["dnl_shift_db_vs_assumed_ops"] for r in sim]):
        if d_ in (10, 30, 60):
            ax.annotate(f"{m_:.0f} per day", xy=(d_, v_), xytext=(3, -6), textcoords="offset points",
                        ha="left", va="top", fontsize=5.5, color=S.MUTED)
    ax.set_ylim(0, 9.5)
    ax.set_xlabel("Demand (passengers per hour)")
    ax.set_ylabel("Change in DNL (dB)")
    S.panel_label(ax, "c", "Noise from simulated movements")
    return S.save(fig, "fig_operations_sim")


FIGS = {"map2_demographics": demographics, "fig_fare_direction": fare_direction, "fig_policy": policy,
        "map_policy_change": map_policy_change, "fig_burden": burden, "fig_operations_sim": operations_sim}

if __name__ == "__main__":
    S.setup()
    for name in (sys.argv[1:] or FIGS):
        print("->", FIGS[name]())
