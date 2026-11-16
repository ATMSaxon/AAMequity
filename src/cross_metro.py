"""Cross-metro comparison and figures.

Aggregates per-metro scenario_summary.csv + acoustic_redlining.json into
cross-metro tables and plots so reviewers can see whether the LA findings
generalize to DFW / SF Bay / NY.
"""

from __future__ import annotations
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import RESULTS, FIGURES


def collect_scenario_summaries(metros: list[str]) -> pd.DataFrame:
    parts = []
    for m in metros:
        path = RESULTS / m / "scenario_summary.csv"
        if not path.exists():
            print(f"  WARN: {path} missing")
            continue
        df = pd.read_csv(path)
        df.insert(0, "metro", m)
        parts.append(df)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    out_path = RESULTS / "cross_metro_summary.csv"
    out.to_csv(out_path, index=False)
    print(f"  -> {out_path}")
    return out


def collect_redlining(metros: list[str]) -> dict:
    out = {}
    for m in metros:
        path = RESULTS / m / "acoustic_redlining.json"
        if not path.exists():
            continue
        out[m] = json.loads(path.read_text())
    out_path = RESULTS / "cross_metro_redlining.json"
    out_path.write_text(json.dumps(out, indent=2, default=float))
    print(f"  -> {out_path}")
    return out


def fig_cross_metro_ci(metros: list[str], df: pd.DataFrame):
    """Bar chart of CI(ΔA, income) and EBR by group across metros, per scenario."""
    if df.empty: return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    pivot = df.pivot(index="scenario", columns="metro", values="ci_dA_vs_income")
    pivot.plot(kind="bar", ax=axes[0])
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_title("CI(ΔA, income) — pro-rich tilt by metro × scenario")
    axes[0].set_ylabel("Concentration index")
    pivot_b = df.pivot(index="scenario", columns="metro", values="ci_B_vs_income")
    pivot_b.plot(kind="bar", ax=axes[1])
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title("CI(B, income) — burden tilt by metro × scenario")
    axes[1].set_ylabel("Concentration index")
    plt.tight_layout()
    out = FIGURES / "fig_cross_metro_ci.pdf"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> {out}")


def fig_cross_metro_redlining(redlining: dict):
    """Spearman ρ(B vs HOLC rank) per metro × scenario."""
    rows = []
    for m, payload in redlining.items():
        for s, info in payload.get("scenarios", {}).items():
            rows.append({"metro": m, "scenario": s,
                          "rho_B": info.get("spearman_B_vs_rank"),
                          "rho_dA": info.get("spearman_dA_vs_rank")})
    if not rows: return
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    pivot_b = df.pivot(index="scenario", columns="metro", values="rho_B")
    pivot_b.plot(kind="bar", ax=axes[0])
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_title("Spearman ρ(B, HOLC rank=D)\n>0: AAM compounds redlining noise")
    pivot_a = df.pivot(index="scenario", columns="metro", values="rho_dA")
    pivot_a.plot(kind="bar", ax=axes[1])
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title("Spearman ρ(ΔA, HOLC rank=D)\n>0: D-graded tracts get more spatial proximity")
    plt.tight_layout()
    out = FIGURES / "fig_cross_metro_redlining.pdf"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> {out}")


def main(metros: list[str] = None):
    metros = metros or ["la", "dfw", "sfbay", "nyc"]
    print(f"Cross-metro analysis: {metros}")
    df = collect_scenario_summaries(metros)
    if df.empty:
        print("  no metro summaries found"); return
    print(df.to_string())
    fig_cross_metro_ci(metros, df)
    redlining = collect_redlining(metros)
    if redlining:
        fig_cross_metro_redlining(redlining)


if __name__ == "__main__":
    import sys
    metros = sys.argv[1:] or None
    main(metros)
