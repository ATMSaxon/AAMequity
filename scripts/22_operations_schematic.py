"""Schematic of the vertiport service assumptions.

Two pads at 12 movements per pad per hour give a movement every 2.5 min at
the vertiport, so a passenger arriving at random waits 1.25 min on average.

Output: figures/fig_operations.{pdf,png,svg}
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import paper_figstyle as S
from src.config import CFG

OPS_PER_PAD_H = CFG.aam.ops_per_pad_per_hour
PADS = CFG.aam.pads_per_vertiport
PAD_GAP = 60.0 / OPS_PER_PAD_H                  # minutes between movements on one pad
HEADWAY = PAD_GAP / PADS                         # minutes between movements at the vertiport
MEAN_WAIT = HEADWAY / 2


def bracket(ax, x0, x1, y, text, color=S.INK, above=True):
    h = 0.11 if above else -0.11
    ax.plot([x0, x0, x1, x1], [y, y + h, y + h, y], color=color, lw=0.6)
    ax.text((x0 + x1) / 2, y + h + (0.05 if above else -0.05), text, ha="center",
            va="bottom" if above else "top", fontsize=6, color=color)


def schedule(ax):
    t_end = 15.0
    rows = {"Pad 1": 2.0, "Pad 2": 1.0, "Vertiport": 0.0}
    pad_times = [np.arange(k * HEADWAY, t_end + 1e-9, PAD_GAP) for k in range(PADS)]
    colors = [S.BLUE, S.TEAL]
    for y in rows.values():
        ax.plot([0, t_end], [y, y], color=S.GRID, lw=0.8, zorder=0)
    for k, times in enumerate(pad_times):
        ax.scatter(times, np.full_like(times, 2.0 - k), marker="^", s=22, color=colors[k], zorder=3,
                   edgecolors="white", linewidths=0.4)
        ax.scatter(times, np.zeros_like(times), marker="^", s=22, color=colors[k], zorder=3,
                   edgecolors="white", linewidths=0.4)
    bracket(ax, 0, PAD_GAP, 2.08, f"{PAD_GAP:g} min ({OPS_PER_PAD_H:g} movements per hour)")
    bracket(ax, PAD_GAP / 2, PAD_GAP / 2 + PAD_GAP, 1.08, f"{PAD_GAP:g} min, offset by {HEADWAY:g} min")
    bracket(ax, 2 * PAD_GAP + HEADWAY, 3 * PAD_GAP, 0.08, f"headway {HEADWAY:g} min")
    # a passenger who arrives at random waits until the next movement
    arrive, board = 8.6, 10.0
    ax.plot([arrive, board], [-0.32, -0.32], color=S.RED, lw=1.6, solid_capstyle="butt")
    ax.scatter([arrive], [-0.32], s=10, color=S.RED, zorder=4)
    ax.annotate("passenger arrives", xy=(arrive, -0.32), xytext=(-3, 0), textcoords="offset points",
                ha="right", va="center", fontsize=6, color=S.RED)
    ax.annotate("boards", xy=(board, -0.32), xytext=(3, 0), textcoords="offset points",
                ha="left", va="center", fontsize=6, color=S.RED)
    ax.set_yticks(list(rows.values()))
    ax.set_yticklabels(["Pad 1", "Pad 2", "Vertiport\n(both pads)"], fontsize=6.5)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(-0.3, t_end + 0.3)
    ax.set_ylim(-0.6, 2.5)
    ax.set_xticks(np.arange(0, t_end + 0.1, HEADWAY))
    ax.set_xticklabels([f"{t:g}" for t in np.arange(0, t_end + 0.1, HEADWAY)])
    ax.set_xlabel("Time (min)")
    ax.text(1.02, 0.5, f"{PADS} pads × {OPS_PER_PAD_H:g} = {PADS * OPS_PER_PAD_H:g} movements per hour\n"
            f"Mean wait = headway / 2 = {MEAN_WAIT:g} min",
            transform=ax.transAxes, ha="left", va="center", fontsize=7.2, color=S.INK, linespacing=1.6)


def main() -> None:
    S.setup()
    fig = plt.figure(figsize=(S.FULL_WIDTH, 1.75))
    ax = fig.add_axes([0.10, 0.22, 0.56, 0.74])
    schedule(ax)
    print("->", S.save(fig, "fig_operations"))


if __name__ == "__main__":
    main()
