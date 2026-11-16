"""House-style helpers for publication figures (scientific-figure-making house style).

Apply via:
    from src.figure_style import apply_publication_style, METRO_COLORS, finalize
    apply_publication_style()
    ...
    finalize(fig, FIGURES / "fig_xxx.pdf")
"""
from __future__ import annotations
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt


PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "green_1": "#DDF3DE", "green_2": "#AADCA9", "green_3": "#8BCF8B",
    "red_1": "#F6CFCB", "red_2": "#E9A6A1", "red_strong": "#B64342",
    "neutral": "#CFCECE", "neutral_dark": "#4D4D4D",
    "highlight": "#FFD700",
    "teal": "#42949E", "violet": "#9A4D8E",
}


METRO_COLORS = {
    "la":    PALETTE["blue_main"],
    "dfw":   PALETTE["teal"],
    "sfbay": PALETTE["green_3"],
    "nyc":   PALETTE["red_strong"],
    "LA":    PALETTE["blue_main"],
    "DFW":   PALETTE["teal"],
    "SFBAY": PALETTE["green_3"],
    "NYC":   PALETTE["red_strong"],
}

METRO_LABELS = {"la": "LA", "dfw": "DFW", "sfbay": "SF Bay", "nyc": "NYC"}

INCOME_GROUP_COLORS = {
    "low_income":  PALETTE["red_strong"],
    "mid_income":  PALETTE["neutral_dark"],
    "high_income": PALETTE["blue_main"],
}

SCENARIO_ORDER = ["S1", "S2", "S3", "S4", "S6", "S7"]  # S5 dropped — infeasible in all 4 metros, no visual content


def apply_illustrator_rc() -> None:
    """rcParams required for clean editing in Adobe Illustrator.

    The two things that break Illustrator import are (1) Type-3 fonts (the
    matplotlib default), which Illustrator cannot edit, and (2) mathtext
    (``$...$``) being composed from a separate math-font subset with a
    non-standard glyph encoding, which Illustrator renders as boxes/garbage.

    Fixes:
      * ``pdf.fonttype = 42`` / ``ps.fonttype = 42`` -> embed editable TrueType.
      * ``svg.fonttype = 'none'`` -> SVG keeps text as real ``<text>`` nodes
        that reference the font by name (Illustrator substitutes its own copy).
      * ``mathtext.default = 'regular'`` -> render math in the *same* sans font
        as the rest of the figure, so ``$\\Delta A$``, ``$\\rho$`` etc. become
        ordinary editable glyphs instead of an unmappable math subset.
    """
    mpl.rcParams.update({
        "font.family": ["Helvetica", "Arial", "DejaVu Sans", "sans-serif"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "mathtext.fontset": "dejavusans",
        "mathtext.default": "regular",
    })


def apply_publication_style(font_size: int = 15, axes_linewidth: float = 1.6) -> None:
    """Configure matplotlib rcParams for publication output (Illustrator-ready)."""
    apply_illustrator_rc()
    mpl.rcParams.update({
        "font.size": font_size,
        "axes.titlesize": font_size + 1,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size - 1,
        "ytick.labelsize": font_size - 1,
        "legend.fontsize": font_size - 1,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": axes_linewidth,
        "axes.titleweight": "bold",
        "axes.titlepad": 8,
        "axes.labelpad": 6,
        "legend.frameon": False,
        "legend.handletextpad": 0.4,
        "legend.columnspacing": 1.2,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "xtick.major.pad": 4,
        "ytick.major.pad": 4,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.06,
    })


def finalize(fig, out: Path, also_png: bool = True, also_svg: bool = True,
             dpi: int = 300) -> None:
    """Save a figure as PDF (+ PNG preview + editable SVG for Illustrator)."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=1.1)
    fig.savefig(out, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    if also_svg:
        # Vector + editable text; the format to open in Adobe Illustrator.
        fig.savefig(out.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.05)
    if also_png:
        fig.savefig(out.with_suffix(".png"), dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def hide_spines(ax, sides: Iterable[str] = ("top", "right")) -> None:
    for s in sides:
        ax.spines[s].set_visible(False)


def style_bar_edges(bars, linewidth: float = 0.8, edgecolor: str = "black") -> None:
    for b in bars:
        try:
            b.set_edgecolor(edgecolor)
            b.set_linewidth(linewidth)
        except AttributeError:
            pass


def annotate_bars_above(ax, bars, fmt: str = "{:.2f}", fontsize: int = 10,
                        offset: float = 0.0, color: str = "black") -> None:
    for rect in bars:
        h = rect.get_height()
        if h == 0 or h is None:
            continue
        ax.text(rect.get_x() + rect.get_width() / 2, h + offset,
                fmt.format(h), ha="center", va="bottom",
                fontsize=fontsize, color=color)


def metro_legend(ax, metros: list[str], loc: str = "best", ncol: int = 1) -> None:
    """Build a consistent metro legend on `ax` (used for color-only legends)."""
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=METRO_COLORS[m], edgecolor="black",
                     linewidth=0.6, label=METRO_LABELS.get(m, m.upper()))
               for m in metros]
    ax.legend(handles=handles, loc=loc, ncol=ncol)
