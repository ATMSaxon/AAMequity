"""Shared style for the manuscript figures.

Figures are drawn at their printed size (full text width 6.6 in) with 7 pt
text, lowercase bold panel letters, and a restrained palette. Maps use UTM
coordinates, grey for unpopulated tracts, county lines, and a scale bar.
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
TIGER = ROOT / "data" / "raw" / "tiger"

FULL_WIDTH = 6.6

INK = "#262626"
MUTED = "#5A5A5A"
GRID = "#E6E6E6"
COUNTY_LINE = "#4D4D4D"
EXCLUDED = "#D9D9D9"
STATE_FILL = "#EFEFEF"
STATE_LINE = "#B5B5B5"

BLUE = "#0F4D92"
BLUE_MID = "#3775BA"
RED = "#B64342"
TEAL = "#42949E"
GREEN = "#5DA35D"
VIOLET = "#9A4D8E"
GOLD = "#D9A21B"
GREY = "#8C8C8C"

METRO_COLORS = {"la": BLUE, "dfw": TEAL, "sfbay": GREEN, "nyc": RED}
METRO_SHORT = {"la": "LA", "dfw": "DFW", "sfbay": "SF Bay", "nyc": "NYC"}
INCOME_COLORS = {"low_income": RED, "mid_income": GREY, "high_income": BLUE}

BLUES = LinearSegmentedColormap.from_list(
    "paper_blues", ["#F4F7FB", "#C9D9EC", "#7FA5D0", "#3775BA", "#0F4D92", "#08264A"])
REDS = LinearSegmentedColormap.from_list(
    "paper_reds", ["#FBF4F3", "#F0CFCB", "#E09E97", "#C9625A", "#A23A39", "#5E1A1C"])

DIV_GAIN = LinearSegmentedColormap.from_list(
    "paper_div_gain", ["#A23A39", "#E09E97", "#F7F7F7", "#7FA5D0", "#0F4D92"])
DIV_BURDEN = LinearSegmentedColormap.from_list(
    "paper_div_burden", ["#0F4D92", "#7FA5D0", "#F7F7F7", "#E09E97", "#A23A39"])

UTM = {"la": 32611, "dfw": 32614, "sfbay": 32610, "nyc": 32618}
LA_MAINLAND_SOUTH_LAT = 33.68  # excludes the offshore Channel Islands from LA views
STATE_TRACT_ZIPS = {"06": "tl_2024_06_tract.zip", "48": "tl_2024_48_tract.zip",
                    "36": "tl_2024_36_tract.zip", "34": "tl_2024_34_tract.zip"}


def setup() -> None:
    mpl.rcParams.update({
        "font.family": ["Helvetica", "Arial", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.titlesize": 7,
        "axes.labelsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "axes.linewidth": 0.6,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.major.pad": 2,
        "ytick.major.pad": 2,
        "axes.labelpad": 3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "legend.handletextpad": 0.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "mathtext.default": "regular",
    })


def panel_label(ax, letter: str, title: str = "", subtitle: str | None = None) -> None:
    top = 10 if subtitle else 2
    ax.annotate(letter, xy=(0, 1), xycoords="axes fraction", xytext=(0, top),
                textcoords="offset points", ha="left", va="bottom",
                fontsize=8, fontweight="bold", color=INK)
    if title:
        ax.annotate(title, xy=(0, 1), xycoords="axes fraction", xytext=(9, top),
                    textcoords="offset points", ha="left", va="bottom", fontsize=7, color=INK)
    if subtitle:
        ax.annotate(subtitle, xy=(0, 1), xycoords="axes fraction", xytext=(9, 2),
                    textcoords="offset points", ha="left", va="bottom", fontsize=6, color=MUTED)


def add_scalebar(ax, km: int = 20) -> None:
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    xs = x0 + 0.03 * (x1 - x0)
    ys = y0 + 0.03 * (y1 - y0)
    ax.plot([xs, xs + km * 1000], [ys, ys], color=INK, lw=1.0, solid_capstyle="butt")
    ax.annotate(f"{km} km", xy=(xs + km * 500, ys), xytext=(0, 2), textcoords="offset points",
                ha="center", va="bottom", fontsize=6, color=INK)


def load_tracts(metro: str) -> gpd.GeoDataFrame:
    return gpd.read_parquet(ROOT / "data" / "processed" / f"tracts_joined_{metro}.parquet")


def load_unpopulated(tracts: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Land tracts in the study counties that are not analysis tracts (no residents)."""
    counties = set(tracts["county_fips"])
    parts = []
    for st in sorted({c[:2] for c in counties}):
        t = gpd.read_file(f"zip://{TIGER / STATE_TRACT_ZIPS[st]}")
        t = t[(t["STATEFP"] + t["COUNTYFP"]).isin(counties)]
        parts.append(t[(~t["GEOID"].isin(set(tracts["tract_geoid"]))) & (t["ALAND"] > 0)])
    return gpd.GeoDataFrame(pd.concat(parts), crs=parts[0].crs).to_crs(tracts.crs)


class MetroBase:
    """Projected tracts, unpopulated tracts, counties and view extent for one metro."""

    def __init__(self, metro: str, tracts: gpd.GeoDataFrame | None = None):
        self.metro = metro
        g = tracts if tracts is not None else load_tracts(metro)
        self.tracts = g.to_crs(UTM[metro])
        self.empty = load_unpopulated(g).to_crs(UTM[metro])
        self.counties = self.tracts.dissolve(by="county_fips")
        self.shell = gpd.GeoSeries(
            [self.tracts.geometry.union_all().union(self.empty.geometry.union_all())],
            crs=self.tracts.crs)
        self.in_view = np.ones(len(self.tracts), dtype=bool)
        if metro == "la":
            lat = g.geometry.representative_point().y.to_numpy()
            self.in_view = lat > LA_MAINLAND_SOUTH_LAT
        x0, y0, x1, y1 = self.tracts[self.in_view].total_bounds
        self.extent = (x0 - 0.02 * (x1 - x0), x1 + 0.02 * (x1 - x0),
                       y0 - 0.12 * (y1 - y0), y1 + 0.02 * (y1 - y0))
        # Only geometry inside the view is drawn, so nothing sits clipped outside the panel.
        from shapely.geometry import box
        frame = box(self.extent[0], self.extent[2], self.extent[1], self.extent[3])
        self.empty = self.empty[self.empty.intersects(frame)]
        self.counties = self.counties.clip(frame)
        self.shell = self.shell.intersection(frame)

    @property
    def aspect(self) -> float:
        x0, x1, y0, y1 = self.extent
        return (x1 - x0) / (y1 - y0)

    def draw(self, ax, column: str | None = None, data: pd.Series | None = None, cmap=None,
             norm=None, fill: str = "#FFFFFF", county_lines: bool = True, scalebar: bool = True) -> None:
        self.empty.plot(ax=ax, color=EXCLUDED, linewidth=0)
        if column is None:
            self.tracts[self.in_view].plot(ax=ax, color=fill, linewidth=0)
        else:
            g = self.tracts.assign(_v=data.to_numpy() if data is not None else self.tracts[column])
            g = g[self.in_view]
            g.plot(ax=ax, column="_v", cmap=cmap, norm=norm, linewidth=0, rasterized=True,
                   missing_kwds={"color": EXCLUDED})
        if county_lines and len(self.counties) > 1:
            self.counties.boundary.plot(ax=ax, color=COUNTY_LINE, linewidth=0.3)
        self.shell.boundary.plot(ax=ax, color=INK, linewidth=0.4)
        x0, x1, y0, y1 = self.extent
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_aspect("equal")
        ax.set_axis_off()
        if scalebar:
            add_scalebar(ax)

    def points(self, lon, lat) -> tuple[np.ndarray, np.ndarray]:
        pts = gpd.GeoSeries(gpd.points_from_xy(lon, lat), crs=4326).to_crs(UTM[self.metro])
        return pts.x.to_numpy(), pts.y.to_numpy()


def horizontal_colorbar(fig, rect, cmap, norm, label: str, ticks=None, ticklabels=None,
                        extend: str = "neither"):
    cax = fig.add_axes(rect)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal", extend=extend)
    if ticks is not None:
        cb.set_ticks(ticks)
    if ticklabels is not None:
        cb.set_ticklabels(ticklabels)
    cb.ax.tick_params(labelsize=6, width=0.5, length=2, pad=1.5)
    cb.outline.set_linewidth(0.5)
    cb.ax.set_title(label, fontsize=6.5, pad=3, color=INK)
    return cax


def unpopulated_key(fig, x: float, y: float, h: float, label: str = "Unpopulated tract") -> None:
    fig.patches.append(mpl.patches.Rectangle((x, y), 0.012, h, transform=fig.transFigure,
                                             facecolor=EXCLUDED, edgecolor="none"))
    fig.text(x + 0.02, y + h / 2, label, fontsize=6, va="center", color=INK)


def save(fig, name: str) -> Path:
    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / f"{name}.pdf"
    fig.savefig(out, dpi=600, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return out
