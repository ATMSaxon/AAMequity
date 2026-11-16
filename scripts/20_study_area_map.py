"""Study-area map: location of the four metros and tract population density.

Panel a locates the four study areas in the contiguous United States.
Panels b-e show each study area's census tracts shaded by residential
population density, with county boundaries.

Output: figures/fig_study_area.{pdf,png,svg}
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm

FIGURES = ROOT / "figures"
PROCESSED = ROOT / "data" / "processed"
STATES_ZIP = ROOT / "data" / "raw" / "tiger" / "cb_2023_us_state_20m.zip"

METROS = ["la", "dfw", "sfbay", "nyc"]
NAMES = {
    "la": "Los Angeles County",
    "dfw": "Dallas–Fort Worth",
    "sfbay": "San Francisco Bay Area",
    "nyc": "New York metropolitan area",
}
SHORT = {"la": "LA", "dfw": "DFW", "sfbay": "SF Bay", "nyc": "NYC"}
UTM = {"la": 32611, "dfw": 32614, "sfbay": 32610, "nyc": 32618}
SCALEBAR_KM = {"la": 20, "dfw": 20, "sfbay": 20, "nyc": 20}
CONUS_EXCLUDE = {"02", "15", "60", "66", "69", "72", "78"}

INK = "#262626"
COUNTY_LINE = "#4D4D4D"
STATE_FILL = "#EFEFEF"
STATE_LINE = "#B5B5B5"
EXCLUDED = "#D9D9D9"
DENSITY_CMAP = LinearSegmentedColormap.from_list(
    "density", ["#F4F7FB", "#C9D9EC", "#7FA5D0", "#3775BA", "#0F4D92", "#08264A"])
DENSITY_NORM = LogNorm(vmin=10, vmax=30000)


def load_states() -> gpd.GeoDataFrame:
    with zipfile.ZipFile(STATES_ZIP) as z:
        shp = [n for n in z.namelist() if n.endswith(".shp")][0]
    states = gpd.read_file(f"zip://{STATES_ZIP}!{shp}")
    return states[~states["STATEFP"].isin(CONUS_EXCLUDE)].to_crs(5070)


def load_tracts(metro: str) -> gpd.GeoDataFrame:
    g = gpd.read_parquet(PROCESSED / f"tracts_joined_{metro}.parquet")
    g = g[["tract_geoid", "county_fips", "pop", "land_area_m2", "geometry"]].copy()
    land_km2 = g["land_area_m2"] / 1e6
    g["density"] = np.where(land_km2 > 0, g["pop"] / land_km2, np.nan)
    return g


STATE_TRACT_ZIPS = {"06": "tl_2024_06_tract.zip", "48": "tl_2024_48_tract.zip",
                    "36": "tl_2024_36_tract.zip", "34": "tl_2024_34_tract.zip"}


def load_unpopulated(g: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Land tracts in the study counties that are not analysis tracts (no residents)."""
    counties = set(g["county_fips"])
    parts = []
    for st in sorted({c[:2] for c in counties}):
        t = gpd.read_file(f"zip://{ROOT / 'data' / 'raw' / 'tiger' / STATE_TRACT_ZIPS[st]}")
        t = t[(t["STATEFP"] + t["COUNTYFP"]).isin(counties)]
        parts.append(t[(~t["GEOID"].isin(set(g["tract_geoid"]))) & (t["ALAND"] > 0)])
    out = gpd.GeoDataFrame(gpd.pd.concat(parts), crs=parts[0].crs)
    return out.to_crs(g.crs)


def add_scalebar(ax, km: int) -> None:
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    xs = x0 + 0.03 * (x1 - x0)
    ys = y0 + 0.03 * (y1 - y0)
    ax.plot([xs, xs + km * 1000], [ys, ys], color=INK, lw=1.0, solid_capstyle="butt")
    ax.annotate(f"{km} km", xy=(xs + km * 500, ys), xytext=(0, 2), textcoords="offset points",
                ha="center", va="bottom", fontsize=6, color=INK)


def panel_label(ax, letter: str, title: str, subtitle: str | None = None) -> None:
    top = 10 if subtitle else 2
    ax.annotate(letter, xy=(0, 1), xycoords="axes fraction", xytext=(0, top),
                textcoords="offset points", ha="left", va="bottom",
                fontsize=8, fontweight="bold", color=INK)
    ax.annotate(title, xy=(0, 1), xycoords="axes fraction", xytext=(9, top),
                textcoords="offset points", ha="left", va="bottom", fontsize=7, color=INK)
    if subtitle:
        ax.annotate(subtitle, xy=(0, 1), xycoords="axes fraction", xytext=(9, 2),
                    textcoords="offset points", ha="left", va="bottom",
                    fontsize=6, color="#5A5A5A")


def overview(ax, states, tracts_by_metro) -> None:
    states.plot(ax=ax, color=STATE_FILL, edgecolor=STATE_LINE, linewidth=0.35)
    offsets = {"la": (-330e3, -90e3), "dfw": (0, -260e3), "sfbay": (-360e3, 60e3),
               "nyc": (90e3, -280e3)}
    for m, g in tracts_by_metro.items():
        area = g.to_crs(5070).dissolve()
        area.plot(ax=ax, color="#0F4D92", edgecolor="#0F4D92", linewidth=0.6)
        c = area.geometry.iloc[0].centroid
        dx, dy = offsets[m]
        ax.annotate(SHORT[m], xy=(c.x, c.y), xytext=(c.x + dx, c.y + dy),
                    fontsize=7, color=INK, ha="center", va="center",
                    arrowprops=dict(arrowstyle="-", color=INK, lw=0.5, shrinkA=1, shrinkB=2))
    ax.set_axis_off()
    ax.set_aspect("equal")


def metro_panel(ax, metro: str, g: gpd.GeoDataFrame) -> None:
    gp = g.to_crs(UTM[metro])
    empty = load_unpopulated(g).to_crs(UTM[metro])
    empty.plot(ax=ax, color=EXCLUDED, linewidth=0)
    shell = gpd.GeoSeries([gp.geometry.union_all().union(empty.geometry.union_all())], crs=gp.crs)
    gp.plot(ax=ax, column="density", cmap=DENSITY_CMAP, norm=DENSITY_NORM,
            linewidth=0, rasterized=True, missing_kwds={"color": EXCLUDED})
    gp.dissolve(by="county_fips").boundary.plot(ax=ax, color=COUNTY_LINE, linewidth=0.3)
    shell.boundary.plot(ax=ax, color=INK, linewidth=0.4)
    ax.set_axis_off()
    ax.set_aspect("equal")
    if metro == "la":
        # Mainland extent; the offshore Channel Islands tracts are not drawn.
        mainland = gp[gp.geometry.centroid.y > gp.total_bounds[1] + 0.3 * (gp.total_bounds[3] - gp.total_bounds[1])]
        x0, y0, x1, y1 = mainland.total_bounds
    else:
        x0, y0, x1, y1 = gp.total_bounds
    pad_x, pad_y = 0.02 * (x1 - x0), 0.02 * (y1 - y0)
    ax.set_xlim(x0 - pad_x, x1 + pad_x)
    ax.set_ylim(y0 - 0.12 * (y1 - y0), y1 + pad_y)
    add_scalebar(ax, SCALEBAR_KM[metro])


def aspect(metro: str, g: gpd.GeoDataFrame) -> float:
    gp = g.to_crs(UTM[metro])
    if metro == "la":
        gp = gp[gp.geometry.centroid.y > gp.total_bounds[1] + 0.3 * (gp.total_bounds[3] - gp.total_bounds[1])]
    x0, y0, x1, y1 = gp.total_bounds
    return 1.04 * (x1 - x0) / (1.14 * (y1 - y0))


def build_figure():
    mpl.rcParams.update({
        "font.family": ["Helvetica", "Arial", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })
    states = load_states()
    tracts = {m: load_tracts(m) for m in METROS}

    sb = states.total_bounds
    us_aspect = (sb[2] - sb[0]) / (sb[3] - sb[1])
    asp = {m: aspect(m, tracts[m]) for m in METROS}

    width = 6.6
    gap = 0.25
    top_h = (width - gap) / (us_aspect + asp["la"])
    bot_h = (width - 2 * gap) / (asp["dfw"] + asp["sfbay"] + asp["nyc"])
    head, cbar_h = 0.22, 0.55
    height = head + top_h + head + bot_h + cbar_h

    fig = plt.figure(figsize=(width, height))
    def box(x, y, w, h):
        return fig.add_axes([x / width, y / height, w / width, h / height])

    y_bot = cbar_h
    y_top = cbar_h + bot_h + head
    ax_us = box(0, y_top, us_aspect * top_h, top_h)
    overview(ax_us, states, tracts)
    panel_label(ax_us, "a", "Study areas in the contiguous United States")

    ax = box(us_aspect * top_h + gap, y_top, asp["la"] * top_h, top_h)
    axes = {"la": ax}
    x = 0.0
    for m in ["dfw", "sfbay", "nyc"]:
        axes[m] = box(x, y_bot, asp[m] * bot_h, bot_h)
        x += asp[m] * bot_h + gap
    for letter, m in zip("bcde", METROS):
        metro_panel(axes[m], m, tracts[m])
        panel_label(axes[m], letter, NAMES[m])

    cax = fig.add_axes([0.30, 0.30 / height, 0.40, 0.08 / height])
    sm = mpl.cm.ScalarMappable(norm=DENSITY_NORM, cmap=DENSITY_CMAP)
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal", extend="both")
    cb.set_ticks([10, 100, 1000, 10000])
    cb.set_ticklabels(["10", "100", "1,000", "10,000"])
    cb.ax.tick_params(labelsize=6, width=0.5, length=2, pad=1.5)
    cb.outline.set_linewidth(0.5)
    cb.ax.set_title("Residents per km² of land", fontsize=6.5, pad=3)
    fig.text(0.745, 0.34 / height, "Unpopulated tract", fontsize=6, va="center", color=INK)
    fig.patches.append(mpl.patches.Rectangle((0.725, 0.30 / height), 0.012, 0.08 / height,
                       transform=fig.transFigure, facecolor=EXCLUDED, edgecolor="none"))

    return fig, [ax_us] + [axes[m] for m in METROS], cax


def main() -> None:
    fig, _, _ = build_figure()
    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / "fig_study_area.pdf"
    fig.savefig(out, dpi=600, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
