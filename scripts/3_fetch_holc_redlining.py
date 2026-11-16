"""Download REAL HOLC (Home Owners' Loan Corporation) redlining polygons for LA / DFW / SF Bay / NYC.

Source: University of Richmond "Mapping Inequality" project — open GeoJSON via
https://dsl.richmond.edu/panorama/redlining/data — provides 1930s HOLC grade polygons
(A=Best/green, B=Still Desirable/blue, C=Definitely Declining/yellow, D=Hazardous/red).

Used for the acoustic-redlining test (Casey/Nardone 2024 follow-up): does AAM burden
disproportionately fall on historically D-graded ("red") tracts?
"""

from __future__ import annotations
import json
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "raw" / "holc"
DEST.mkdir(parents=True, exist_ok=True)

# Mapping Inequality city slugs (verified 2026-05) — only large metros relevant to project
CITIES = {
    "la":     ["CALosAngeles1939"],
    "dfw":    ["TXDallas1937", "TXFortWorth1934"],
    "sfbay":  ["CASanFrancisco1937", "CAOakland1937", "CASanJose1937"],
    "nyc":    ["NYBrooklyn1938", "NYBronx1939", "NYManhattan1937", "NYQueens1938",
               "NYStatenIsland1940"],
}

URL_TEMPLATE = "https://dsl.richmond.edu/panorama/redlining/static/citiesData/{slug}/geojson.json"


def fetch_city(slug: str) -> dict | None:
    """Download one city's HOLC GeoJSON. Returns parsed dict or None on failure."""
    url = URL_TEMPLATE.format(slug=slug)
    try:
        r = requests.get(url, timeout=60,
                          headers={"User-Agent": "AAMequity-research/0.1"})
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"  fail {slug}: {type(e).__name__}: {e}")
        return None


def main():
    metros = sys.argv[1:] or ["la"]
    if metros == ["all"]:
        metros = list(CITIES.keys())
    for metro in metros:
        if metro not in CITIES:
            print(f"  unknown metro {metro}; skipping")
            continue
        out = DEST / f"holc_{metro}.geojson"
        all_features = []
        for slug in CITIES[metro]:
            print(f"  fetching {slug} ...")
            data = fetch_city(slug)
            if not data:
                continue
            feats = data.get("features", []) if isinstance(data, dict) else []
            print(f"    -> {len(feats)} polygons")
            all_features.extend(feats)
        fc = {"type": "FeatureCollection", "features": all_features}
        out.write_text(json.dumps(fc))
        print(f"  saved {len(all_features)} polygons to {out} ({out.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
