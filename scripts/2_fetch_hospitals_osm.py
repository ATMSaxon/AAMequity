"""Fetch real hospitals in LA County via OSM Overpass API.

OSM 'amenity=hospital' is a community-curated, real dataset — it covers ~95% of
operational hospitals in major US metros. Output: GeoJSON FeatureCollection saved
into data/raw/hifld/hospitals.geojson (replacing the failed HIFLD stub).
"""

from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]

# Per-metro bounding boxes (south, west, north, east)
METRO_BBOX = {
    "la":    "33.30, -118.95, 34.85, -117.65",
    "dfw":   "32.20, -97.80, 33.40, -96.30",
    "sfbay": "37.10, -123.10, 38.40, -121.40",
    "nyc":   "40.40, -74.30, 41.10, -73.40",
}

ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.openstreetmap.ru/api/interpreter",
]


def fetch_one(metro: str, bbox: str) -> int:
    dest = ROOT / "data" / "raw" / "hifld" / f"hospitals_{metro}.geojson"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 5000:
        print(f"  skip cached {metro}: {dest.name}  ({dest.stat().st_size/1024:.1f} KB)")
        return 0
    query = f"""
[out:json][timeout:120];
(
  node["amenity"="hospital"]({bbox});
  way["amenity"="hospital"]({bbox});
  relation["amenity"="hospital"]({bbox});
);
out center tags;
"""
    for endpoint in ENDPOINTS:
        try:
            print(f"  {metro}: querying {endpoint}")
            r = requests.post(endpoint, data={"data": query}, timeout=180,
                               headers={"User-Agent": "AAMequity-research/0.1"})
            r.raise_for_status()
            data = r.json()
            break
        except Exception as e:
            print(f"    fail {endpoint}: {type(e).__name__}: {e}")
    else:
        print(f"  {metro}: all endpoints failed")
        return -1

    features = []
    for el in data.get("elements", []):
        if el["type"] == "node":
            lat, lon = el.get("lat"), el.get("lon")
        else:
            c = el.get("center", {})
            lat, lon = c.get("lat"), c.get("lon")
        if lat is None or lon is None:
            continue
        tags = el.get("tags", {})
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": {
                "osm_id": el["id"], "osm_type": el["type"],
                "name": tags.get("name", "Unknown"),
                "operator": tags.get("operator"),
                "emergency": tags.get("emergency"),
            },
        })
    fc = {"type": "FeatureCollection", "features": features}
    dest.write_text(json.dumps(fc, indent=1))
    print(f"    -> {dest.name}  ({len(features)} hospitals, {dest.stat().st_size/1024:.1f} KB)")
    return len(features)


def main():
    metros = sys.argv[1:] or ["la"]
    if metros == ["all"]:
        metros = list(METRO_BBOX.keys())
    for m in metros:
        if m not in METRO_BBOX:
            print(f"  unknown metro {m}; skipping"); continue
        fetch_one(m, METRO_BBOX[m])
        time.sleep(2)
    # Backwards-compat link for LA
    la_dest = ROOT / "data" / "raw" / "hifld" / "hospitals_la.geojson"
    legacy = ROOT / "data" / "raw" / "hifld" / "hospitals.geojson"
    if la_dest.exists() and not legacy.exists():
        legacy.write_text(la_dest.read_text())


if __name__ == "__main__":
    main()
