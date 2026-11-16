"""
Real-data download orchestrator for AAMequity.

Downloads ALL inputs from authoritative public sources. NO synthetic data.
Sources verified 2026-05-03:
  TIGER 2024  (Census)            tract / county geographies
  ACS 5-yr    (Census API/FTP)    tract demographics
  LODES 8     (LEHD)              home-work O-D, WAC, RAC for CA (and DFW/SF/NY for robustness)
  GTFS        (LA Metro etc.)     transit network
  CalEnviroScreen 4.0 (OEHHA)     EJ vulnerability percentile
  HIFLD       (Geoplatform)       hospitals (vertiport medical-mission anchors)
  FAA NASR    (FAA)               heliport / airport facility data
  ACS 5-yr    (Census summary file) commute tables B08013, B08134, B08301, B08406
  OSM         (Geofabrik)         state road-network extracts

Each downloader is idempotent — checks if file already exists.
Run:  python scripts/1_download_data.py [--metro la|dfw|sfbay|nyc|all]
"""

from __future__ import annotations
import argparse
import hashlib
import os
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

# ---------- HTTP session with retries ---------------------------------------
def _session() -> requests.Session:
    s = requests.Session()
    retry = Retry(total=4, backoff_factor=2.0,
                  status_forcelist=[429, 500, 502, 503, 504],
                  allowed_methods=frozenset(["GET", "HEAD"]))
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.mount("http://", HTTPAdapter(max_retries=retry))
    s.headers["User-Agent"] = "AAMequity-research/1.0 (+https://github.com/ATMSaxon/AAMequity)"
    return s

SESSION = _session()


def _stream_download(url: str, dest: Path, expected_min_bytes: int = 1024) -> Path:
    """Download URL → dest with streaming, skip if already present and big enough."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size >= expected_min_bytes:
        print(f"  skip (cached, {dest.stat().st_size/1e6:.1f} MB): {dest.name}")
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  downloading: {url}")
    t0 = time.time()
    with SESSION.get(url, stream=True, timeout=60, allow_redirects=True) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 16):
                if chunk:
                    f.write(chunk)
        tmp.rename(dest)
    print(f"    -> {dest.name}  {dest.stat().st_size/1e6:.1f} MB in {time.time()-t0:.1f}s")
    return dest


def _try_urls(urls: list[str], dest: Path, expected_min_bytes: int = 1024) -> Path | None:
    """Try a list of fallback URLs for the same logical resource."""
    for u in urls:
        try:
            return _stream_download(u, dest, expected_min_bytes=expected_min_bytes)
        except Exception as e:  # noqa: BLE001 — fall through to next URL
            print(f"    fail: {u.split('?')[0][:80]}  ({type(e).__name__}: {str(e)[:120]})")
            continue
    print(f"    ALL URLS FAILED for {dest.name} — manual fetch required")
    return None


# ---------- 1. TIGER/Line geographies ---------------------------------------
TIGER_BASE = "https://www2.census.gov/geo/tiger/TIGER2024"

# State FIPS for our metros' anchor states
STATE_FIPS = {
    "ca": "06",  # California (LA + SF Bay)
    "tx": "48",  # Texas (Dallas-FW)
    "ny": "36",  # New York
    "nj": "34",  # NJ portion of NY metro
    "ct": "09",  # CT portion of NY metro
}

def download_tiger(states: list[str]) -> dict[str, Path]:
    """Census TIGER 2024 tract shapefiles per state."""
    print("\n[1] TIGER 2024 census-tract shapefiles")
    out = {}
    for st in states:
        fips = STATE_FIPS[st]
        url = f"{TIGER_BASE}/TRACT/tl_2024_{fips}_tract.zip"
        dest = RAW / "tiger" / f"tl_2024_{fips}_tract.zip"
        out[st] = _stream_download(url, dest, expected_min_bytes=200_000)
    # Counties (for clipping)
    url = f"{TIGER_BASE}/COUNTY/tl_2024_us_county.zip"
    out["counties"] = _stream_download(url, RAW / "tiger" / "tl_2024_us_county.zip")
    return out


# ---------- 2. LODES 8 (home-work O-D + WAC + RAC) --------------------------
LODES_BASE = "https://lehd.ces.census.gov/data/lodes/LODES8"
LODES_YEAR = "2021"  # latest stable

def download_lodes(states: list[str]) -> dict[str, list[Path]]:
    """LODES 8 OD (main jobs), WAC, RAC per state."""
    print(f"\n[2] LODES 8 ({LODES_YEAR}) OD/WAC/RAC")
    out = {}
    for st in states:
        files = []
        # Origin-destination main jobs (intra-state)
        for kind, dirname, fname in [
            ("od_main", "od", f"{st}_od_main_JT00_{LODES_YEAR}.csv.gz"),
            ("od_aux",  "od", f"{st}_od_aux_JT00_{LODES_YEAR}.csv.gz"),
            ("wac",     "wac", f"{st}_wac_S000_JT00_{LODES_YEAR}.csv.gz"),
            ("rac",     "rac", f"{st}_rac_S000_JT00_{LODES_YEAR}.csv.gz"),
        ]:
            url = f"{LODES_BASE}/{st}/{dirname}/{fname}"
            dest = RAW / "lodes" / fname
            try:
                files.append(_stream_download(url, dest, expected_min_bytes=10_000))
            except requests.HTTPError as e:
                print(f"    WARN {fname}: {e}")
        out[st] = files
    return out


# ---------- 3. GTFS feeds --------------------------------------------------
# Multiple fallback URLs per feed (sources move; mirror via transit.land or transitfeeds)
GTFS_FEEDS = {
    "la_metro_bus": [
        "https://gitlab.com/LACMTA/gtfs_bus/-/raw/master/gtfs_bus.zip",
        "https://api.metro.net/agencies/lametro/gtfs/",
        "https://transitfeeds.com/p/la-metro/175/latest/download",
    ],
    "la_metro_rail": [
        "https://gitlab.com/LACMTA/gtfs_rail/-/raw/master/gtfs_rail.zip",
        "https://api.metro.net/agencies/lametro-rail/gtfs/",
        "https://transitfeeds.com/p/la-metro-rail/176/latest/download",
    ],
    "dart": [
        "https://www.dart.org/transitdata/latest/google_transit.zip",
        "https://transitfeeds.com/p/dart/47/latest/download",
    ],
    "trinity_metro": [
        "https://ridetrinitymetro.org/wp-content/uploads/google_transit/google_transit.zip",
        "https://transitfeeds.com/p/the-t/76/latest/download",
    ],
    "bart": [
        "https://www.bart.gov/dev/schedules/google_transit.zip",
        "https://transitfeeds.com/p/bart/58/latest/download",
    ],
    "sfmta": [
        "https://gtfs.sfmta.com/transitdata/google_transit.zip",
        "https://transitfeeds.com/p/sfmta/60/latest/download",
    ],
    "mta_nyc_subway": [
        "http://web.mta.info/developers/data/nyct/subway/google_transit.zip",
        "https://transitfeeds.com/p/mta/79/latest/download",
    ],
}

def download_gtfs(metros: list[str]) -> dict[str, Path]:
    """GTFS feeds for selected metros."""
    print("\n[3] GTFS transit feeds")
    metro_to_feeds = {
        "la":    ["la_metro_bus", "la_metro_rail"],
        "dfw":   ["dart", "trinity_metro"],
        "sfbay": ["bart", "sfmta"],
        "nyc":   ["mta_nyc_subway"],
    }
    out = {}
    for m in metros:
        for feed_name in metro_to_feeds.get(m, []):
            urls = GTFS_FEEDS[feed_name]
            dest = RAW / "gtfs" / f"{feed_name}.zip"
            r = _try_urls(urls, dest, expected_min_bytes=50_000)
            if r is not None:
                out[feed_name] = r
    return out


# ---------- 4. CalEnviroScreen 4.0 -----------------------------------------
CES_URL = "https://oehha.ca.gov/sites/default/files/media/downloads/calenviroscreen/document/calenviroscreen40resultsdatadictionaryf2021.zip"

def download_ces() -> Path:
    """CalEnviroScreen 4.0 results + data dictionary (October 2021 release)."""
    print("\n[4] CalEnviroScreen 4.0 (OEHHA)")
    dest = RAW / "calenviroscreen" / "ces40_results.zip"
    try:
        return _stream_download(CES_URL, dest, expected_min_bytes=500_000)
    except requests.HTTPError as e:
        print(f"    WARN CES: {e} — try mirror at https://oehha.ca.gov/calenviroscreen/maps-data/download-data")
        return dest


# ---------- 5. Hospitals (HIFLD / fallbacks) -------------------------------
HIFLD_HOSPITALS_URLS = [
    "https://hifld-geoplatform.opendata.arcgis.com/api/download/v1/items/6ac5e325468c4cb9b905f1728d6fbf0f/csv?layers=0",
    "https://opendata.arcgis.com/api/v3/datasets/6ac5e325468c4cb9b905f1728d6fbf0f_0/downloads/data?format=geojson&spatialRefId=4326&where=1%3D1",
    "https://services1.arcgis.com/Hp6G80Pky0om7QvQ/arcgis/rest/services/Hospital/FeatureServer/0/query?where=1%3D1&outFields=*&f=geojson",
    # Fallback: CMS Hospital General Info (CSV with lat/lon)
    "https://data.cms.gov/provider-data/sites/default/files/resources/092256becd267d9eeccf73bf7d16c46b_1715094000/Hospital_General_Information.csv",
]

def download_hospitals() -> Path | None:
    """HIFLD or CMS US hospitals — vertiport medical-mission anchors."""
    print("\n[5] Hospitals (HIFLD → CMS fallback)")
    dest = RAW / "hifld" / "hospitals.geojson"
    return _try_urls(HIFLD_HOSPITALS_URLS, dest, expected_min_bytes=200_000)


# ---------- 6. FAA airports / heliports ------------------------------------
# FAA Aircraft Facilities CSV (no auth required) — extracted from NASR
# Endpoint: https://adip.faa.gov/publicApi/v1/airports
# Also available: aviation.dev mirror, BTS T-100 etc.
FAA_AIRPORTS_URL = "https://adip.faa.gov/publicApi/v1/airports?format=csv&type=ALL"

def download_faa_airports() -> Path:
    """FAA airport / heliport facility list (CSV)."""
    print("\n[6] FAA airport+heliport facilities")
    dest = RAW / "faa" / "faa_airports.csv"
    try:
        return _stream_download(FAA_AIRPORTS_URL, dest, expected_min_bytes=100_000)
    except requests.HTTPError as e:
        print(f"    WARN FAA API: {e}")
        # Fallback to OurAirports.com (open data, sourced from FAA + community)
        alt = "https://davidmegginson.github.io/ourairports-data/airports.csv"
        print(f"    fallback: OurAirports → {alt}")
        return _stream_download(alt, dest, expected_min_bytes=1_000_000)


# ---------- 7. LA County boundary (for clipping) ---------------------------
LA_BOUNDARY_URLS = [
    "https://opendata.arcgis.com/datasets/0c0d0adc5d3946f7b86347405e0b3289_27.geojson",  # LACounty Boundary
    "https://services.arcgis.com/RmCCgQtiZLDFtsAn/arcgis/rest/services/County_Boundary/FeatureServer/27/query?where=1%3D1&outFields=*&f=geojson",
    "https://services1.arcgis.com/0VnNnJp3Fa7jM8MS/arcgis/rest/services/County_Boundaries/FeatureServer/0/query?where=COUNTY%3D%27LOS+ANGELES%27&outFields=*&f=geojson",
]

def download_la_boundary() -> Path | None:
    """LA County boundary (multiple fallback feature services)."""
    print("\n[7] LA County boundary")
    dest = RAW / "la_county" / "la_county_boundary.geojson"
    return _try_urls(LA_BOUNDARY_URLS, dest, expected_min_bytes=5_000)


# ---------- 8. ACS via Census API ------------------------------------------
def download_acs(states: list[str], year: int = 2022) -> dict[str, Path]:
    """
    ACS 5-year tract-level demographics for the variables we need.

    Variables map (ACS 5-yr table B-codes):
      B19013_001E  Median household income
      B17001_002E  Pop in poverty
      B17001_001E  Pop universe (poverty)
      B03002_*     Race/Hispanic origin
      B25044_003E  Owner-occupied no vehicle
      B25044_010E  Renter-occupied no vehicle
      B08301_010E  Commute by public transit
      B08301_001E  Commute total
      B01001_*     Age structure (incl. 65+)
      B18101_*     Disability status
      B25003_003E  Renter-occupied housing units
      B25070_*     Gross rent as % income
      B16004_*     Language isolation
    """
    print(f"\n[8] ACS {year} 5-year tract demographics via Census API")
    api_key = os.environ.get("CENSUS_API_KEY", "")
    if not api_key:
        print("    NOTE: no CENSUS_API_KEY set — Census API allows ≤500 calls/day without key,")
        print("          which IS enough for tract-level pulls per state. Sign up free:")
        print("          https://api.census.gov/data/key_signup.html")

    base = f"https://api.census.gov/data/{year}/acs/acs5"
    variables = [
        "NAME",
        "B19013_001E",  # median HH income
        "B17001_002E", "B17001_001E",  # poverty num/den
        "B03002_001E", "B03002_003E", "B03002_004E", "B03002_005E", "B03002_006E", "B03002_012E",  # race/hisp
        "B25044_003E", "B25044_010E",  # zero-vehicle (owner + renter)
        "B25044_001E",  # vehicle-availability universe
        "B08301_010E", "B08301_001E",  # transit commute
        "B01001_001E",  # total pop (age universe)
        "B01001_020E", "B01001_021E", "B01001_022E", "B01001_023E", "B01001_024E", "B01001_025E",  # male 65+
        "B01001_044E", "B01001_045E", "B01001_046E", "B01001_047E", "B01001_048E", "B01001_049E",  # female 65+
        "B18101_001E", "B18101_004E", "B18101_007E", "B18101_010E", "B18101_013E", "B18101_016E",  # disability (sample)
        "B25003_003E", "B25003_001E",  # renter-occupied / occupied housing
        "B16004_003E", "B16004_001E",  # speak only English / total 5+
    ]
    var_str = ",".join(variables)

    out = {}
    for st in states:
        fips = STATE_FIPS[st]
        url = f"{base}?get={var_str}&for=tract:*&in=state:{fips}"
        if api_key:
            url += f"&key={api_key}"
        dest = RAW / "acs" / f"acs5_{year}_state{fips}_tract.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and dest.stat().st_size > 100_000:
            print(f"  skip (cached): {dest.name}")
            out[st] = dest
            continue
        print(f"  GET ACS {year} state={fips} ({len(variables)} variables)")
        try:
            r = SESSION.get(url, timeout=120)
            r.raise_for_status()
            dest.write_bytes(r.content)
            print(f"    -> {dest.name}  {dest.stat().st_size/1e6:.2f} MB")
            out[st] = dest
        except Exception as e:
            print(f"    ERROR {st}: {e}")
    return out


# ---------- 9. LA County parcels — for vertiport candidate filtering -------
# LA County Assessor parcel point-level data is huge (~5GB).
# Use the simpler "Land Use" parcel summary (smaller) for compatibility filtering.
LA_LANDUSE_URL = (
    "https://opendata.arcgis.com/api/v3/datasets/"
    "8eb1f324a0814552b4d6d59f12d6f10c_4/downloads/data?format=geojson&spatialRefId=4326&where=1%3D1"
)

LA_LANDUSE_URLS = [
    LA_LANDUSE_URL,
    # Alternative: SCAG (Southern California Association of Governments) regional land-use 2019
    "https://services1.arcgis.com/tp9wqSVX1AVPHr8c/arcgis/rest/services/SCAG_LU_2019/FeatureServer/0/query?where=COUNTY%3D%27Los+Angeles%27&outFields=*&f=geojson&resultRecordCount=2000",
]

def download_la_landuse() -> Path | None:
    """LA County parcel land-use polygons (LARIAC / Assessor or SCAG)."""
    print("\n[9] LA County parcel land use")
    dest = RAW / "la_county" / "la_landuse.geojson"
    return _try_urls(LA_LANDUSE_URLS, dest, expected_min_bytes=500_000)


# ---------- ACS commute tables (summary file) --------------------------------
ACS_SF_BASE = ("https://www2.census.gov/programs-surveys/acs/summary_file/2022/"
               "table-based-SF/data/5YRData")
ACS_COMMUTE_TABLES = {"b08013": "tract", "b08134": "tract", "b08301": "tract", "b08406": "county"}
SUMLEV_PREFIX = {"tract": "1400000US", "county": "0500000US"}

def download_acs_commute(states: list[str]) -> dict[str, Path]:
    """ACS 2018-2022 commute tables for the metro states, filtered from the national summary file."""
    print("\n[10] ACS commute tables (summary file)")
    out = {}
    fips = sorted(STATE_FIPS[s] for s in states)
    for table, level in ACS_COMMUTE_TABLES.items():
        dest = RAW / "acs_commute" / f"acs5_2022_{table}_{level}_5states.psv"
        if dest.exists():
            print(f"  skip (cached): {dest.name}")
            out[table] = dest
            continue
        prefixes = tuple(SUMLEV_PREFIX[level] + f for f in fips)
        full = _stream_download(f"{ACS_SF_BASE}/acsdt5y2022-{table}.dat",
                                RAW / "acs_commute" / f"acsdt5y2022-{table}.dat")
        with open(full) as src, open(dest, "w") as dst:
            dst.write(src.readline())
            for line in src:
                if line.startswith(prefixes):
                    dst.write(line)
        full.unlink()
        out[table] = dest
    return out


# ---------- OSM road network (Geofabrik) -----------------------------------
GEOFABRIK = "https://download.geofabrik.de/north-america/us"
OSM_EXTRACTS = {
    "la": ["california/socal"], "sfbay": ["california/norcal"], "dfw": ["texas"],
    "nyc": ["new-york", "new-jersey"],
}

def download_osm(metros: list[str]) -> dict[str, Path]:
    """Geofabrik state extracts used by src/network_od.py."""
    print("\n[11] OpenStreetMap extracts (Geofabrik)")
    out = {}
    for m in metros:
        for region in OSM_EXTRACTS[m]:
            name = region.split("/")[-1]
            out[name] = _stream_download(f"{GEOFABRIK}/{region}-latest.osm.pbf",
                                         RAW / "osm" / f"{name}-latest.osm.pbf",
                                         expected_min_bytes=10_000_000)
    return out


# ---------- main -----------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metro", choices=["la", "dfw", "sfbay", "nyc", "all"],
                    default="la",
                    help="Which metro(s) to fetch data for. 'all' = LA primary + 3 robustness metros.")
    ap.add_argument("--skip-large", action="store_true",
                    help="Skip large downloads (LA parcel land-use, full LODES OD).")
    args = ap.parse_args()

    if args.metro == "all":
        metros = ["la", "dfw", "sfbay", "nyc"]
    else:
        metros = [args.metro]

    state_set = set()
    for m in metros:
        if m == "la" or m == "sfbay":
            state_set.add("ca")
        if m == "dfw":
            state_set.add("tx")
        if m == "nyc":
            state_set.update(["ny", "nj", "ct"])
    states = sorted(state_set)
    print(f"\n=== AAMequity data download | metros={metros} | states={states} ===")

    # Order: small-and-essential first, large last. Each step is independently fault-tolerant.
    steps = [
        ("la_boundary",   lambda: download_la_boundary()),
        ("ces",           lambda: download_ces()),
        ("faa_airports",  lambda: download_faa_airports()),
        ("hospitals",     lambda: download_hospitals()),
        ("tiger",         lambda: download_tiger(states)),
        ("acs",           lambda: download_acs(states)),
        ("gtfs",          lambda: download_gtfs(metros)),
        ("acs_commute",   lambda: download_acs_commute(states)),
        ("lodes",         lambda: download_lodes(states)),
        ("osm",           lambda: download_osm(metros)),
    ]
    if not args.skip_large:
        steps.append(("la_landuse", lambda: download_la_landuse()))

    failures = []
    for name, fn in steps:
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            print(f"  ! step '{name}' failed: {type(e).__name__}: {e}")
            failures.append(name)
    if failures:
        print(f"\nNOTE: failed steps (continuing anyway): {failures}")

    print("\n=== download done ===")
    print(f"raw data dir: {RAW}")
    sizes = []
    for sub in sorted(RAW.iterdir()):
        if sub.is_dir():
            total = sum(f.stat().st_size for f in sub.rglob("*") if f.is_file())
            sizes.append((sub.name, total))
    print("\nSummary:")
    for name, sz in sizes:
        print(f"  {name:25s}  {sz/1e6:8.1f} MB")
    print(f"  {'TOTAL':25s}  {sum(s for _, s in sizes)/1e6:8.1f} MB")


if __name__ == "__main__":
    main()
