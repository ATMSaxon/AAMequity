"""Load REAL downloaded data into clean GeoDataFrames / DataFrames.

All loaders read from data/raw/, write canonical processed versions to data/processed/.
NO synthetic data ever. If a required file is missing, raise FileNotFoundError with the
download command the user must run.
"""

from __future__ import annotations
import json
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from .config import RAW, PROC, LA_COUNTY_FIPS, METRO_COUNTY_FIPS


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def _require(path: Path, hint: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required input: {path}\n"
            f"Download with: python scripts/1_download_data.py {hint}"
        )
    return path


def _county_filter(metro: str = "la") -> set[str]:
    """Set of county-FIPS strings for the metro."""
    return set(METRO_COUNTY_FIPS[metro])


# ----------------------------------------------------------------------------
# 1. TIGER census tract geometries
# ----------------------------------------------------------------------------
def load_tracts(metro: str = "la", refresh: bool = False) -> gpd.GeoDataFrame:
    """Tract polygons clipped to the metro's counties. Returns EPSG:4326."""
    out = PROC / f"tracts_{metro}.parquet"
    if out.exists() and not refresh:
        return gpd.read_parquet(out)

    counties = _county_filter(metro)
    states_needed = {fips[:2] for fips in counties}

    parts = []
    for st in states_needed:
        zip_path = _require(RAW / "tiger" / f"tl_2024_{st}_tract.zip",
                            "--metro " + metro)
        gdf = gpd.read_file(f"zip://{zip_path}")
        gdf["county_fips"] = gdf["STATEFP"] + gdf["COUNTYFP"]
        gdf = gdf[gdf["county_fips"].isin(counties)].copy()
        parts.append(gdf)
    out_gdf = pd.concat(parts, ignore_index=True)

    # Standard column normalization
    out_gdf = out_gdf.rename(columns={
        "GEOID": "tract_geoid",
        "STATEFP": "state_fips",
        "COUNTYFP": "county_code",
        "ALAND": "land_area_m2",
        "INTPTLAT": "lat", "INTPTLON": "lon",
    })
    out_gdf["lat"] = pd.to_numeric(out_gdf["lat"], errors="coerce")
    out_gdf["lon"] = pd.to_numeric(out_gdf["lon"], errors="coerce")
    out_gdf = out_gdf[["tract_geoid", "state_fips", "county_fips", "county_code",
                       "land_area_m2", "lat", "lon", "geometry"]]
    out_gdf = out_gdf.set_crs("EPSG:4269", allow_override=True).to_crs("EPSG:4326")
    out_gdf.to_parquet(out)
    return out_gdf


# ----------------------------------------------------------------------------
# 2. ACS demographics
# ----------------------------------------------------------------------------
ACS_VARS = {
    "B19013_001E": "median_hh_income",
    "B17001_002E": "n_in_poverty", "B17001_001E": "n_pov_universe",
    "B03002_001E": "n_total_pop_race",
    "B03002_003E": "n_white_nh", "B03002_004E": "n_black_nh",
    "B03002_005E": "n_aian_nh", "B03002_006E": "n_asian_nh",
    "B03002_012E": "n_hispanic",
    "B25044_003E": "owner_no_vehicle", "B25044_010E": "renter_no_vehicle",
    "B25044_001E": "vehicle_universe",
    "B08301_010E": "transit_commute", "B08301_001E": "commute_total",
    "B01001_001E": "n_total_pop_age",
    "B01001_020E": "m65_66", "B01001_021E": "m67_69", "B01001_022E": "m70_74",
    "B01001_023E": "m75_79", "B01001_024E": "m80_84", "B01001_025E": "m85p",
    "B01001_044E": "f65_66", "B01001_045E": "f67_69", "B01001_046E": "f70_74",
    "B01001_047E": "f75_79", "B01001_048E": "f80_84", "B01001_049E": "f85p",
    "B18101_001E": "n_disab_universe",
    "B18101_004E": "m_under5_disab", "B18101_007E": "m5_17_disab",
    "B18101_010E": "m18_34_disab", "B18101_013E": "m35_64_disab",
    "B18101_016E": "m65_74_disab",
    "B25003_003E": "renter_occupied", "B25003_001E": "occupied_units",
    "B16004_003E": "english_only_5p", "B16004_001E": "language_universe_5p",
}


def load_acs(metro: str = "la", year: int = 2022, refresh: bool = False) -> pd.DataFrame:
    """Load ACS 5-year tract data, compute derived equity-relevant features."""
    out = PROC / f"acs_{metro}_{year}.parquet"
    if out.exists() and not refresh:
        return pd.read_parquet(out)

    counties = _county_filter(metro)
    states_needed = {fips[:2] for fips in counties}
    parts = []
    for st in states_needed:
        path = _require(RAW / "acs" / f"acs5_{year}_state{st}_tract.json",
                        "--metro " + metro)
        raw = json.loads(path.read_text())
        cols = raw[0]
        df = pd.DataFrame(raw[1:], columns=cols)
        df["tract_geoid"] = df["state"] + df["county"] + df["tract"]
        df["county_fips"] = df["state"] + df["county"]
        df = df[df["county_fips"].isin(counties)].copy()
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)

    # Convert ACS fields to numeric (Census uses "" or negative codes for missing)
    for code, friendly in ACS_VARS.items():
        if code in df.columns:
            df[friendly] = pd.to_numeric(df[code], errors="coerce")
            df.loc[df[friendly] < 0, friendly] = np.nan
    df = df[["tract_geoid", "county_fips"] + list(ACS_VARS.values())]

    # Derived features
    df["pct_minority"] = (df[["n_black_nh", "n_aian_nh", "n_asian_nh", "n_hispanic"]].sum(axis=1)
                          / df["n_total_pop_race"]).clip(0, 1)
    df["pct_poverty"] = (df["n_in_poverty"] / df["n_pov_universe"]).clip(0, 1)
    df["pct_zero_vehicle"] = ((df["owner_no_vehicle"] + df["renter_no_vehicle"])
                               / df["vehicle_universe"]).clip(0, 1)
    df["pct_transit_commute"] = (df["transit_commute"] / df["commute_total"]).clip(0, 1)
    df["pct_renter"] = (df["renter_occupied"] / df["occupied_units"]).clip(0, 1)
    df["pct_65_plus"] = (
        df[["m65_66","m67_69","m70_74","m75_79","m80_84","m85p",
            "f65_66","f67_69","f70_74","f75_79","f80_84","f85p"]].sum(axis=1)
        / df["n_total_pop_age"]
    ).clip(0, 1)
    df["pct_disab_sample"] = (
        df[["m_under5_disab","m5_17_disab","m18_34_disab","m35_64_disab","m65_74_disab"]].sum(axis=1)
        / df["n_disab_universe"]
    ).clip(0, 1)
    df["pct_lang_isolated"] = (1.0 - df["english_only_5p"] / df["language_universe_5p"]).clip(0, 1)

    df["pop"] = df["n_total_pop_age"]
    df.to_parquet(out)
    return df


# ----------------------------------------------------------------------------
# 3. LODES home-work O-D + WAC + RAC
# ----------------------------------------------------------------------------
def load_lodes_od(metro: str = "la", refresh: bool = False) -> pd.DataFrame:
    """LODES 8 OD intra-state main jobs aggregated to tract pairs."""
    out = PROC / f"lodes_od_{metro}.parquet"
    if out.exists() and not refresh:
        return pd.read_parquet(out)

    counties = _county_filter(metro)
    states_needed = {fips[:2] for fips in counties}
    state_to_lodes = {"06": "ca", "48": "tx", "36": "ny", "34": "nj", "09": "ct"}

    rows = []
    for st_fips in states_needed:
        st = state_to_lodes[st_fips]
        path = _require(RAW / "lodes" / f"{st}_od_main_JT00_2021.csv.gz",
                        "--metro " + metro)
        df = pd.read_csv(path, dtype={"w_geocode": str, "h_geocode": str})
        # geocode is 15-char block FIPS — first 11 chars = tract
        df["origin_tract"] = df["h_geocode"].str[:11]
        df["dest_tract"] = df["w_geocode"].str[:11]
        df["origin_county"] = df["h_geocode"].str[:5]
        df["dest_county"] = df["w_geocode"].str[:5]
        df = df[df["origin_county"].isin(counties) & df["dest_county"].isin(counties)]
        df = df.groupby(["origin_tract", "dest_tract"], as_index=False)["S000"].sum()
        df = df.rename(columns={"S000": "jobs"})
        rows.append(df)
    df = pd.concat(rows, ignore_index=True)
    df.to_parquet(out)
    return df


def load_lodes_wac(metro: str = "la", refresh: bool = False) -> pd.DataFrame:
    """LODES Work Area Characteristics — jobs per destination tract by wage tier."""
    out = PROC / f"lodes_wac_{metro}.parquet"
    if out.exists() and not refresh:
        return pd.read_parquet(out)

    counties = _county_filter(metro)
    state_to_lodes = {"06": "ca", "48": "tx", "36": "ny", "34": "nj", "09": "ct"}
    parts = []
    for st_fips in {fips[:2] for fips in counties}:
        st = state_to_lodes[st_fips]
        path = _require(RAW / "lodes" / f"{st}_wac_S000_JT00_2021.csv.gz",
                        "--metro " + metro)
        df = pd.read_csv(path, dtype={"w_geocode": str})
        df["dest_tract"] = df["w_geocode"].str[:11]
        df["dest_county"] = df["w_geocode"].str[:5]
        df = df[df["dest_county"].isin(counties)]
        # Aggregate to tract: total + high-wage (CE03 = $3333+/mo)
        agg = df.groupby("dest_tract", as_index=False).agg(
            jobs_total=("C000", "sum"),
            jobs_high_wage=("CE03", "sum"),
            jobs_mid_wage=("CE02", "sum"),
            jobs_low_wage=("CE01", "sum"),
        )
        parts.append(agg)
    df = pd.concat(parts, ignore_index=True)
    df.to_parquet(out)
    return df


# ----------------------------------------------------------------------------
# 4. CalEnviroScreen 4.0
# ----------------------------------------------------------------------------
def load_ces(refresh: bool = False) -> pd.DataFrame:
    """CalEnviroScreen 4.0 tract-level scores (Pollution Burden × Population Characteristics)."""
    out = PROC / "ces40.parquet"
    if out.exists() and not refresh:
        return pd.read_parquet(out)

    zip_path = _require(RAW / "calenviroscreen" / "ces40_results.zip", "--metro la")
    with zipfile.ZipFile(zip_path) as z:
        # Find any .xlsx or .csv with "results" in the name
        candidates = [n for n in z.namelist()
                      if n.lower().endswith((".xlsx", ".csv"))
                      and "result" in n.lower()]
        if not candidates:
            candidates = [n for n in z.namelist() if n.lower().endswith((".xlsx", ".csv"))]
        member = candidates[0]
        with z.open(member) as f:
            if member.lower().endswith(".xlsx"):
                df = pd.read_excel(f)
            else:
                df = pd.read_csv(f)

    # CES uses 11-char tract IDs as 'Census Tract'
    tract_col = next((c for c in df.columns if "census tract" in c.lower()), None)
    if tract_col is None:
        tract_col = "Census Tract"
    df["tract_geoid"] = df[tract_col].astype(str).str.zfill(11)

    # Standardize key score columns (CES naming varies slightly across releases)
    rename = {}
    for c in df.columns:
        cl = c.lower()
        if cl == "ces 4.0 score" or cl == "ces 4.0 percentile":
            rename[c] = cl.replace(" ", "_").replace(".", "")
        elif "pollution burden" in cl and "score" in cl:
            rename[c] = "pollution_burden_score"
        elif "pollution burden" in cl and "pctl" in cl.replace("percentile", "pctl"):
            rename[c] = "pollution_burden_pctl"
        elif "population characteristics" in cl and "score" in cl:
            rename[c] = "pop_char_score"
        elif "population characteristics" in cl:
            rename[c] = "pop_char_pctl"
        elif cl.startswith("ces") and "score" in cl:
            rename[c] = "ces_score"
        elif cl.startswith("ces") and "percentile" in cl:
            rename[c] = "ces_pctl"
    df = df.rename(columns=rename)

    keep = ["tract_geoid"] + [c for c in [
        "ces_score", "ces_pctl", "pollution_burden_score", "pollution_burden_pctl",
        "pop_char_score", "pop_char_pctl"
    ] if c in df.columns]
    df = df[keep].drop_duplicates("tract_geoid")
    for c in keep[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.to_parquet(out)
    return df


# ----------------------------------------------------------------------------
# 5. Hospitals (HIFLD or CMS)
# ----------------------------------------------------------------------------
def load_hospitals(metro: str = "la", refresh: bool = False) -> gpd.GeoDataFrame:
    """Hospitals as point geometries (vertiport medical-mission anchors)."""
    out = PROC / f"hospitals_{metro}.parquet"
    if out.exists() and not refresh:
        return gpd.read_parquet(out)

    # Per-metro file preferred; legacy fallback for LA
    raw_path = RAW / "hifld" / f"hospitals_{metro}.geojson"
    if not raw_path.exists() or raw_path.stat().st_size < 5000:
        legacy = RAW / "hifld" / "hospitals.geojson"
        if legacy.exists() and legacy.stat().st_size > 5000:
            raw_path = legacy
        else:
            raise FileNotFoundError(
                f"Hospitals data missing for {metro} — run scripts/2_fetch_hospitals_osm.py {metro}")

    counties = _county_filter(metro)
    # Could be GeoJSON or CSV
    if raw_path.read_text(errors="ignore")[:200].lstrip().startswith("{"):
        gdf = gpd.read_file(raw_path)
    else:
        df = pd.read_csv(raw_path)
        # CMS columns: Address, City, State, ZIP code; lat/lon may be in 'Location'
        if "Location" in df.columns:
            df = df.dropna(subset=["Location"])
            df["lon"] = df["Location"].str.extract(r"\(([-0-9.]+)").astype(float)
            df["lat"] = df["Location"].str.extract(r"\(.+, ([-0-9.]+)\)").astype(float)
        elif "X" in df.columns and "Y" in df.columns:
            df["lon"], df["lat"] = df["X"], df["Y"]
        gdf = gpd.GeoDataFrame(
            df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs="EPSG:4326")
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")

    # Crude bbox filter (expand to county-set bounding box)
    tracts = load_tracts(metro)
    minx, miny, maxx, maxy = tracts.total_bounds
    gdf = gdf.cx[minx:maxx, miny:maxy].copy()
    # Spatial join to tract → keep only rows within metro counties
    gdf = gpd.sjoin(gdf, tracts[["tract_geoid", "county_fips", "geometry"]],
                    how="inner", predicate="within")
    gdf = gdf.rename(columns={"NAME": "hospital_name"}).reset_index(drop=True)
    keep = ["hospital_name", "tract_geoid", "county_fips", "geometry"]
    keep = [k for k in keep if k in gdf.columns] + ["geometry"] if "geometry" not in keep else keep
    gdf = gdf[[c for c in keep if c in gdf.columns]].drop_duplicates()
    gdf.to_parquet(out)
    return gdf


# ----------------------------------------------------------------------------
# 6. FAA / OurAirports — heliports + airports for vertiport candidates
# ----------------------------------------------------------------------------
def load_faa_heliports(metro: str = "la", refresh: bool = False) -> gpd.GeoDataFrame:
    """Existing heliports + small airports — vertiport candidate seeds."""
    out = PROC / f"heliports_{metro}.parquet"
    if out.exists() and not refresh:
        return gpd.read_parquet(out)

    csv_path = _require(RAW / "faa" / "faa_airports.csv", "--metro " + metro)
    df = pd.read_csv(csv_path, low_memory=False)

    # OurAirports CSV columns: type, latitude_deg, longitude_deg, iso_country, iso_region, ...
    if "latitude_deg" in df.columns:
        df = df[df["iso_country"] == "US"].copy()
        df["lat"] = pd.to_numeric(df["latitude_deg"], errors="coerce")
        df["lon"] = pd.to_numeric(df["longitude_deg"], errors="coerce")
        df = df.dropna(subset=["lat", "lon"])
        # Keep heliports + small airports (potential vertiport sites)
        df = df[df["type"].isin(["heliport", "small_airport", "seaplane_base"])].copy()
    elif "Latitude" in df.columns:
        df["lat"], df["lon"] = pd.to_numeric(df["Latitude"], errors="coerce"), pd.to_numeric(df["Longitude"], errors="coerce")
        df = df.dropna(subset=["lat", "lon"])
    else:
        raise ValueError(f"Unknown FAA CSV schema; columns: {df.columns.tolist()[:20]}")

    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs="EPSG:4326")

    # Filter to metro counties via spatial join
    tracts = load_tracts(metro)
    minx, miny, maxx, maxy = tracts.total_bounds
    gdf = gdf.cx[minx:maxx, miny:maxy].copy()
    gdf = gpd.sjoin(gdf, tracts[["tract_geoid", "county_fips", "geometry"]],
                    how="inner", predicate="within")
    name_col = "name" if "name" in gdf.columns else ("Name" if "Name" in gdf.columns else None)
    if name_col:
        gdf["facility_name"] = gdf[name_col]
    gdf = gdf[[c for c in ["facility_name", "type", "tract_geoid", "county_fips", "geometry"] if c in gdf.columns]]
    gdf = gdf.reset_index(drop=True)
    gdf.to_parquet(out)
    return gdf


# ----------------------------------------------------------------------------
# 7. GTFS — derive transit hubs (rail stops + high-frequency bus stops)
# ----------------------------------------------------------------------------
def load_transit_stops(metro: str = "la", min_freq_per_hour: float = 4.0,
                       refresh: bool = False) -> gpd.GeoDataFrame:
    """Bus + rail transit stops as point geometries; flag high-frequency hubs."""
    out = PROC / f"transit_stops_{metro}.parquet"
    if out.exists() and not refresh:
        return gpd.read_parquet(out)

    metro_to_feeds = {
        "la":    ["la_metro_bus", "la_metro_rail"],
        "dfw":   ["dart", "trinity_metro"],
        "sfbay": ["bart", "sfmta"],
        "nyc":   ["mta_nyc_subway"],
    }
    parts = []
    for feed in metro_to_feeds[metro]:
        zpath = RAW / "gtfs" / f"{feed}.zip"
        if not zpath.exists():
            print(f"  WARN: {zpath} missing — skipping feed '{feed}'")
            continue
        with zipfile.ZipFile(zpath) as z:
            with z.open("stops.txt") as f:
                stops = pd.read_csv(f, dtype={"stop_id": str})
            try:
                with z.open("stop_times.txt") as f:
                    st = pd.read_csv(f, dtype={"trip_id": str, "stop_id": str},
                                     usecols=["trip_id", "stop_id"])
            except KeyError:
                st = pd.DataFrame({"trip_id": pd.Series(dtype=str),
                                    "stop_id": pd.Series(dtype=str)})
        # Cast both stop_id columns explicitly to avoid pandas merge dtype coercion error
        stops["stop_id"] = stops["stop_id"].astype(str)
        st["stop_id"] = st["stop_id"].astype(str)
        # Approximate weekday daily frequency = arrivals per stop / typical service hours (15)
        freq = (st.groupby("stop_id").size().reset_index(name="n_arrivals_total")
                  .assign(freq_per_hour=lambda x: x["n_arrivals_total"] / 15.0))
        stops = stops.merge(freq, on="stop_id", how="left").fillna({"freq_per_hour": 0.0})
        stops["feed"] = feed
        stops["is_rail"] = "rail" in feed.lower() or "subway" in feed.lower() or feed in {"bart"}
        parts.append(stops[["stop_id", "stop_name", "stop_lat", "stop_lon",
                            "freq_per_hour", "feed", "is_rail"]])

    if not parts:
        raise FileNotFoundError(
            "No GTFS feeds available for metro=%s. Run scripts/1_download_data.py" % metro)
    stops = pd.concat(parts, ignore_index=True)
    stops = stops.dropna(subset=["stop_lat", "stop_lon"])
    gdf = gpd.GeoDataFrame(stops,
        geometry=gpd.points_from_xy(stops["stop_lon"], stops["stop_lat"]),
        crs="EPSG:4326")
    gdf["is_hub"] = gdf["is_rail"] | (gdf["freq_per_hour"] >= min_freq_per_hour)

    tracts = load_tracts(metro)
    gdf = gpd.sjoin(gdf, tracts[["tract_geoid", "county_fips", "geometry"]],
                    how="inner", predicate="within")
    gdf = gdf.drop(columns=["index_right"]).reset_index(drop=True)
    gdf.to_parquet(out)
    return gdf


# ----------------------------------------------------------------------------
# 8. Joined tract dataset (geometry + ACS + CES)
# ----------------------------------------------------------------------------
def load_tracts_joined(metro: str = "la", refresh: bool = False) -> gpd.GeoDataFrame:
    """Tract polygons ⨝ ACS demographics ⨝ CES vulnerability."""
    out = PROC / f"tracts_joined_{metro}.parquet"
    if out.exists() and not refresh:
        return gpd.read_parquet(out)

    tracts = load_tracts(metro)
    acs = load_acs(metro)
    out_gdf = tracts.merge(acs, on="tract_geoid", how="left", suffixes=("", "_acs"))

    # CES is California-only; only attempt for CA metros
    if metro in {"la", "sfbay"}:
        try:
            ces = load_ces()
            out_gdf = out_gdf.merge(ces, on="tract_geoid", how="left")
        except FileNotFoundError as e:
            print(f"  WARN: CES not loaded — {e}")
    out_gdf = out_gdf.dropna(subset=["pop"]).copy()
    out_gdf = out_gdf[out_gdf["pop"] > 0]
    out_gdf.to_parquet(out)
    return out_gdf
