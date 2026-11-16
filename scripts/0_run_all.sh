#!/usr/bin/env bash
# Reproduce the results, tables and figures of the paper from public data.
set -euo pipefail
cd "$(dirname "$0")/.."
METROS="la dfw sfbay nyc"
WORKERS="${WORKERS:-8}"

echo "[1/7] Download public data (about 3 GB, mostly OpenStreetMap extracts and LODES)"
python3 scripts/1_download_data.py --metro all --skip-large
python3 scripts/2_fetch_hospitals_osm.py all
python3 scripts/3_fetch_holc_redlining.py all

echo "[2/7] Street-network travel times between tracts and candidate sites"
for m in $METROS; do python3 -m src.network_od "$m" "$WORKERS"; done

echo "[3/7] Congestion calibration against ACS commute times"
for m in $METROS; do python3 -m src.congestion "$m" "$WORKERS"; done

echo "[4/7] Policy scenarios S0 to S7"
python3 -m src.main --metro all --n-vertiports 25 --max-candidates 50

echo "[5/7] Fare tiers, supplementary experiments, greenfield pool, operations, robustness"
for m in $METROS; do
    for s in 9_revision_diagnostics 10_network_baseline 11_fare_analysis 12_revision_experiments \
             13_greenfield 14_operations 15_spatial_diagnostics 16_robustness 17_subsidy_robustness; do
        python3 "scripts/$s.py" "$m"
    done
done

echo "[6/7] Tables"
python3 scripts/18_paper_tables.py

echo "[7/7] Figures"
python3 scripts/20_study_area_map.py
python3 scripts/21_paper_figures.py
python3 scripts/22_operations_schematic.py
