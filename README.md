# AAMequity

Code and data for the paper "Who Benefits and Who Bears the Burden? Equity-Aware Vertiport Siting and Policy Design for Advanced Air Mobility", submitted to Transportation Research Part A: Policy and Practice.

The study selects 25 vertiports for each of four US metropolitan areas (Los Angeles County, Dallas-Fort Worth, the San Francisco Bay Area and the New York metropolitan area) with a mixed-integer siting model. It measures how the job accessibility gains and the local burdens of each network are distributed across census tracts by income, minority share and car ownership. Eight policy scenarios (S0 to S7) are evaluated at four fare tiers. Ground travel times come from the OpenStreetMap street network, with congestion calibrated to ACS commute times.

## Installation

The code runs on Python 3.10 or later.

```bash
python3 -m pip install -r requirements.txt
```

## Reproducing the paper

```bash
bash scripts/0_run_all.sh
```

The script downloads the data that the repository does not include, computes the street-network travel times, calibrates congestion, runs the scenarios and the supplementary analyses, and writes the tables to `results/revision/tables/` and the figures to `figures/`. The network and congestion steps run in parallel; set `WORKERS` to the number of processes to use (the default is 8).

Each analysis script also runs for a single metro, for example

```bash
python3 -m src.network_od la 8
python3 -m src.congestion la 8
python3 scripts/11_fare_analysis.py la
```

## Repository layout

| Path | Contents |
|---|---|
| `src/` | Model code: data loading, candidate sites, accessibility, Burden Index, siting model, scenarios, equity metrics |
| `scripts/` | Data download, analyses, tables and figures, numbered in the order they run |
| `data/raw/` | Raw inputs small enough to include (ACS, CalEnviroScreen, FAA, HIFLD hospitals, HOLC grades) |
| `data/processed/` | Tract and candidate-site tables built from the raw inputs |
| `results/<metro>/` | Scenario outcomes per fare tier, congestion calibration |
| `results/revision/` | Results behind the tables and figures of the paper, and the LaTeX tables |
| `tests/` | Smoke test on a subset of Los Angeles County |

### Model code

| Module | Role |
|---|---|
| `config.py` | Metro definitions, paths and model parameters |
| `data_layer.py` | Tract-level ACS, LODES, TIGER, GTFS, CalEnviroScreen, FAA and hospital data |
| `candidates.py` | Existing-facility and greenfield candidate pools |
| `network_od.py` | Street-network travel times between tracts and candidate sites |
| `congestion.py` | Congestion multipliers and terminal time calibrated to ACS commute times |
| `accessibility.py` | Gravity accessibility with ground and AAM travel costs |
| `burden.py` | Noise, visual intrusion, land-use conflict, ground traffic and safety components of the Burden Index |
| `optimizer.py` | Surrogate MILP for site selection (PuLP with CBC) |
| `scenarios.py` | Scenarios S0 to S7 and post-solve evaluation |
| `equity_metrics.py` | Concentration index, ratio to the metropolitan mean, bootstrap intervals |
| `spatial_stats.py` | Global Moran's I |
| `redlining.py` | Burden by 1939 HOLC grade |

### Scripts

| Script | Output |
|---|---|
| `1_download_data.py`, `2_fetch_hospitals_osm.py`, `3_fetch_holc_redlining.py` | `data/raw/` |
| `9_revision_diagnostics.py` | Calibration and scenario diagnostics |
| `10_network_baseline.py` | Scenario outcomes under network ground times |
| `11_fare_analysis.py` | Fare sweep and scenarios at the four fare tiers |
| `12_revision_experiments.py` | Bootstrap intervals, demographic groups, HOLC benchmarks, noise bands, network size |
| `13_greenfield.py` | Greenfield candidate pool |
| `14_operations.py` | Discrete-event simulation of vertiport operations |
| `15_spatial_diagnostics.py` | Moran's I and HOLC profiles of every scenario |
| `16_robustness.py` | Robustness to the calibration, model parameters and component weights |
| `17_subsidy_robustness.py` | Robustness of the fare-subsidy results |
| `18_paper_tables.py` | LaTeX tables |
| `20_study_area_map.py`, `21_paper_figures.py`, `22_operations_schematic.py` | Figures |

## Data

All inputs are public. The repository includes the smaller raw files and the processed tract tables. `scripts/1_download_data.py` downloads the remaining files from the original sources.

| Dataset | Source | Included |
|---|---|---|
| American Community Survey 2018-2022, tract demographics and commute tables | US Census Bureau | Yes |
| LODES 8, 2021 workplace and origin-destination files | US Census Bureau, LEHD | Processed workplace tables |
| TIGER/Line 2024 tract boundaries | US Census Bureau | Processed tract tables |
| GTFS feeds of LA Metro, BART, DART and MTA New York City Transit | Transit agencies | Processed stop tables |
| OpenStreetMap state extracts | Geofabrik | No |
| Airports and heliports | FAA | Yes |
| Hospitals | HIFLD and OpenStreetMap | Yes |
| CalEnviroScreen 4.0 | California OEHHA | Yes |
| HOLC area descriptions, 1935-1940 | Mapping Inequality, University of Richmond | Yes |

The HOLC polygons are distributed by Mapping Inequality under the CC BY-NC-SA 4.0 license, and the same license applies to `data/raw/holc/`. OpenStreetMap data is available under the Open Database License. The Geofabrik extracts are updated daily, so a fresh download gives a slightly different street network from the October 2026 extracts used in the paper.

## License

The code is released under the MIT License. The data files keep the terms of their original sources.
