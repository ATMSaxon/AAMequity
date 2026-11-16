"""Spatial autocorrelation and HOLC-grade profiles of every scenario at each fare tier.

Usage:  python scripts/15_spatial_diagnostics.py <metro>
Output: results/<metro>/<tier>/spatial_autocorr.json, acoustic_redlining.json
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import CFG
from src.redlining import acoustic_redlining_test
from src.spatial_stats import spatial_autocorr_per_scenario


def main():
    metro = sys.argv[1]
    for tier in CFG.aam.fare_per_km:
        spatial_autocorr_per_scenario(metro, fare_tier=tier)
        acoustic_redlining_test(metro, fare_tier=tier)
        print(f"[{metro}] {tier}: spatial diagnostics written", flush=True)
    print(f"{metro}: done", flush=True)


if __name__ == "__main__":
    main()
