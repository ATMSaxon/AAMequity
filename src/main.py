"""Entry point — runs the full AAMequity pipeline for one metro.

Usage:
    python -m src.main --metro la                      # primary case
    python -m src.main --metro la --sensitivity         # also run sensitivity sweeps
    python -m src.main --metro la --n-vertiports 25 --max-candidates 80
    python -m src.main --metro all                      # LA + 3 robustness metros
"""

from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

# Allow running both as `python -m src.main` and `python src/main.py`
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.config import CFG, RESULTS
    from src.scenarios import run_all_scenarios
    from src.sensitivity import run_all_sweeps
else:
    from .config import CFG, RESULTS
    from .scenarios import run_all_scenarios
    from .sensitivity import run_all_sweeps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metro", choices=["la", "dfw", "sfbay", "nyc", "all"], default="la")
    ap.add_argument("--n-vertiports", type=int, default=25)
    ap.add_argument("--max-candidates", type=int, default=80)
    ap.add_argument("--fare-tier", choices=list(CFG.aam.fare_per_km), default="long_run")
    ap.add_argument("--sensitivity", action="store_true",
                    help="Run sensitivity sweeps (fare/wait/capacity/noise/equity)")
    args = ap.parse_args()

    metros = ["la", "dfw", "sfbay", "nyc"] if args.metro == "all" else [args.metro]

    summaries = {}
    for m in metros:
        print(f"\n========== METRO: {m} ==========")
        t0 = time.time()
        summary = run_all_scenarios(
            metro=m,
            n_vertiports=args.n_vertiports,
            max_candidates=args.max_candidates,
            fare_tier=args.fare_tier,
        )
        print(summary.to_string())
        summaries[m] = summary
        print(f"  elapsed: {time.time()-t0:.1f}s")

    if args.sensitivity:
        for m in metros:
            print(f"\n========== SENSITIVITY: {m} ==========")
            run_all_sweeps(metro=m, n_vertiports=args.n_vertiports,
                            max_candidates=min(args.max_candidates, 60))

    # Cross-metro comparison
    if len(summaries) > 1:
        all_df = []
        for m, s in summaries.items():
            s2 = s.copy()
            s2.insert(0, "metro", m)
            all_df.append(s2)
        joined = __import__("pandas").concat(all_df, ignore_index=True)
        joined.to_csv(RESULTS / "cross_metro_summary.csv", index=False)
        print(f"\nCross-metro summary -> {RESULTS / 'cross_metro_summary.csv'}")


if __name__ == "__main__":
    main()
