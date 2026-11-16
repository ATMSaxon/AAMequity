"""AAMequity — Equity-aware vertiport siting & policy evaluation framework.

Public API surface (import directly):
  from aam.config import CFG
  from aam.data_layer import load_la_tracts, load_acs, load_lodes_od, load_gtfs, load_ces
  from aam.candidates import build_vertiport_candidates
  from aam.accessibility import compute_base_accessibility, compute_aam_accessibility
  from aam.burden import compute_burden_index
  from aam.equity_metrics import gini, theil, palma, concentration_index, edr, min_gain, ebr
  from aam.optimizer import solve_scenario
  from aam.scenarios import SCENARIOS, run_all_scenarios
  from aam.sensitivity import run_sensitivity_sweep
"""
