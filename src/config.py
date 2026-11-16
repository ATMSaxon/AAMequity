"""Centralized configuration — paths, parameters, scenario constants.

All numeric AAM service parameters trace to the literature; see source citations.
Edit `aam_params` to override default scenario values.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
PROC = DATA / "processed"
CACHE = DATA / "cache"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
for _d in (RAW, PROC, CACHE, RESULTS, FIGURES):
    _d.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# LA County FIPS for filtering CA datasets
# ----------------------------------------------------------------------------
LA_COUNTY_FIPS = "06037"  # state 06 + county 037
CA_STATE_FIPS = "06"

# Robustness metros (county FIPS for filtering tract sets)
METRO_COUNTY_FIPS = {
    "la":    ["06037"],                                                    # LA County
    "dfw":   ["48113", "48439", "48121", "48085", "48251"],                # Dallas, Tarrant, Denton, Collin, Johnson
    "sfbay": ["06001", "06013", "06041", "06055", "06075", "06081",        # 9-county SF Bay
              "06085", "06095", "06097"],
    "nyc":   ["36005", "36047", "36061", "36081", "36085",                 # 5 NYC boroughs
              "34003", "34013", "34017", "34031",                          # 4 NJ counties (closest)
              "36059", "36103", "36119"],                                  # Long Island + Westchester
}

# ----------------------------------------------------------------------------
# AAM service parameters (Pascioni 2022, Rizzi 2024, Boddupalli 2024 calibrated)
# ----------------------------------------------------------------------------
@dataclass
class AAMParams:
    # Cruise speed (km/h). Joby S4: 320 km/h cruise; Archer Midnight: 240 km/h.
    cruise_speed_kmh: float = 280.0

    # Operational time components (minutes)
    t_security_min: float = 8.0       # passenger check-in + buffer (Boddupalli scenario midpoint)
    t_takeoff_min: float = 1.5        # vertical climb-out
    t_landing_min: float = 1.5        # vertical approach
    t_taxi_pad_min: float = 1.0       # ground roll on pad

    # Wait time scales as a function of frequency (per vertiport pad). Default: 12 ops/h.
    # Vascik & Hansman (2020) NASA capacity analyses suggest mature operations 8-15 ops/h/pad.
    ops_per_pad_per_hour: float = 12.0
    pads_per_vertiport: int = 2       # multi-pad default
    # Mean wait = 60 / (2 * ops_per_pad * n_pads) min (Poisson approximation), unless a
    # passenger wait is supplied, e.g. from the vertiport operations simulation.
    passenger_wait_min: float | None = None

    # Fares ($/trip) — three tiers, anchored to Cohen-Shaheen (2024) cost-down trajectory.
    # Launch (~2025): $250 premium; mid-term scaling: $80-$100; mature subsidized: $30-$50.
    # We model the MID-TERM operational scenario (Cohen-Shaheen mature pricing).
    fare_high: float = 100.0          # mature commercial pricing (Joby/Archer post-scaling)
    fare_mid:  float =  60.0          # standard service tier
    fare_low:  float =  30.0          # subsidized public-service S6 (transit-fare-equivalent)

    # Generalized-cost fare-to-time weight λ (min/$) = 1 / VOT_per_min.
    # Convention: GC_min = T + λ · Fare (Boddupalli 2024 RPL VOT distribution; lower λ = higher VOT).
    # VOT_low  = $15/hr = $0.25/min   →  λ = 4.0 min/$  (transit-dependent group)
    # VOT_mid  = $25/hr = $0.42/min   →  λ = 2.4 min/$  (median commuter)
    # VOT_high = $64/hr = $1.07/min   →  λ = 0.94 min/$ (top decile; Boddupalli sample)
    lambda_low_income:  float = 4.00
    lambda_mid_income:  float = 2.40
    lambda_high_income: float = 0.94

    # Distance-based fares ($ per passenger-km flown). Launch, near-term and long-run are
    # the passenger prices per VTOL mile of the Uber Elevate (2016, p. 95) vehicle
    # economics: $2.97 initially, $0.98 near-term and $0.47 long-term. The public-service
    # tier is a policy fare below any published commercial price point.
    fare_per_km: dict = field(default_factory=lambda: {
        "launch": 2.97 / 1.609344, "near_term": 0.98 / 1.609344, "long_run": 0.47 / 1.609344,
        "public_service": 0.10})
    fare_base: float = 0.0

    # Value of travel time: share of hourly household income (annual income / 2080 h),
    # the USDOT (2016) guidance for personal local travel.
    vot_income_share: float = 0.5
    vot_hours_per_year: float = 2080.0
    vot_income_bins: int = 10         # tracts share the VOT of their income decile

    # Means-tested subsidy: share of the fare paid for residents of low-income tracts
    subsidy_share: float = 0.5

    # AAM in-cabin VOT discount (people enjoy AAM more than ground transit)
    aam_vot_discount: float = 1.0     # multiplier on T_flight; 1.0 = neutral; 0.7 = preferred

    # Maximum acceptable trip distance (km) — beyond which AAM is not worth boarding
    max_trip_km: float = 100.0
    min_trip_km: float = 12.0         # below this, ground transport beats AAM at any plausible fare


@dataclass
class BurdenParams:
    # Noise. Take-off and landing of the Joby pre-production aircraft measured below
    # 65 dBA (maximum A-weighted level) at 100 m (Pascioni et al. 2022). The level of an
    # event falls by 20 log10(d) with distance plus atmospheric absorption (ISO 9613-2,
    # 1 kHz at 10 C and 70% relative humidity). Its effective duration, 10 s at 100 m,
    # grows in proportion to distance up to the length of the take-off or landing phase,
    # and the sound exposure level is SEL = L_Amax + 10 log10(duration).
    la_max_takeoff_100m_dba: float = 65.0
    event_duration_100m_s: float = 10.0
    event_duration_max_s: float = 90.0
    atmospheric_absorption_db_per_km: float = 3.7
    night_ops_share: float = 0.0             # operations between 22:00 and 07:00 (10 dB penalty)
    noise_near_field_m: float = 50.0         # closest receptor distance evaluated

    # Burden Index component weights (sensitivity-tested)
    w_noise: float = 0.40
    w_visual: float = 0.10
    w_traffic: float = 0.15
    w_landuse: float = 0.20
    w_safety: float = 0.15


@dataclass
class OptimizerParams:
    # Budget: number of vertiports to site (cardinality budget)
    n_vertiports_budget: int = 25

    # Per-tract burden cap (S5): share of the highest tract burden under the
    # accessibility-maximising network of the same size
    burden_cap_share: float = 0.75

    # Group floors are expressed relative to the best gain any network of the same size
    # can deliver to the group, so a floor below 1 is always feasible on its own.
    equity_floor_share: float = 0.90        # S3: share of the best attainable low-income gain
    procedural_coverage_share: float = 0.90 # S7: share of the best attainable gain of procedurally disadvantaged tracts

    # Multi-objective weights
    alpha_benefit: float = 1.0     # population-weighted ΔA
    beta_cost: float = 0.0         # cost penalty (assume uniform unit cost in main runs)
    gamma_inequality: float = 0.0  # inequality penalty (default off; switch on for S3 variants)
    delta_burden: float = 0.0      # burden penalty (default off; switch on for S5)

    # Solver
    solver_name: str = "cbc"       # "cbc" (open-source via PuLP) or "gurobi"
    time_limit_sec: int = 600
    mip_gap: float = 0.02


@dataclass
class Config:
    aam: AAMParams = field(default_factory=AAMParams)
    burden: BurdenParams = field(default_factory=BurdenParams)
    opt: OptimizerParams = field(default_factory=OptimizerParams)

    metro: str = "la"
    seed: int = 0

    # Perturbations of the calibrated ground times, for robustness analysis
    congestion_multiplier_scale: float = 1.0   # scales every (kappa - 1)
    terminal_time_shift_min: float = 0.0

    # Ground travel times: "network" = OSM street network with congestion calibrated
    # to ACS commute times (src/network_od.py, src/congestion.py);
    # "euclidean" = straight-line distance at a constant speed.
    ground_time_source: str = "network"


CFG = Config()
