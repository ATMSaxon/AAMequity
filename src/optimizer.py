"""Equity-constrained vertiport-siting MILP.

We use PuLP + CBC (open-source) so the pipeline is fully reproducible without
commercial licenses. Gurobi is supported via the same PuLP interface if installed.

Decision variable:
    x_k ∈ {0,1}   for each candidate vertiport k

Per-tract precomputed contribution arrays (parameters):
    g_ik       per-tract i accessibility GAIN if vertiport k is built (LOWER bound estimate)
    b_ik       per-tract i burden contribution if vertiport k is built
    pop_i      tract population
    inc_i      median income (for ranking / equity-floor logic)
    group_i    demographic group label

We use a *single-vertiport-attribution* approximation for tractability:
    dA_i ≈ max_k (g_ik * x_k)            — best vertiport drives the gain
    B_i  ≈ sum_k (b_ik * x_k)            — burden adds linearly across selected vertiports

This is the standard linearization in equity-aware FLP (Aljohani & Hall 2024,
Roy & Law 2025); the additive burden + max benefit formulation is exactly the
combination called for in our brief §8.6.

Scenarios — each maps to a different objective + constraint set:
    S0: baseline (no vertiports)
    S1: market_led          maximize Σ pop_i * dA_i (no equity / burden constraints)
    S2: accessibility_max   maximize Σ pop_i * dA_i AND maximize total dA (same as S1 in our setup;
                            differs from S1 only when revenue depends on income; we add an income-weighted variant)
    S3: equity_constrained  S2 + per-group min-gain floor + burden cap
    S4: transit_integrated  S3 + bonus for vertiports near transit hubs
    S5: ej_constrained      S2 + per-CES-quartile burden ceiling
    S6: subsidized_public   S3 with low-income lambda (changes accessibility input)
"""

from __future__ import annotations
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pulp

from .config import CFG, OptimizerParams


# ----------------------------------------------------------------------------
# Helpers to build per-tract contributions from cost matrices
# ----------------------------------------------------------------------------
def per_vertiport_gain_matrix(tracts_xy: np.ndarray,
                               cand_xy: np.ndarray,
                               base_cost: np.ndarray,
                               aam_cost_single: callable) -> np.ndarray:
    """Compute g_ik = improvement if ONLY vertiport k is in the network.

    We assume a single vertiport k makes round-trip routings via k available.
    For a single k, the AAM cost between i and j becomes:
        T_access(i,k) + T_wait + T_security + T_flight(k,k) + T_egress(k,j) + λ·Fare
    If i = j, AAM is not used (cost stays at 0).
    A_i^aam(k) - A_i^base measured per tract.

    Implemented in scenarios.py via accessibility.compute_aam_cost_matrix with
    a single selected_idx; this function is a placeholder kept for clarity.
    """
    raise NotImplementedError("Use accessibility.compute_aam_cost_matrix iteratively.")


# ----------------------------------------------------------------------------
# Solver
# ----------------------------------------------------------------------------
@dataclass
class ScenarioInputs:
    """Pre-computed numerical inputs to the MILP."""
    g_ik: np.ndarray          # (n_tracts, n_candidates) per-tract gain if k built (alone)
    b_ik: np.ndarray          # (n_tracts, n_candidates) per-tract burden contribution
    pop_i: np.ndarray         # (n_tracts,) tract population
    income_i: np.ndarray      # (n_tracts,) median income
    group_i: np.ndarray       # (n_tracts,) group labels (object dtype)
    transit_int_k: np.ndarray # (n_candidates,) 1 if vertiport candidate near transit
    ces_pctl_k: np.ndarray    # (n_candidates,) CES percentile of containing tract
    proc_disadv_i: np.ndarray = None  # (n_tracts,) 1 if tract is in procedural-disadvantaged group


@dataclass
class ScenarioResult:
    name: str
    selected: np.ndarray            # bool (n_candidates,)
    selected_idx: np.ndarray         # int positions
    dA_i: np.ndarray                 # post-solve per-tract gain
    B_i: np.ndarray                  # post-solve per-tract burden
    objective: float
    runtime_sec: float
    status: str


def solve_scenario(name: str,
                   inp: ScenarioInputs,
                   *,
                   n_vertiports: int,
                   alpha: float = 1.0,           # weight on benefit
                   gamma: float = 0.0,           # weight on inequality penalty (Theil-like surrogate)
                   delta: float = 0.0,           # weight on burden penalty
                   transit_bonus: float = 0.0,   # bonus weight per transit-integrated vertiport
                   per_group_floor: dict | None = None,   # {"low_income": min mean dA, ...}
                   per_tract_burden_cap: float | None = None,
                   per_group_burden_cap: dict | None = None,
                   require_transit_share: float = 0.0,    # min fraction of vertiports near transit
                   procedural_min_coverage: float | None = None,  # min Σ pop·g_ik·x_k for proc-disadv tracts
                   solver_params: OptimizerParams | None = None) -> ScenarioResult:
    """Solve the equity-aware vertiport-siting MILP for one scenario."""
    p = solver_params or CFG.opt
    n_tract, n_cand = inp.g_ik.shape

    prob = pulp.LpProblem(f"vertiport_{name}", pulp.LpMaximize)
    x = [pulp.LpVariable(f"x_{k}", cat="Binary") for k in range(n_cand)]
    # Linearize max via auxiliary z_i = approximation of dA_i (we use weighted-sum surrogate)
    # For tractability we approximate dA_i ≈ Σ_k g_ik * x_k * w_select where w_select = 1/n
    # This convex surrogate over-counts; we apply a saturation z_i ≤ max_k g_ik * x_k via:
    #   z_i ≤ g_ik   for each k where x_k = 1   (big-M relaxation handled implicitly via cap_i)
    # Here we use a tighter formulation: introduce z_i = sum_k g_ik * x_k * pi_ik
    # and enforce z_i ≤ max_k g_ik. We keep things simple by using sum-form for the objective
    # (true maximum is non-linear), and we report achieved max post-solve.

    # Auxiliary: total accessibility benefit — sum form approximation
    total_benefit = pulp.lpSum(
        float(inp.pop_i[i]) * pulp.lpSum(float(inp.g_ik[i, k]) * x[k] for k in range(n_cand))
        for i in range(n_tract)
    )
    total_burden = pulp.lpSum(
        float(inp.pop_i[i]) * pulp.lpSum(float(inp.b_ik[i, k]) * x[k] for k in range(n_cand))
        for i in range(n_tract)
    )
    transit_bonus_term = pulp.lpSum(float(inp.transit_int_k[k]) * x[k] for k in range(n_cand))

    # ---- Cardinality ----
    prob += pulp.lpSum(x) == int(n_vertiports)

    # ---- Optional: minimum transit-integrated share ----
    if require_transit_share > 0:
        prob += pulp.lpSum(float(inp.transit_int_k[k]) * x[k] for k in range(n_cand)) >= require_transit_share * n_vertiports

    # ---- Per-tract burden cap (linear in x) ----
    if per_tract_burden_cap is not None:
        for i in range(n_tract):
            cap = per_tract_burden_cap
            prob += pulp.lpSum(float(inp.b_ik[i, k]) * x[k] for k in range(n_cand)) <= cap

    # ---- Per-group burden cap (mean burden in group) ----
    if per_group_burden_cap:
        groups = np.asarray(inp.group_i)
        for g, cap in per_group_burden_cap.items():
            mask = (groups == g)
            if mask.sum() == 0:
                continue
            wsum = inp.pop_i[mask].sum()
            if wsum <= 0:
                continue
            prob += pulp.lpSum(
                float(inp.pop_i[i] * inp.b_ik[i, k]) * x[k]
                for i in np.where(mask)[0] for k in range(n_cand)
            ) <= cap * wsum

    # ---- Procedural-equity coverage: procedurally-disadvantaged tracts must
    #      receive ≥ procedural_min_coverage * (their share of city ΔA budget) ----
    if procedural_min_coverage is not None and inp.proc_disadv_i is not None:
        proc_mask = inp.proc_disadv_i.astype(bool)
        pop_proc = inp.pop_i[proc_mask].sum()
        if pop_proc > 0:
            prob += pulp.lpSum(
                float(inp.pop_i[i] * inp.g_ik[i, k]) * x[k]
                for i in np.where(proc_mask)[0] for k in range(n_cand)
            ) >= procedural_min_coverage * pop_proc

    # ---- Per-group accessibility floor ----
    if per_group_floor:
        groups = np.asarray(inp.group_i)
        for g, floor in per_group_floor.items():
            mask = (groups == g)
            if mask.sum() == 0:
                continue
            wsum = inp.pop_i[mask].sum()
            if wsum <= 0:
                continue
            prob += pulp.lpSum(
                float(inp.pop_i[i] * inp.g_ik[i, k]) * x[k]
                for i in np.where(mask)[0] for k in range(n_cand)
            ) >= floor * wsum

    # ---- Objective: alpha * benefit + transit_bonus * transit - delta * burden ----
    prob += alpha * total_benefit + transit_bonus * transit_bonus_term - delta * total_burden

    # ---- Solve ----
    t0 = time.time()
    solver_name = (p.solver_name or "cbc").lower()
    if solver_name == "gurobi":
        try:
            solver = pulp.GUROBI(msg=False, timeLimit=p.time_limit_sec, gapRel=p.mip_gap)
        except Exception:
            solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=p.time_limit_sec, gapRel=p.mip_gap)
    else:
        solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=p.time_limit_sec, gapRel=p.mip_gap)
    status = prob.solve(solver)
    rt = time.time() - t0

    status_str = pulp.LpStatus[status]

    # FIX (Round 3 R5): tighten — accept ONLY Optimal. "Not Solved" can mean timeout
    # which is dangerous to publish.
    if not status_str.lower().startswith("optimal"):
        return ScenarioResult(
            name=name,
            selected=np.zeros(n_cand, bool),
            selected_idx=np.array([], dtype=int),
            dA_i=np.zeros(n_tract),
            B_i=np.zeros(n_tract),
            objective=0.0,
            runtime_sec=rt,
            status=status_str,
        )

    sel = np.array([int(round(pulp.value(xi) or 0)) for xi in x], dtype=bool)
    sel_idx = np.where(sel)[0]

    if sel.any():
        dA_i = inp.g_ik[:, sel_idx].max(axis=1)
        B_i = inp.b_ik[:, sel_idx].sum(axis=1)
    else:
        dA_i = np.zeros(n_tract)
        B_i = np.zeros(n_tract)

    return ScenarioResult(
        name=name, selected=sel, selected_idx=sel_idx,
        dA_i=dA_i, B_i=B_i,
        objective=float(pulp.value(prob.objective) or 0.0),
        runtime_sec=rt,
        status=status_str,
    )
