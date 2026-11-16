"""LaTeX tables for the paper, generated from results/revision/*.json.

Usage:  python scripts/18_paper_tables.py
Output: results/revision/tables/<name>.tex (one table environment per file)
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REV = ROOT / "results" / "revision"
OUT = REV / "tables"
METROS = ["la", "dfw", "sfbay", "nyc"]
NAME = {"la": "LA", "dfw": "DFW", "sfbay": "SF Bay", "nyc": "NYC"}
TIER_NAME = {"launch": "Launch", "near_term": "Near-term", "long_run": "Long-run",
             "public_service": "Public-service"}
GAIN_THRESHOLD_PCT = 0.1     # gain, as % of base accessibility, defining the break-even fare


def load(kind: str, metro: str) -> dict:
    return json.loads((REV / f"{kind}_{metro}.json").read_text())


def calibration(metro: str) -> dict:
    return json.loads((ROOT / "results" / metro / "congestion_calibration.json").read_text())


def sgn(x: float, nd: int = 2) -> str:
    return f"${x:+.{nd}f}$"


def interval(d: dict, nd: int = 2) -> str:
    return f"${d['estimate']:+.{nd}f}$ [${d['low']:+.{nd}f}$, ${d['high']:+.{nd}f}$]"


def pct(x: float) -> str:
    """Percentage with enough digits to show small values."""
    return f"{x:.2f}" if x >= 0.1 else (f"{x:.3f}" if x >= 0.001 else "0.000")


def table(name: str, caption: str, spec: str, header: str, rows: list[str], size: str = "\\small",
          fit: bool = False) -> None:
    """Write one table environment; `fit` scales the tabular to the text width."""
    body = " \\\\\n".join(rows) + " \\\\"
    tabular = (f"\\begin{{tabular}}{{{spec}}}\n\\toprule\n{header} \\\\\n\\midrule\n{body}\n\\bottomrule\n"
               f"\\end{{tabular}}")
    if fit:
        tabular = f"\\resizebox{{\\linewidth}}{{!}}{{%\n{tabular}}}"
    text = (f"\\begin{{table}}[!htbp]\n\\centering\n\\caption{{{caption}}}\n\\label{{tab:{name}}}\n{size}\n"
            f"{tabular}\n\\end{{table}}\n")
    (OUT / f"{name}.tex").write_text(text)


def break_even(rows: list[dict], key: str) -> float | None:
    """Fare per km at which `key` falls to GAIN_THRESHOLD_PCT, by linear interpolation."""
    rows = sorted(rows, key=lambda r: r["fare_per_km"])
    if rows[0][key] < GAIN_THRESHOLD_PCT:
        return None
    for lo, hi in zip(rows, rows[1:]):
        if lo[key] >= GAIN_THRESHOLD_PCT > hi[key]:
            f = (lo[key] - GAIN_THRESHOLD_PCT) / (lo[key] - hi[key])
            return lo["fare_per_km"] + f * (hi["fare_per_km"] - lo["fare_per_km"])
    return rows[-1]["fare_per_km"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fare = {m: load("fare", m) for m in METROS}
    exp = {m: load("experiments", m) for m in METROS}
    rob = {m: load("robustness", m) for m in METROS}
    green = {m: load("greenfield", m) for m in METROS}
    ops = {m: load("operations", m) for m in METROS}
    sub = {m: load("subsidy", m) for m in METROS}
    cal = {m: calibration(m) for m in METROS}

    # ---- calibration ------------------------------------------------------------------
    rows = []
    for m in METROS:
        c, folds = cal[m], rob[m]["calibration_cross_validation"]["folds"]
        k = c["kappa"]
        r = [f["pearson_r"] for f in folds]
        mae = sum(f["mean_abs_error_min"] for f in folds) / len(folds)
        rows.append(f"{NAME[m]} & {c['tau_min']:.1f} & {k[4]:.2f} & {k[5]:.2f} & {k[6]:.2f} & {k[7]:.2f} & "
                    f"{c['observed_mean_min']:.1f} & {c['predicted_mean_min']:.1f} & "
                    f"{min(r):.2f}--{max(r):.2f} & {mae:.1f}")
    table("calibration",
          "Calibrated door-to-door driving times. $\\tau$ is the terminal time; $\\kappa$ are travel-time "
          "multipliers relative to posted speeds for surface streets (s) and freeways (f) in areas of 5{,}000--20{,}000 "
          "and above 20{,}000 residents and jobs per km$^2$. Mean commute times are for car commuters (ACS). "
          "Out-of-sample correlation and mean absolute error of tract mean commute times are from five-fold "
          "cross-validation. Where the densest class carries less than 2\\% of commute time it shares the "
          "multipliers of the 5{,}000--20{,}000 class.",
          "lrrrrrrrrr",
          "Metro & $\\tau$ (min) & $\\kappa^{\\mathrm{s}}_{5\\text{--}20\\mathrm{k}}$ & $\\kappa^{\\mathrm{f}}_{5\\text{--}20\\mathrm{k}}$ & "
          "$\\kappa^{\\mathrm{s}}_{>20\\mathrm{k}}$ & $\\kappa^{\\mathrm{f}}_{>20\\mathrm{k}}$ & Observed & Predicted & $r$ (held out) & MAE (min)",
          rows, size="\\footnotesize")

    # ---- fare thresholds --------------------------------------------------------------
    rows = []
    for m in METROS:
        sweep = fare[m]["fare_sweep"]["rows"]
        free = next(r for r in sweep if r["fare_per_km"] == 0.0)
        cells = []
        for key in ("mean_dA_pct_of_base", "dA_pct_low_income", "dA_pct_mid_income", "dA_pct_high_income"):
            b = break_even(sweep, key)
            cells.append("---" if b is None else f"{b:.2f}")
        rows.append(f"{NAME[m]} & {free['mean_dA_pct_of_base']:.2f} & {sgn(free['ci_dA_vs_income'])} & " + " & ".join(cells))
    table("fare_threshold",
          "Accessibility gain without a fare, and the fare below which the gain of the full candidate network "
          f"exceeds {GAIN_THRESHOLD_PCT}\\% of baseline accessibility, for all residents and by income tercile. "
          "Gains are population-weighted and expressed relative to baseline job accessibility.",
          "lrrrrrr",
          "Metro & Gain at zero fare (\\%) & CI at zero fare & \\multicolumn{4}{c}{Break-even fare (\\$/km)} \\\\\n"
          " & & & All & Low income & Middle income & High income",
          rows)

    # ---- direction of the income concentration -------------------------------------------
    rows = []
    for m in METROS:
        s = rob[m]["sign_uniform_value_of_time"]["public_service"]
        opt, rnd, grn = s["optimised"], s["random_networks_ci"], s["greenfield_random_networks_ci"]
        inc = exp[m]["tiers"]["public_service"]["uncertainty"]["ci_dA_vs_income"]
        rows.append(f"{NAME[m]} & {interval(opt['interval'])} & {sgn(rnd['mean'])} [{sgn(rnd['p2.5'])}, {sgn(rnd['p97.5'])}] & "
                    f"{sgn(grn['mean'])} [{sgn(grn['p2.5'])}, {sgn(grn['p97.5'])}] & {interval(inc)}")
    table("direction",
          "Income concentration index of accessibility gains at the public-service fare. Negative values favour "
          "lower-income tracts. Under a uniform value of time: the accessibility-maximising network with a 95\\% "
          "tract-bootstrap interval, and the mean and 95\\% range over random 25-site networks drawn from the "
          "existing-facility pool and from the greenfield pool. Last column: the accessibility-maximising network "
          "when the value of time follows tract income.",
          "lllll",
          "Metro & Optimised network & Random, existing facilities & Random, greenfield & Income-specific value of time",
          rows, size="\\footnotesize")

    # ---- scenarios at the public-service fare ---------------------------------------------
    by = {m: {r["scenario"]: r for r in fare[m]["scenarios"]["public_service"]} for m in METROS}
    rows = []
    for scen in ("S1", "S2", "S3", "S4", "S5", "S6", "S7"):
        cells = []
        for m in METROS:
            r = by[m][scen]
            if r["n_vertiports"] == 0:
                cells.append("\\multicolumn{3}{c}{infeasible}")
            elif "not binding" in r["status"]:
                cells.append("\\multicolumn{3}{c}{same as S2}")
            else:
                cells.append(f"{pct(r['mean_dA_pct_of_base'])} & {sgn(r['ci_dA_vs_income'])} & {sgn(r['ci_B_vs_income'])}")
        rows.append(f"{scen} & " + " & ".join(cells))
    table("scenarios",
          "Scenario outcomes at the public-service fare with income-specific value of time. Gain: population-weighted "
          "accessibility gain (\\% of baseline); CI$_{\\Delta A}$ and CI$_{B}$: income concentration index of the gain "
          "and of the burden. `Same as S2' marks a constraint that the accessibility-maximising network already satisfies.",
          "l" + "rrr" * 4,
          "Scenario & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{NAME[m]}}}" for m in METROS) + " \\\\\n"
          + " ".join(f"\\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(4)) + "\n"
          " & " + " & ".join(["Gain & CI$_{\\Delta A}$ & CI$_{B}$"] * 4),
          rows, size="\\footnotesize\\setlength{\\tabcolsep}{3.5pt}")

    # ---- subsidy ----------------------------------------------------------------------------
    rows = []
    for m in METROS:
        r = sub[m]["tiers"]["public_service"]["0.5"]
        c = r["change_in_ci_interval"]
        rows.append(
            f"{NAME[m]} & {pct(r['none']['mean_dA_pct_of_base'])} & {pct(r['none']['low_income_dA_pct_of_base'])} & {sgn(r['none']['ci_dA_vs_income'])} & "
            f"{pct(r['means_tested']['mean_dA_pct_of_base'])} & {pct(r['means_tested']['low_income_dA_pct_of_base'])} & {sgn(r['means_tested']['ci_dA_vs_income'])} & "
            f"{pct(r['universal']['mean_dA_pct_of_base'])} & {pct(r['universal']['low_income_dA_pct_of_base'])} & {sgn(r['universal']['ci_dA_vs_income'])} & "
            f"{interval(c['means_tested'])}")
    table("subsidy",
          "Fare subsidy at the public-service fare, full candidate network. A means-tested subsidy halves the fare "
          "for residents of low-income tracts; a universal cut halves it for everyone. All: gain of all residents; "
          "Low: gain of the low-income tercile (\\% of baseline accessibility). Last column: change in the "
          "concentration index under the means-tested subsidy with a 95\\% tract-bootstrap interval.",
          "lrrrrrrrrrl",
          "Metro & \\multicolumn{3}{c}{No subsidy} & \\multicolumn{3}{c}{Means-tested} & \\multicolumn{3}{c}{Universal} & $\\Delta$CI, means-tested \\\\\n"
          " & All & Low & CI & All & Low & CI & All & Low & CI &",
          rows, size="\\footnotesize")

    # ---- constraint frontiers ---------------------------------------------------------------
    rows = []
    for m in METROS:
        p = fare[m]["pareto"]["public_service"]
        ref = next(r for r in p if r["constraint"] == "cap" and r["level"] == 1.0)["mean_dA"]
        cap = {r["level"]: r for r in p if r["constraint"] == "cap"}
        flo = {r["level"]: r for r in p if r["constraint"] == "floor"}
        cell = lambda r: "infeasible" if r.get("mean_dA") is None or r["status"] != "Optimal" else f"{100 * r['mean_dA'] / ref:.0f}"
        rows.append(f"{NAME[m]} & " + " & ".join(cell(cap[c]) for c in (0.9, 0.7, 0.5, 0.4, 0.3)) + " & "
                    f"{100 * flo[0.0]['low_income_share_of_best']:.0f} & " + " & ".join(cell(flo[f]) for f in (0.9, 0.95)))
    table("frontiers",
          "Accessibility gain retained under a per-tract burden cap and under a low-income floor, public-service "
          "fare, as a percentage of the gain of the accessibility-maximising network. Caps are shares of that "
          "network's highest tract burden. Floors are shares of the largest low-income gain any 25-site network can "
          "deliver; the column `Attained' gives the share reached without a floor.",
          "lrrrrrrrr",
          "Metro & \\multicolumn{5}{c}{Burden cap} & \\multicolumn{3}{c}{Low-income floor} \\\\\n"
          " & 0.9 & 0.7 & 0.5 & 0.4 & 0.3 & Attained (\\%) & 0.9 & 0.95",
          rows)

    # ---- burden distribution ----------------------------------------------------------------
    rows = []
    for m in METROS:
        t = exp[m]["tiers"]["public_service"]
        g, n = t["groups"], t["noise"]
        share40 = 100 * n["population_at_or_above_dnl"]["40"] / n["population_total"]
        share35 = 100 * n["population_at_or_above_dnl"]["35"] / n["population_total"]
        rows.append(f"{NAME[m]} & {interval(t['uncertainty']['ci_B_vs_income'])} & {interval(g['ci_B_vs_minority_share'])} & "
                    f"{interval(g['ci_B_vs_zero_vehicle_share'])} & {g['edr_B_carless_group']['high_carless']:.2f} & "
                    f"{n['max_dnl']:.0f} & {share35:.1f} & {share40:.2f}")
    table("burden",
          "Distribution of burden under the accessibility-maximising network, public-service fare. Concentration "
          "indices of the Burden Index against tract income, minority share and zero-vehicle household share, with "
          "95\\% tract-bootstrap intervals; exposure-disparity ratio (EDR) of the quartile of tracts with most "
          "zero-vehicle households; highest day--night average sound level and the population share at or above 35 "
          "and 40~dB.",
          "llllrrrr",
          "Metro & \\multicolumn{3}{c}{Concentration index of burden against} & EDR & \\multicolumn{3}{c}{Day--night average sound level} \\\\\n"
          " & income & minority share & zero-vehicle share & & Max (dB) & $\\geq$35 dB (\\%) & $\\geq$40 dB (\\%)",
          rows, size="\\footnotesize\\setlength{\\tabcolsep}{4pt}")

    # ---- HOLC diagnostic --------------------------------------------------------------------
    rows = []
    for m in METROS:
        t = exp[m]["tiers"]["public_service"]
        r, rn = t["redlining"], t["redlining"]["random_networks"]
        g = r["by_grade"]
        rows.append(f"{NAME[m]} & {exp[m]['n_tracts_with_holc']} & " + " & ".join(f"{g[k]['mean_B']:.3f}" if k in g else "---" for k in "ABCD")
                    + f" & {interval(t['uncertainty']['rho_B_vs_holc'])} & [{sgn(rn['rho_B_p2.5'])}, {sgn(rn['rho_B_p97.5'])}] & "
                    f"{sgn(r['rho_B_vs_holc_given_candidate_density'])}")
    table("holc",
          "Burden and 1939 HOLC grades under the accessibility-maximising network, public-service fare. Mean Burden "
          "Index by grade; Spearman correlation between burden and HOLC rank (D highest) with a 95\\% tract-bootstrap "
          "interval; the 95\\% range of the same correlation over 2{,}000 random 25-site networks from the candidate "
          "pool; and the partial correlation controlling for the number of candidates within 3~km.",
          "lrrrrrlll",
          "Metro & Tracts & $\\bar B$(A) & $\\bar B$(B) & $\\bar B$(C) & $\\bar B$(D) & Spearman & Random networks & Partial Spearman",
          rows, size="\\footnotesize")

    # ---- operations --------------------------------------------------------------------------
    sim = ops["la"]["simulation"]["results"]
    rows = [f"{r['demand_pax_per_hour']} & {r['mean_wait_min']:.1f} & {r['p90_wait_min']:.1f} & {r['load_factor']:.2f} & "
            f"{min(r['pad_utilisation'], 1.0):.2f} & {r['movements_per_day']:.0f} & {r['dnl_shift_db_vs_assumed_ops']:+.1f}"
            for r in sim]
    table("operations",
          "Simulated operations of a two-pad vertiport with four-seat aircraft, a maximum hold of 10~min and "
          "turnarounds of 5--12~min over a 16-hour day. P90 is the 90th percentile of passenger wait. The last "
          "column is the change in day--night average sound level relative to the movement count assumed in the burden calculation.",
          "rrrrrrr",
          "Demand (pax/h) & Mean wait (min) & P90 wait (min) & Load factor & Pad use & Movements/day & $\\Delta$DNL (dB)",
          rows, size="\\footnotesize")

    rows = []
    for m in METROS:
        by = {(r["wait"], round(r["fare_per_km"], 3)): r for r in ops[m]["accessibility_by_wait"]}
        waits = [w for w in dict.fromkeys(r["wait"] for r in ops[m]["accessibility_by_wait"])]
        for rate, label in ((0.0, "Zero fare"), (0.1, "Public-service")):
            rows.append(f"{NAME[m]} & {label} & " + " & ".join(pct(by[(w, rate)]["mean_dA_pct_of_base"]) for w in waits))
    wait_min = [f"{next(r['wait_min'] for r in ops['la']['accessibility_by_wait'] if r['wait'] == w):.1f}"
                for w in dict.fromkeys(r["wait"] for r in ops["la"]["accessibility_by_wait"])]
    table("wait",
          "Accessibility gain of the full candidate network (\\% of baseline) by passenger wait: the fixed-headway "
          "value and the simulated waits at 10, 30 and 60 passengers per hour.",
          "ll" + "r" * len(wait_min),
          "Metro & Fare & " + " & ".join(f"{w} min" for w in wait_min),
          rows)

    # ---- robustness summary -------------------------------------------------------------------
    rows = []
    for m in METROS:
        par = rob[m]["parameter_sensitivity"]
        calp = rob[m]["calibration_perturbation"]
        uni = [v["uniform"]["ci_dA_vs_income"] for v in par.values()] + [v["uniform"]["ci_dA_vs_income"] for v in calp.values()]
        inc = [v["income"]["ci_dA_vs_income"] for v in par.values()] + [v["income"]["ci_dA_vs_income"] for v in calp.values()]
        gain = [v["income"]["mean_dA_pct_of_base"] for v in par.values()] + [v["income"]["mean_dA_pct_of_base"] for v in calp.values()]
        b, o = rob[m]["burden"], rob[m]["optimisation"]
        rw = b["random_weights"]
        rows.append(f"{NAME[m]} & {sgn(min(uni))} to {sgn(max(uni))} & {sgn(min(inc))} to {sgn(max(inc))} & "
                    f"{pct(min(gain))}--{pct(max(gain))} & {sgn(rw['p2.5'])} to {sgn(rw['p97.5'])} & {100 * o['surrogate_over_greedy']:.0f}")
    n_var = len(rob["la"]["parameter_sensitivity"]) + len(rob["la"]["calibration_perturbation"]) - 2
    table("robustness",
          f"Robustness at the public-service fare, full candidate network. Ranges over the main specification and {n_var} variants: calibrated "
          "ground times perturbed (congestion multipliers $\\pm$20\\%, free flow, terminal time $\\pm$3~min) and one "
          "accessibility parameter changed at a time (distance decay, trip-length limits, cruise speed, security time, "
          "passenger wait, opportunity measure). Burden concentration over 500 random draws of the component weights "
          "(95\\% range), for the network selected under the uniform value of time. Last column: realised gain of "
          "the surrogate-optimal network relative to a greedy network built on the realised objective. VOT: value of time.",
          "llllll",
          "Metro & CI (gain), uniform VOT & CI (gain), income VOT & Gain (\\%) & CI (burden) & Surrogate/greedy (\\%)",
          rows, size="\\footnotesize\\setlength{\\tabcolsep}{4pt}")

    # ---- existing facilities and greenfield ---------------------------------------------------
    rows = []
    for m in METROS:
        pools = green[m]["tiers"]["public_service"]
        if "greenfield_49" not in pools:
            raise SystemExit(f"greenfield results for {m} predate the equal-size pool; rerun scripts/13_greenfield.py")
        for key, label in (("existing_facilities", "Existing facilities"), ("greenfield_49", "Greenfield")):
            r = pools[key]
            rows.append(f"{NAME[m]} & {label} & {pct(r['mean_dA_pct_of_base'])} & {sgn(r['ci_dA_vs_income'])} & "
                        f"{sgn(r['ci_dnl_energy_vs_income'])} & {r['population_dnl_ge_40'] / 1000:.0f} & "
                        f"{r['median_income_of_host_tracts'] / r['metro_median_tract_income']:.2f}")
    table("greenfield",
          "Accessibility-maximising networks from the existing-facility pool and from a greenfield pool of the same "
          "size placed by population coverage, public-service fare. Gain relative to baseline accessibility; income "
          "concentration of the gain and of noise energy; residents at or above 40~dB day-night average sound level; "
          "median income of host tracts relative to the metro median.",
          "llrrrrr",
          "Metro & Candidate pool & Gain (\\%) & CI (gain) & CI (noise) & $\\geq$40 dB (thousand) & Host income ratio",
          rows, size="\\footnotesize")

    print("tables written:", ", ".join(sorted(p.name for p in OUT.glob("*.tex"))))


if __name__ == "__main__":
    main()
