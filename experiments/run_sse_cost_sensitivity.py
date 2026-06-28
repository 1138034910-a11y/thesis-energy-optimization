#!/usr/bin/env python3
"""
================================================================================
SSE Sensitivity to BESS Cost Parameters (Robustness Check)
================================================================================
Purpose: Verify that the SSE three-stage sign-pattern (near-neutral →
         substitution → strong substitution) survives ±20% BESS cost
         perturbation under the SAME protocol used for the main-text results.

Design:
  - 3 cost scenarios × 5 H₂ tank sizes = 15 TSSP runs
  - All parameters identical to main experiment:
      * Gaussian Copula correlation parameter ρ = -0.30
        (corresponds to Kendall τ ≈ -0.20 under the Gaussian Copula mapping)
      * Representative-day seed = 42
      * Carbon price = 80 CNY/t
      * EV warm start (MIPStart) for every TSSP run
  - Only BESS power/energy CAPEX is perturbed.

Critical protocol note:
  The main-text H₂-scale results were generated with rho_override=-0.30 in
  generate_reduced_scenarios. The variable passed to rho_override is the
  Gaussian Copula Pearson correlation ρ, NOT Kendall's τ. For Gaussian Copula:
      τ = (2/π) * arcsin(ρ),  ρ = sin(π/2 * τ).
  Setting ρ = -0.30 yields Kendall τ ≈ -0.20, matching the manuscript.

Execution Mode:
  - SERIAL (Gurobi academic license is single-user; concurrent processes
    would queue for the license token. Each run is assigned 8 threads.)

Runtime Estimate:
  - ~20–24 h for 15 runs (average 1.3–1.6 h per run on 8 Gurobi threads)

Output:
  - CSV tables saved to `results/tables/`
  - Console log of SSE regime classification for each scenario
  - Protocol-consistency check against main-text baseline values
================================================================================
"""

import sys
import os
import time
import numpy as np
import pandas as pd

# ------------------------------------------------------------------------------
# 0. PATH SETUP (robust to execution directory)
# ------------------------------------------------------------------------------
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)
sys.path.insert(0, os.path.join(_project_root, "src"))

from config import (
    EconParams, PhysParams, ScenarioParams, RepDayParams,
    SolverParams, DataPaths, build_load_profile
)
from src.representative_days import run_representative_day_pipeline, scale_annual_constraints
from src.scenario_generator import generate_reduced_scenarios
from src.deterministic_model import build_deterministic_model
from src.stochastic_model import build_two_stage_model, solve_and_extract

# ------------------------------------------------------------------------------
# 1. CONFIGURATION
# ------------------------------------------------------------------------------
REPDAY_SEED = 42
# Gaussian Copula Pearson correlation parameter ρ (NOT Kendall's τ).
# ρ = -0.30 corresponds to Kendall τ ≈ -0.20, matching the main experiment.
COPULA_RHO = -0.30
CARBON_PRICE = 80.0

# H₂ tank sizes (kg) — same five-point grid as the main manuscript
H2_TANK_SIZES = [200_000, 400_000, 600_000, 800_000, 1_000_000]  # 200–1000 t

# Main-text baseline BESS power values, used for protocol-consistency check.
# Source: results/tables/h2_sensitivity_v3_rigorous_vss.csv (ρ=-0.30, EV MIPStart)
MAIN_TEXT_BESS_P = {
    200: 5518.92,
    400: 5758.33,
    600: 4061.03,
    800: 3709.78,
    1000: 2645.60,
}

# Cost scenarios: (label, BESS_P_cost_multiplier, BESS_E_cost_multiplier)
COST_SCENARIOS = [
    ("baseline", 1.00, 1.00),   # matches main-text cost assumptions
    ("bess_low", 0.80, 0.80),   # BESS 20% cheaper
    ("bess_high", 1.20, 1.20),  # BESS 20% more expensive
]

# Solver configuration — serial execution, 8 threads per Gurobi run
SOLVER_CFG = {
    "MIPGap": SolverParams["MIPGap"],
    "TimeLimit": SolverParams["TimeLimit"],
    "Threads": 8,
    "MIPFocus": SolverParams["MIPFocus"],
    "Heuristics": SolverParams["Heuristics"],
    "Presolve": SolverParams["Presolve"],
    "Cuts": SolverParams["Cuts"],
    "Crossover": SolverParams["Crossover"],
    "ImproveStartGap": SolverParams.get("ImproveStartGap", 0.10),
}

MIPGAP_WARNING = 5.0  # %

OUTPUT_DIR = os.path.join(_project_root, "results", "tables")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------------------------------------------------------------------
# 2. DATA PREPARATION (run once, shared across all scenarios)
# ------------------------------------------------------------------------------
def prepare_shared_data():
    """Load KAN forecasts, build representative days, and generate scenarios."""
    print("=" * 70)
    print("SSE Sensitivity to BESS Cost (±20%) — Protocol-Consistent Version")
    print("=" * 70)
    print(f"RepDay seed: {REPDAY_SEED}")
    print(f"Copula Gaussian ρ: {COPULA_RHO} (Kendall τ ≈ {2/np.pi*np.arcsin(COPULA_RHO):.3f})")
    print(f"Carbon price: {CARBON_PRICE} CNY/t")
    print(f"H₂ sizes:    {[k / 1000 for k in H2_TANK_SIZES]} t")
    print(f"Scenarios:   {[s[0] for s in COST_SCENARIOS]}")
    print(f"Total runs:  {len(COST_SCENARIOS) * len(H2_TANK_SIZES)}")
    print(f"MIPStart:    EV warm start for every TSSP run")
    print("=" * 70)

    # --- Load KAN forecasts ---
    kan_path = os.path.join(_project_root, "results", "tables", "kan_forecasts.csv")
    if not os.path.exists(kan_path):
        raise FileNotFoundError(f"KAN forecast file not found: {kan_path}")

    print("\n[1/3] Loading KAN forecasts...")
    kan_df = pd.read_csv(kan_path)
    wind_actual = kan_df["wind_mu"].bfill().values
    solar_actual = kan_df["solar_mu"].bfill().values
    load_full = build_load_profile(8760, PhysParams["Load_Base"])

    # --- Representative-day aggregation ---
    print(f"[2/3] Representative-day aggregation (seed={REPDAY_SEED})...")
    reps, wind_r, solar_r, load_r, weights_days = run_representative_day_pipeline(
        wind_actual, solar_actual, load_full,
        n_days=RepDayParams["n_days"], seed=REPDAY_SEED
    )

    # --- Scenario generation (Copula) ---
    print(f"[3/3] Scenario generation (Copula ρ={COPULA_RHO})...")
    wind_sigma = kan_df["wind_sigma"].values[:len(wind_actual)]
    solar_sigma = kan_df["solar_sigma"].values[:len(solar_actual)]

    wind_sigma_r, solar_sigma_r = [], []
    for d in reps["day_indices"]:
        s, e = d * 24, d * 24 + 24
        wind_sigma_r.extend(wind_sigma[s:e])
        solar_sigma_r.extend(solar_sigma[s:e])
    wind_sigma_r = np.array(wind_sigma_r)
    solar_sigma_r = np.array(solar_sigma_r)

    wind_sc, solar_sc, weights_sc = generate_reduced_scenarios(
        wind_r, wind_sigma_r, solar_r, solar_sigma_r,
        n_sample=ScenarioParams["N_sample"],
        n_scenario=ScenarioParams["N_scenario"],
        seed=REPDAY_SEED,
        rho_override=COPULA_RHO
    )
    print(f"      Scenarios: {len(weights_sc)} (weights: {weights_sc})")

    return {
        "load_r": load_r, "wind_r": wind_r, "solar_r": solar_r,
        "wind_sc": wind_sc, "solar_sc": solar_sc,
        "weights_sc": weights_sc, "reps": reps
    }


# ------------------------------------------------------------------------------
# 3. SINGLE RUN WORKER
# ------------------------------------------------------------------------------
def run_single(job_spec, shared_data):
    """
    Execute one (scenario, H₂ size) pair with EV warm start.
    """
    scenario_label = job_spec["scenario_label"]
    bess_p_mult = job_spec["bess_p_mult"]
    bess_e_mult = job_spec["bess_e_mult"]
    h2_kg = job_spec["h2_kg"]
    run_id = job_spec["run_id"]
    total_runs = job_spec["total_runs"]

    load_r = shared_data["load_r"]
    wind_r = shared_data["wind_r"]
    solar_r = shared_data["solar_r"]
    wind_sc = shared_data["wind_sc"]
    solar_sc = shared_data["solar_sc"]
    weights_sc = shared_data["weights_sc"]
    reps = shared_data["reps"]
    h2_t = h2_kg / 1000

    print(f"\n[{run_id}/{total_runs}] Scenario='{scenario_label}', H₂={h2_t:.0f}t")
    print("-" * 60)

    # Build parameter dict for this scenario
    econ_mod = dict(EconParams)
    econ_mod["Carbon_price"] = CARBON_PRICE * 1e-4
    econ_mod["C_inv_bess_p"] = EconParams["C_inv_bess_p"] * bess_p_mult
    econ_mod["C_inv_bess_e"] = EconParams["C_inv_bess_e"] * bess_e_mult

    phys_mod = dict(PhysParams)
    phys_mod["Cap_H2_Tank"] = h2_kg
    phys_mod["Cap_H2_Tank_Max"] = h2_kg

    econ_base, phys_mod = scale_annual_constraints(econ_mod, phys_mod, reps)
    phys_mod["T"] = len(load_r)

    print(f"      BESS cost: P={econ_base['C_inv_bess_p']:.0f}, "
          f"E={econ_base['C_inv_bess_e']:.0f} (×{bess_p_mult:.2f})")

    t0 = time.time()

    try:
        # --- EV solution (deterministic, used as MIPStart) ---
        print("      [EV] Solving deterministic benchmark for MIPStart...")
        ev_res, _ = build_deterministic_model(
            load_r, wind_r, solar_r, econ_base, phys_mod, SOLVER_CFG
        )
        if not ev_res:
            raise RuntimeError("EV model failed to solve")
        print(f"      [EV] BESS={ev_res['capacity']['BESS_P_MW']:.0f} MW, "
              f"ELC={ev_res['capacity']['ELC_P_MW']:.0f} MW")

        # --- TSSP with EV warm start ---
        m, var_dict = build_two_stage_model(
            load_r, wind_sc, solar_sc, weights_sc,
            econ_base, phys_mod, SOLVER_CFG
        )
        var_dict["x_bess_p"].Start = ev_res["capacity"]["BESS_P_MW"]
        var_dict["x_bess_e"].Start = ev_res["capacity"]["BESS_E_MWh"]
        var_dict["x_elc_p"].Start = ev_res["capacity"]["ELC_P_MW"]
        var_dict["x_h2_tank"].Start = h2_kg
        var_dict["x_fc_p"].Start = ev_res["capacity"]["FC_P_MW"]

        tssp_res, status = solve_and_extract(
            m, var_dict, load_r, wind_sc, solar_sc, weights_sc, phys_mod
        )
    except Exception as e:
        print(f"      ✗ EXCEPTION: {e}")
        tssp_res = None
        status = f"EXCEPTION: {e}"

    solve_time = time.time() - t0

    if tssp_res is None:
        print(f"      ✗ FAILED (status={status})")
        rec = {
            "scenario": scenario_label,
            "bess_p_mult": bess_p_mult,
            "bess_e_mult": bess_e_mult,
            "h2_t": h2_t,
            "h2_kg": h2_kg,
            "status": status,
            "BESS_P_MW": None,
            "BESS_E_MWh": None,
            "ELC_MW": None,
            "FC_MW": None,
            "objval_10k": None,
            "mipgap_pct": None,
            "solve_time_s": round(solve_time, 1),
            "mipgap_warning": False,
        }
    else:
        gap = round(tssp_res["mipgap"] * 100, 2)
        rec = {
            "scenario": scenario_label,
            "bess_p_mult": bess_p_mult,
            "bess_e_mult": bess_e_mult,
            "h2_t": h2_t,
            "h2_kg": h2_kg,
            "status": tssp_res["status"],
            "BESS_P_MW": round(tssp_res["capacity"]["BESS_P_MW"], 2),
            "BESS_E_MWh": round(tssp_res["capacity"]["BESS_E_MWh"], 2),
            "ELC_MW": round(tssp_res["capacity"]["ELC_P_MW"], 2),
            "FC_MW": round(tssp_res["capacity"]["FC_P_MW"], 2),
            "objval_10k": round(tssp_res["objval"] / 10000, 2),
            "mipgap_pct": gap,
            "solve_time_s": round(solve_time, 1),
            "mipgap_warning": gap > MIPGAP_WARNING,
        }
        warn = " ⚠ MIPGAP HIGH" if rec["mipgap_warning"] else ""
        print(f"      ✓ BESS={rec['BESS_P_MW']} MW / {rec['BESS_E_MWh']} MWh, "
              f"ELC={rec['ELC_MW']} MW, Gap={gap}%{warn}")
        print(f"        Solve time: {solve_time / 3600:.2f} h")

    return rec


# ------------------------------------------------------------------------------
# 4. MAIN EXECUTION
# ------------------------------------------------------------------------------
def main():
    shared = prepare_shared_data()

    jobs = []
    run_id = 0
    total_runs = len(COST_SCENARIOS) * len(H2_TANK_SIZES)

    for scenario_label, bess_p_mult, bess_e_mult in COST_SCENARIOS:
        for h2_kg in H2_TANK_SIZES:
            run_id += 1
            jobs.append({
                "scenario_label": scenario_label,
                "bess_p_mult": bess_p_mult,
                "bess_e_mult": bess_e_mult,
                "h2_kg": h2_kg,
                "run_id": run_id,
                "total_runs": total_runs,
            })

    print(f"\n>>> Running {len(jobs)} jobs SERIALLY (Gurobi academic license)")
    print(f">>> Estimated total time: ~20–24 h")
    print(">>> Start time: " + time.strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 70)

    results = []
    for job in jobs:
        rec = run_single(job, shared)
        results.append(rec)
        df_partial = pd.DataFrame(results)
        ts = time.strftime("%Y%m%d_%H%M%S")
        df_partial.to_csv(
            os.path.join(OUTPUT_DIR,
                         f"sse_bess_cost_sensitivity_partial_{ts}.csv"),
            index=False, encoding="utf-8-sig"
        )

    # --- Compute and compare SSE ---
    print("\n" + "=" * 70)
    print("SSE COMPARISON ACROSS COST SCENARIOS")
    print("=" * 70)

    df = pd.DataFrame(results)
    valid = df[df["BESS_P_MW"].notna()]

    def arc_elasticity(q1, q2, x1, x2):
        dq = q2 - q1
        dx = x2 - x1
        qbar = (q1 + q2) / 2
        xbar = (x1 + x2) / 2
        if abs(qbar) < 1e-6 or abs(xbar) < 1e-6:
            return None
        return (dq / qbar) / (dx / xbar)

    intervals = [
        (200_000, 400_000, "200→400 t"),
        (400_000, 600_000, "400→600 t"),
        (600_000, 800_000, "600→800 t"),
        (800_000, 1_000_000, "800→1000 t"),
    ]

    print(f"\n{'Scenario':<15} {'Interval':<18} {'SSE':>8} {'Regime':<22}")
    print("-" * 65)

    for scenario_label in [s[0] for s in COST_SCENARIOS]:
        scen_df = valid[valid["scenario"] == scenario_label]
        for h2_1, h2_2, label in intervals:
            row1 = scen_df[scen_df["h2_kg"] == h2_1]
            row2 = scen_df[scen_df["h2_kg"] == h2_2]
            if len(row1) == 0 or len(row2) == 0:
                continue
            q1 = row1["BESS_P_MW"].values[0]
            q2 = row2["BESS_P_MW"].values[0]
            eps = arc_elasticity(q1, q2, h2_1 / 1000, h2_2 / 1000)
            if eps is None:
                continue
            if eps > 0:
                regime = "Near-neutral"
            elif eps > -1:
                regime = "Substitution"
            else:
                regime = "Strong substitution"
            print(f"{scenario_label:<15} {label:<18} {eps:>8.3f} {regime:<22}")

    # --- Protocol-consistency check ---
    print("\n" + "=" * 70)
    print("PROTOCOL-CONSISTENCY CHECK: baseline vs. main-text values")
    print("=" * 70)
    baseline_df = valid[valid["scenario"] == "baseline"].sort_values("h2_t")
    max_diff_pct = 0.0
    for _, r in baseline_df.iterrows():
        h2_t = int(r["h2_t"])
        main_val = MAIN_TEXT_BESS_P.get(h2_t)
        if main_val is None:
            continue
        diff_pct = abs(r["BESS_P_MW"] - main_val) / main_val * 100
        max_diff_pct = max(max_diff_pct, diff_pct)
        marker = "✓" if diff_pct < 5.0 else "⚠"
        print(f"  {marker} H₂={h2_t}t: this_run={r['BESS_P_MW']:.0f}, "
              f"main_text={main_val:.0f}, diff={diff_pct:.1f}%")
    if max_diff_pct < 5.0:
        print("\n✓ Baseline reproduces main-text values within 5%.")
    else:
        print(f"\n⚠ Baseline deviates from main-text values by up to {max_diff_pct:.1f}%.")
        print("  Possible causes: solver non-uniqueness, parameter mismatch, or code drift.")
        print("  Do NOT use these results as robustness checks until consistency is restored.")

    # --- High-MIPGap warning summary ---
    high_gap = valid[valid["mipgap_warning"] == True]
    if len(high_gap) > 0:
        print(f"\n⚠ WARNING: {len(high_gap)} run(s) exceeded MIPGap threshold "
              f"({MIPGAP_WARNING}%):")
        for _, r in high_gap.iterrows():
            print(f"   {r['scenario']:<15} H₂={r['h2_t']:.0f}t  "
                  f"Gap={r['mipgap_pct']:.2f}%")
    else:
        print(f"\n✓ All runs converged within MIPGap threshold "
              f"({MIPGAP_WARNING}%).")

    # --- Save final ---
    ts = time.strftime("%Y%m%d_%H%M%S")
    final_path = os.path.join(OUTPUT_DIR,
                              f"sse_bess_cost_sensitivity_{ts}.csv")
    df.to_csv(final_path, index=False, encoding="utf-8-sig")
    print(f"\n{'=' * 70}")
    print(f"Final results saved: {final_path}")
    print(f"Finish time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)


if __name__ == "__main__":
    main()
