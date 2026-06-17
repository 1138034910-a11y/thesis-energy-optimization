#!/usr/bin/env python3
"""
================================================================================
MIPStart Robustness Validation for 400 t H₂ Base Case
================================================================================
Purpose: Verify that the 400 t non-monotonic BESS peak (~5,758 MW) is not
         a solver-artifact by testing multiple MIPStart seeds under IDENTICAL
         model conditions (τ=-0.20, RepDay seed=42, H₂=400 t, carbon=80 CNY/t).

Design:
  - 5 MIPStart strategies, all else held constant.
  - Only first-stage capacity variables receive Start values.
  - Second-stage dispatch variables are left to solver heuristics.

MIPStart seeds:
  0. Replicate main experiment: use published 400 t TSSP capacities.
  1. EV basin: use deterministic EV solution as warm start.
  2. Low-BESS heuristic: start from an artificially low BESS point.
  3. High-BESS heuristic: start from an artificially high BESS point.
  4. No warm start: solver explores from scratch.

Expected runtime: ~1.5–4 h per seed (same as main experiment).
================================================================================
"""

import sys
import os
import time
import json
import numpy as np
import pandas as pd

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)
sys.path.insert(0, os.path.join(_project_root, "src"))

from config import (
    EconParams, PhysParams, KANParams, ScenarioParams, RepDayParams,
    SolverParams, DataPaths, build_load_profile
)
from src.representative_days import run_representative_day_pipeline, scale_annual_constraints
from src.scenario_generator import generate_reduced_scenarios
from src.stochastic_model import build_two_stage_model, solve_and_extract

# ==============================================================================
# 0. EXPERIMENT CONFIGURATION (identical to main experiment)
# ==============================================================================
H2_TANK_KG = 400_000.0          # 400 t — fixed for this validation
CARBON_PRICE = 80.0             # CNY/t — default base-case carbon price
REPDAY_SEED = 42                # identical to main experiment
COPULA_TAU = -0.20              # identical to main experiment

# Solver configuration — use SAME parameters as main experiment
SOLVER_CFG = {
    "MIPGap": SolverParams["MIPGap"],          # 0.02
    "TimeLimit": SolverParams["TimeLimit"],    # 14400 s
    "Threads": SolverParams["Threads"],        # 0 = all cores
    "MIPFocus": SolverParams["MIPFocus"],      # 1
    "Heuristics": SolverParams["Heuristics"],  # 0.5
    "Presolve": SolverParams["Presolve"],      # 2
    "Cuts": SolverParams["Cuts"],              # 2
    "Crossover": SolverParams["Crossover"],    # 0
    "ImproveStartGap": SolverParams.get("ImproveStartGap", 0.10),
}

# MIPStart strategies (label, capacity dict)
MIPSTART_SEEDS = [
    {
        "label": "main_experiment",
        "description": "Replicate published 400t TSSP solution (h2_sensitivity_v3)",
        "cap": {
            "BESS_P_MW": 5758.330101216735,
            "BESS_E_MWh": 21690.961604435244,
            "ELC_P_MW": 30827.25532223626,
            "FC_P_MW": 3677.1649277239294,
        }
    },
    {
        "label": "ev_basin",
        "description": "Deterministic EV solution as warm start",
        "cap": {
            "BESS_P_MW": 5428.247235436475,
            "BESS_E_MWh": 18013.02068435976,
            "ELC_P_MW": 31368.249526919728,
            "FC_P_MW": 3912.8442264489013,
        }
    },
    {
        "label": "low_bess",
        "description": "Artificially low BESS heuristic",
        "cap": {
            "BESS_P_MW": 3000.0,
            "BESS_E_MWh": 10000.0,
            "ELC_P_MW": 35000.0,
            "FC_P_MW": 4000.0,
        }
    },
    {
        "label": "high_bess",
        "description": "Artificially high BESS heuristic",
        "cap": {
            "BESS_P_MW": 8000.0,
            "BESS_E_MWh": 25000.0,
            "ELC_P_MW": 25000.0,
            "FC_P_MW": 3000.0,
        }
    },
    {
        "label": "no_warmstart",
        "description": "No MIPStart — solver explores from scratch",
        "cap": None
    },
]

OUTPUT_DIR = os.path.join(_project_root, "results", "tables")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================================================================
# 1. DATA PREPARATION (run once, shared across all seeds)
# ==============================================================================
print("=" * 70)
print("MIPStart Robustness Validation — 400 t H₂ Base Case")
print("=" * 70)
print(f"H₂ tank: {H2_TANK_KG/1000:.0f} t")
print(f"Carbon price: {CARBON_PRICE} CNY/t")
print(f"RepDay seed: {REPDAY_SEED}")
print(f"Copula τ: {COPULA_TAU}")
print(f"MIPGap: {SOLVER_CFG['MIPGap']*100:.1f}%")
print(f"TimeLimit: {SOLVER_CFG['TimeLimit']/3600:.1f} h")
print(f"MIPStart strategies: {len(MIPSTART_SEEDS)}")
print("=" * 70)

print("\n[1/3] Loading data...")
kan_df = pd.read_csv("results/tables/kan_forecasts.csv")
wind_actual = kan_df["wind_mu"].bfill().values
solar_actual = kan_df["solar_mu"].bfill().values
load_full = build_load_profile(8760, PhysParams["Load_Base"])

print("[2/3] Representative-day aggregation (seed={})...".format(REPDAY_SEED))
reps, wind_r, solar_r, load_r, weights_days = run_representative_day_pipeline(
    wind_actual, solar_actual, load_full,
    n_days=RepDayParams["n_days"], seed=REPDAY_SEED
)
econ_base, phys_mod = scale_annual_constraints(EconParams, PhysParams, reps)
phys_mod["T"] = len(load_r)
# CRITICAL: fix H₂ tank to 400 t for all seeds
phys_mod["Cap_H2_Tank_Max"] = H2_TANK_KG
# Fix carbon price to 80 CNY/t
econ_base["Carbon_price"] = CARBON_PRICE * 1e-4   # convert to 10^4 CNY

T = phys_mod["T"]
print(f"      Representative days: {RepDayParams['n_days']} → T={T} h")

print("[3/3] Scenario generation (Copula τ={})...".format(COPULA_TAU))
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
    rho_override=COPULA_TAU
)
print(f"      Scenarios: {len(weights_sc)} (weights: {weights_sc})")
print("=" * 70)

# ==============================================================================
# 2. LOOP OVER MIPSTART SEEDS
# ==============================================================================
results = []

for idx, ms in enumerate(MIPSTART_SEEDS, 1):
    label = ms["label"]
    cap = ms["cap"]
    print(f"\n[{idx}/{len(MIPSTART_SEEDS)}] MIPStart = '{label}'")
    print(f"      Description: {ms['description']}")
    if cap:
        print(f"      Start values: BESS={cap['BESS_P_MW']:.1f}/{cap['BESS_E_MWh']:.1f}, "
              f"ELC={cap['ELC_P_MW']:.1f}, FC={cap['FC_P_MW']:.1f}, H₂={H2_TANK_KG/1000:.0f}t")
    else:
        print(f"      Start values: NONE (cold start)")
    print("-" * 50)

    t_start = time.time()

    # Build model (identical for all seeds)
    m, var_dict = build_two_stage_model(
        load_r, wind_sc, solar_sc, weights_sc, econ_base, phys_mod, SOLVER_CFG
    )

    # Apply MIPStart if provided
    if cap is not None:
        var_dict["x_bess_p"].Start = cap["BESS_P_MW"]
        var_dict["x_bess_e"].Start = cap["BESS_E_MWh"]
        var_dict["x_elc_p"].Start = cap["ELC_P_MW"]
        var_dict["x_fc_p"].Start = cap["FC_P_MW"]
        var_dict["x_h2_tank"].Start = H2_TANK_KG
        print(f"      [MIPStart] First-stage capacity Start values set.")
    else:
        print(f"      [MIPStart] No Start values — cold start.")

    # Solve
    print(f"      Solving TSSP (MIPGap={SOLVER_CFG['MIPGap']*100:.1f}%, TimeLimit={SOLVER_CFG['TimeLimit']/3600:.1f}h)...")
    t0 = time.time()
    tssp_res, status = solve_and_extract(
        m, var_dict, load_r, wind_sc, solar_sc, weights_sc, phys_mod
    )
    solve_time = time.time() - t0
    total_time = time.time() - t_start

    if tssp_res is None:
        print(f"      ✗ FAILED — status={status}, no feasible solution.")
        results.append({
            "mipstart_label": label,
            "mipstart_description": ms["description"],
            "status": status,
            "BESS_P_MW": None,
            "BESS_E_MWh": None,
            "ELC_P_MW": None,
            "FC_P_MW": None,
            "H2_Tank_kg": None,
            "objval_10k": None,
            "mipgap_pct": None,
            "solve_time_s": round(solve_time, 1),
            "total_time_s": round(total_time, 1),
        })
        continue

    # Extract
    rec = {
        "mipstart_label": label,
        "mipstart_description": ms["description"],
        "status": tssp_res["status"],
        "BESS_P_MW": round(tssp_res["capacity"]["BESS_P_MW"], 2),
        "BESS_E_MWh": round(tssp_res["capacity"]["BESS_E_MWh"], 2),
        "ELC_P_MW": round(tssp_res["capacity"]["ELC_P_MW"], 2),
        "FC_P_MW": round(tssp_res["capacity"]["FC_P_MW"], 2),
        "H2_Tank_kg": round(tssp_res["capacity"]["H2_Tank_kg"], 2),
        "objval_10k": round(tssp_res["objval"] / 10000, 2),
        "mipgap_pct": round(tssp_res["mipgap"] * 100, 2),
        "solve_time_s": round(solve_time, 1),
        "total_time_s": round(total_time, 1),
    }
    results.append(rec)
    print(f"      ✓ SUCCESS")
    print(f"        BESS={rec['BESS_P_MW']} MW / {rec['BESS_E_MWh']} MWh")
    print(f"        ELC={rec['ELC_P_MW']} MW, FC={rec['FC_P_MW']} MW")
    print(f"        Obj={rec['objval_10k']} (10⁴ CNY), Gap={rec['mipgap_pct']}%")
    print(f"        Solve time: {rec['solve_time_s']/3600:.2f} h")

    # Save intermediate result after each seed
    df_partial = pd.DataFrame(results)
    ts = time.strftime("%Y%m%d_%H%M%S")
    partial_path = os.path.join(OUTPUT_DIR, f"mipstart_robustness_400t_partial_{ts}.csv")
    df_partial.to_csv(partial_path, index=False, encoding="utf-8-sig")
    print(f"      [Saved] {partial_path}")

# ==============================================================================
# 3. FINAL SUMMARY
# ==============================================================================
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

df_results = pd.DataFrame(results)
print(df_results.to_string(index=False))

# Robustness assessment across successful runs
print("\n" + "-" * 70)
print("Robustness Assessment (successful runs only)")
print("-" * 70)

success_df = df_results[df_results["BESS_P_MW"].notna()]
if len(success_df) >= 2:
    for col in ["BESS_P_MW", "BESS_E_MWh", "ELC_P_MW", "FC_P_MW", "objval_10k"]:
        vals = success_df[col].astype(float)
        mean_v = vals.mean()
        std_v = vals.std()
        cv = abs(std_v / mean_v) * 100 if mean_v != 0 else 0
        status = "✅ Robust" if cv < 5 else ("~ Moderate" if cv < 10 else "⚠️ High variance")
        print(f"  {col:15s}: mean={mean_v:12.2f}, std={std_v:10.2f}, CV={cv:6.2f}%  {status}")

    # Specific assessment for the 400t peak claim
    bess_p_vals = success_df["BESS_P_MW"].astype(float).values
    bess_p_range = bess_p_vals.max() - bess_p_vals.min()
    bess_p_cv = abs(success_df["BESS_P_MW"].astype(float).std() / success_df["BESS_P_MW"].astype(float).mean()) * 100
    print(f"\n  BESS_P range: {bess_p_range:.1f} MW (CV={bess_p_cv:.2f}%)")
    if bess_p_cv < 5:
        print("  → The 400t BESS peak is ROBUST across MIPStart seeds.")
    elif bess_p_cv < 10:
        print("  → The 400t BESS peak is MODERATELY robust; minor seed-dependent variation.")
    else:
        print("  → The 400t BESS peak is SENSITIVE to MIPStart; may be a local basin feature.")
else:
    print("  Insufficient successful runs for robustness assessment.")

# Save final
print("\n" + "=" * 70)
ts = time.strftime("%Y%m%d_%H%M%S")
final_path = os.path.join(OUTPUT_DIR, f"mipstart_robustness_400t_{ts}.csv")
df_results.to_csv(final_path, index=False, encoding="utf-8-sig")
print(f"Final results saved: {final_path}")
print("=" * 70)
