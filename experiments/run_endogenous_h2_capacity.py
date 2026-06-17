"""
Endogenous H2 Tank Capacity — Single-Run Sensitivity Test
===========================================================

Objective:
    Relax the H2 tank capacity upper bound and let the optimiser choose
    the endogenous capacity alongside BESS, ELC, and FC. This validates
    the exogenous-scanning design by showing where the unconstrained
    optimum lies relative to the five-point grid (200–1,000 t).

Method:
    - Data preparation identical to run_h2_sensitivity_v3_rigorous_vss.py
    - Cap_H2_Tank_Max = 2,000,000 kg (2,000 t), far above the scanned grid
    - EV warm start -> TSSP solve
    - No EEV (this is a structural diagnostic, not a VSS experiment)

Expected outcome:
    The objective decreases monotonically with H2 scale (Table S10),
    so the endogenous optimum will likely lie near or at the upper bound.
    This confirms that an endogenous-only model would miss the 200–800 t
    regime transition entirely, validating the exogenous scanning design.

Runtime: ~4 h (one TSSP solve, same as a standard base-case run).
"""
import os
import sys
import time
import json

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)
sys.path.insert(0, os.path.join(_project_root, "src"))
os.chdir(_project_root)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "2"

import numpy as np
import pandas as pd

from config import (
    EconParams, PhysParams, ScenarioParams, SolverParams,
    DataPaths, build_load_profile, RepDayParams
)
from src.representative_days import run_representative_day_pipeline
from src.scenario_generator import generate_reduced_scenarios
from src.deterministic_model import build_deterministic_model
from src.stochastic_model import build_two_stage_model, solve_and_extract

# ============================================================
# Experiment Configuration
# ============================================================
OUTPUT_JSON = "results/tables/endogenous_h2_capacity.json"
OUTPUT_CSV = "results/tables/endogenous_h2_capacity.csv"
H2_UPPER_BOUND_KG = 2_000_000   # 2,000 t — well above the scanned grid

CARBON_PRICE_CNY_PER_TON = 80   # Base-case carbon price

print("=" * 70)
print("Endogenous H2 Tank Capacity — Single-Run Sensitivity Test")
print("=" * 70)
print(f"H2 upper bound: {H2_UPPER_BOUND_KG / 1000:.0f} t")
print(f"Carbon price: {CARBON_PRICE_CNY_PER_TON} CNY/t")
print(f"Solver: MIPGap={SolverParams['MIPGap']}, TimeLimit={SolverParams['TimeLimit']} s")
print()

# ============================================================
# Step 1: Data Preparation (identical to v3 rigorous VSS)
# ============================================================
print("[Step 1] Data preparation...")
t0_total = time.time()

# 1. Raw actual data (length alignment only)
df_w = pd.read_csv(DataPaths["wind_pred"])
df_s = pd.read_csv(DataPaths["solar_pred"])
wind_actual_raw = df_w["actual_pu"].bfill().values
solar_actual_raw = df_s["actual_pu"].bfill().values

# 2. KAN predictive means (mu) and uncertainties (sigma)
kan_df = pd.read_csv("results/tables/kan_forecasts.csv")
wind_mu_full = kan_df["wind_mu"].bfill().values
solar_mu_full = kan_df["solar_mu"].bfill().values

# 3. Align lengths
n_len = min(len(wind_actual_raw), len(wind_mu_full))
wind_actual_raw = wind_actual_raw[:n_len]
solar_actual_raw = solar_actual_raw[:n_len]
wind_mu_full = wind_mu_full[:n_len]
solar_mu_full = solar_mu_full[:n_len]

# 4. EV input uses KAN-mu (BUG-01 fix, consistent with v3)
wind_actual = wind_mu_full
solar_actual = solar_mu_full

load_full = build_load_profile(8760, PhysParams["Load_Base"])

# 5. Representative-day aggregation
reps, wind_r, solar_r, load_r, weights_days = run_representative_day_pipeline(
    wind_actual, solar_actual, load_full,
    n_days=RepDayParams["n_days"], seed=RepDayParams["seed"]
)

# 6. Extract sigmas for the selected representative days
wind_sigma_full = kan_df["wind_sigma"].values[:n_len]
solar_sigma_full = kan_df["solar_sigma"].values[:n_len]
wind_sigma_r = []
solar_sigma_r = []
for d in reps["day_indices"]:
    s = d * 24
    e = s + 24
    wind_sigma_r.extend(wind_sigma_full[s:e])
    solar_sigma_r.extend(solar_sigma_full[s:e])
wind_sigma_r = np.array(wind_sigma_r)
solar_sigma_r = np.array(solar_sigma_r)

# 7. Scenario generation (fixed seed, Gaussian Copula)
wind_sc, solar_sc, weights_sc = generate_reduced_scenarios(
    wind_r, wind_sigma_r, solar_r, solar_sigma_r,
    n_sample=ScenarioParams["N_sample"],
    n_scenario=ScenarioParams["N_scenario"],
    seed=ScenarioParams["seed"],
    rho_override=-0.30
)

print(f"  Data ready: {time.time() - t0_total:.1f} s")
print(f"  RepDays: {RepDayParams['n_days']}, Scenarios: {len(weights_sc)}")
print()

# ============================================================
# Step 2: Modify parameters for endogenous H2
# ============================================================
print("[Step 2] Parameter modification for endogenous H2...")

phys_mod = dict(PhysParams)
phys_mod["Cap_H2_Tank_Max"] = H2_UPPER_BOUND_KG
phys_mod["T"] = len(load_r)
# NOTE: Cap_H2_Tank is NOT set here because it is unused by the model.
# The model only reads Cap_H2_Tank_Max as the upper bound of x_h2_tank.

econ_mod = dict(EconParams)
econ_mod["Carbon_price"] = CARBON_PRICE_CNY_PER_TON * 1e-4  # Convert to 10^4 CNY/ton

print(f"  Cap_H2_Tank_Max = {H2_UPPER_BOUND_KG} kg")
print(f"  Carbon_price = {econ_mod['Carbon_price']} (10^4 CNY/ton)")
print()

# ============================================================
# Step 3: EV Solve (H2 endogenous)
# ============================================================
print("[Step 3] EV solve (H2 endogenous)...")
t0 = time.time()

ev_res, _, _ = build_deterministic_model(load_r, wind_r, solar_r, econ_mod, phys_mod, SolverParams)
ev_time = time.time() - t0

if not ev_res:
    print("  ERROR: EV solve failed. Aborting.")
    sys.exit(1)

print(f"  EV: Obj={ev_res['objval']:.2f}, Gap={ev_res['mipgap']:.4f}%, Time={ev_time:.1f}s")
print(f"      Cap: BESS_P={ev_res['capacity']['BESS_P_MW']:.0f}, "
      f"BESS_E={ev_res['capacity']['BESS_E_MWh']:.0f}, "
      f"ELC={ev_res['capacity']['ELC_P_MW']:.0f}, "
      f"FC={ev_res['capacity']['FC_P_MW']:.0f}, "
      f"H2={ev_res['capacity']['H2_Tank_kg']:.0f}")
print()

# ============================================================
# Step 4: TSSP Solve (H2 endogenous) with EV warm start
# ============================================================
print("[Step 4] TSSP solve (H2 endogenous)...")
t0 = time.time()

tssp_model, tssp_var_dict = build_two_stage_model(
    load_r, wind_sc, solar_sc, weights_sc, econ_mod, phys_mod, SolverParams
)

# Warm start from EV solution
tssp_var_dict["x_bess_p"].Start = ev_res["capacity"]["BESS_P_MW"]
tssp_var_dict["x_bess_e"].Start = ev_res["capacity"]["BESS_E_MWh"]
tssp_var_dict["x_elc_p"].Start = ev_res["capacity"]["ELC_P_MW"]
tssp_var_dict["x_h2_tank"].Start = ev_res["capacity"]["H2_Tank_kg"]
tssp_var_dict["x_fc_p"].Start = ev_res["capacity"]["FC_P_MW"]

tssp_res, tssp_status = solve_and_extract(
    tssp_model, tssp_var_dict, load_r, wind_sc, solar_sc, weights_sc, phys_mod
)
tssp_time = time.time() - t0

if not tssp_res:
    print(f"  ERROR: TSSP solve failed (status={tssp_status}).")
    sys.exit(1)

print(f"  TSSP: Obj={tssp_res['objval']:.2f}, Gap={tssp_res['mipgap']:.4f}%, Time={tssp_time:.1f}s")
print(f"        Cap: BESS_P={tssp_res['capacity']['BESS_P_MW']:.0f}, "
      f"BESS_E={tssp_res['capacity']['BESS_E_MWh']:.0f}, "
      f"ELC={tssp_res['capacity']['ELC_P_MW']:.0f}, "
      f"FC={tssp_res['capacity']['FC_P_MW']:.0f}, "
      f"H2={tssp_res['capacity']['H2_Tank_kg']:.0f}")
print()

# ============================================================
# Step 5: Cross-check against the fixed-grid results
# ============================================================
print("[Step 5] Cross-check against fixed-grid results...")

GRID_RESULTS = {
    200_000: {"tssp_obj": -2690443.76, "bess_p": 5519},
    400_000: {"tssp_obj": -2850892.56, "bess_p": 5758},
    600_000: {"tssp_obj": -3015915.87, "bess_p": 4061},
    800_000: {"tssp_obj": -3165933.36, "bess_p": 3710},
    1000_000: {"tssp_obj": -3311397.17, "bess_p": 2646},
}

endog_h2 = tssp_res["capacity"]["H2_Tank_kg"]
endog_obj = tssp_res["objval"]
endog_bess = tssp_res["capacity"]["BESS_P_MW"]

# Find nearest grid point
nearest_cap = min(GRID_RESULTS.keys(), key=lambda c: abs(c - endog_h2))
nearest = GRID_RESULTS[nearest_cap]

print(f"  Endogenous H2 capacity: {endog_h2:.0f} kg ({endog_h2/1000:.0f} t)")
print(f"  Nearest grid point: {nearest_cap/1000:.0f} t")
print(f"  Endogenous BESS_P: {endog_bess:.0f} MW")
print(f"  Grid BESS_P at {nearest_cap/1000:.0f} t: {nearest['bess_p']} MW")
print(f"  Endogenous objective: {endog_obj:.2f}")
print(f"  Grid objective at {nearest_cap/1000:.0f} t: {nearest['tssp_obj']:.2f}")

if endog_h2 > 1_000_000:
    print(f"\n  >>> INTERPRETATION: Endogenous optimum lies ABOVE the scanned grid.")
    print(f"      This confirms that exogenous scanning is necessary to reveal")
    print(f"      the regime map between 200 t and 1,000 t.")
elif endog_h2 < 200_000:
    print(f"\n  >>> INTERPRETATION: Endogenous optimum lies BELOW the scanned grid.")
    print(f"      This is unexpected; check solver convergence.")
else:
    print(f"\n  >>> INTERPRETATION: Endogenous optimum falls INSIDE the scanned grid.")
    print(f"      The exogenous scan covers the endogenous optimum.")
print()

# ============================================================
# Step 6: Save results
# ============================================================
print("[Step 6] Saving results...")

result = {
    "run_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "carbon_price_CNY_per_ton": CARBON_PRICE_CNY_PER_TON,
    "h2_upper_bound_kg": H2_UPPER_BOUND_KG,
    "ev": {
        "obj": ev_res["objval"],
        "mipgap": ev_res["mipgap"],
        "runtime_s": ev_time,
        "capacity": ev_res["capacity"],
    },
    "tssp": {
        "obj": tssp_res["objval"],
        "mipgap": tssp_res["mipgap"],
        "runtime_s": tssp_time,
        "capacity": tssp_res["capacity"],
        "costs": tssp_res["costs"],
    },
    "comparison": {
        "endogenous_h2_kg": endog_h2,
        "nearest_grid_point_t": nearest_cap / 1000,
        "grid_obj_at_nearest": nearest["tssp_obj"],
        "grid_bess_p_at_nearest": nearest["bess_p"],
    }
}

os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2, default=float)
print(f"  JSON saved: {OUTPUT_JSON}")

df_out = pd.DataFrame([{
    "Run": "endogenous_h2",
    "CarbonPrice_CNY_t": CARBON_PRICE_CNY_PER_TON,
    "H2_UB_kg": H2_UPPER_BOUND_KG,
    "EV_Obj": ev_res["objval"],
    "EV_H2_kg": ev_res["capacity"]["H2_Tank_kg"],
    "EV_BESS_P_MW": ev_res["capacity"]["BESS_P_MW"],
    "EV_BESS_E_MWh": ev_res["capacity"]["BESS_E_MWh"],
    "EV_ELC_MW": ev_res["capacity"]["ELC_P_MW"],
    "EV_FC_MW": ev_res["capacity"]["FC_P_MW"],
    "TSSP_Obj": tssp_res["objval"],
    "TSSP_MIPGap": tssp_res["mipgap"],
    "TSSP_Time_s": tssp_time,
    "TSSP_H2_kg": tssp_res["capacity"]["H2_Tank_kg"],
    "TSSP_BESS_P_MW": tssp_res["capacity"]["BESS_P_MW"],
    "TSSP_BESS_E_MWh": tssp_res["capacity"]["BESS_E_MWh"],
    "TSSP_ELC_MW": tssp_res["capacity"]["ELC_P_MW"],
    "TSSP_FC_MW": tssp_res["capacity"]["FC_P_MW"],
}])
df_out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
print(f"  CSV saved: {OUTPUT_CSV}")

print()
print("=" * 70)
print(f"Total runtime: {time.time() - t0_total:.1f} s")
print("=" * 70)
print("Next steps:")
print("  1. Check TSSP_H2_kg in the CSV. If it hits the upper bound (2,000,000 kg),")
print("     this validates Supplementary Note 12: the endogenous optimum lies")
print("     beyond the scanned grid, confirming the necessity of exogenous scanning.")
print("  2. If TSSP_H2_kg is well below the bound (e.g. < 1,200,000 kg), update")
print("     Note 12 accordingly: the scan still covers the endogenous optimum.")
print("  3. Share the CSV with me; I will update the manuscript text if needed.")
print("=" * 70)
