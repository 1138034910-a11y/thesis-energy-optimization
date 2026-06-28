"""
Simplified Copula correlation sensitivity experiment at the 400 t base case.

Purpose:
  Test whether the SSE sign pattern around the near-neutral/substitution boundary
  is robust to the Gaussian Copula Pearson correlation parameter. The main
  manuscript uses rho = -0.30 (Kendall tau ≈ -0.20). This script tests two
  alternatives:
    - rho = -0.20 (weaker dependence)
    - rho = -0.41 (empirical Kendall tau ≈ -0.27)
  at H2 = 400 t only, which is the regime-transition point.

Protocol:
  - 400 t H2 tank, carbon price = 80 CNY/t.
  - 20 representative days, 4 scenarios.
  - EV solution used as MIPStart.
  - TimeLimit = 7,200 s (2 h) per TSSP to avoid heartbeat loss on workstations.

Output:
  results/tables/copula_sensitivity_400t.csv
  results/tables/copula_sensitivity_400t.json
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
# Experiment configuration
# ============================================================
H2_CAP = 400_000
RHO_VALUES = [-0.20, -0.41]
CARBON_PRICE = 80.0
CARBON_PRICE_MODEL = CARBON_PRICE / 10000.0
TIME_LIMIT = 7200  # 2 hours per TSSP

OUTPUT_CSV = "results/tables/copula_sensitivity_400t.csv"
OUTPUT_JSON = "results/tables/copula_sensitivity_400t.json"
PARTIAL_PREFIX = "results/tables/copula_sensitivity_400t_partial"

print("=" * 70)
print("Copula correlation sensitivity experiment (400 t base case)")
print("=" * 70)
print(f"H2 tank = {H2_CAP/1000:.0f} t, carbon price = {CARBON_PRICE} CNY/t")
print(f"Gaussian Copula rho values: {RHO_VALUES}")
print(f"TimeLimit per TSSP = {TIME_LIMIT}s")
print()

# ============================================================
# Data preparation
# ============================================================
print("[Step 1] Data preparation...")
t0_total = time.time()

df_w = pd.read_csv(DataPaths["wind_pred"])
df_s = pd.read_csv(DataPaths["solar_pred"])
wind_actual_raw = df_w["actual_pu"].bfill().values
solar_actual_raw = df_s["actual_pu"].bfill().values

kan_df = pd.read_csv("results/tables/kan_forecasts.csv")
wind_mu_full = kan_df["wind_mu"].bfill().values
solar_mu_full = kan_df["solar_mu"].bfill().values

n_len = min(len(wind_actual_raw), len(wind_mu_full))
wind_actual_raw = wind_actual_raw[:n_len]
solar_actual_raw = solar_actual_raw[:n_len]
wind_mu_full = wind_mu_full[:n_len]
solar_mu_full = solar_mu_full[:n_len]

wind_actual = wind_mu_full
solar_actual = solar_mu_full
load_full = build_load_profile(8760, PhysParams["Load_Base"])

reps, wind_r, solar_r, load_r, weights_days = run_representative_day_pipeline(
    wind_actual, solar_actual, load_full,
    n_days=RepDayParams["n_days"], seed=RepDayParams["seed"]
)

wind_sigma_full = kan_df["wind_sigma"].values[:n_len]
solar_sigma_full = kan_df["solar_sigma"].values[:n_len]
wind_sigma_r, solar_sigma_r = [], []
for d in reps["day_indices"]:
    s, e = d * 24, (d + 1) * 24
    wind_sigma_r.extend(wind_sigma_full[s:e])
    solar_sigma_r.extend(solar_sigma_full[s:e])
wind_sigma_r = np.array(wind_sigma_r)
solar_sigma_r = np.array(solar_sigma_r)

print(f"  Data prep done: {time.time()-t0_total:.1f}s")
print(f"  Representative days: {RepDayParams['n_days']}, Hours: {len(load_r)}")

# ============================================================
# Main loop over rho
# ============================================================
results = []
solver_params = dict(SolverParams)
solver_params["TimeLimit"] = TIME_LIMIT

for rho in RHO_VALUES:
    print(f"\n{'='*70}")
    print(f"Gaussian Copula rho = {rho}")
    print(f"{'='*70}")

    print("  Generating scenarios...")
    t0 = time.time()
    wind_sc, solar_sc, weights_sc = generate_reduced_scenarios(
        wind_r, wind_sigma_r, solar_r, solar_sigma_r,
        n_sample=ScenarioParams["N_sample"],
        n_scenario=ScenarioParams["N_scenario"],
        seed=ScenarioParams["seed"],
        rho_override=rho
    )
    print(f"  Scenarios generated: {time.time()-t0:.1f}s, weights={weights_sc}")

    cap_t = int(H2_CAP / 1000)
    print(f"\n  H2 tank = {cap_t} t")

    phys_mod = dict(PhysParams)
    phys_mod["Cap_H2_Tank"] = H2_CAP
    phys_mod["Cap_H2_Tank_Max"] = H2_CAP
    phys_mod["T"] = len(load_r)

    econ_base = dict(EconParams)
    econ_base["Carbon_price"] = CARBON_PRICE_MODEL

    res_point = {
        "rho": rho,
        "H2_Tank_t": cap_t,
        "H2_Tank_kg": H2_CAP,
    }

    # ---------- EV ----------
    t0 = time.time()
    print("    [EV] solving...")
    ev_res, _, _ = build_deterministic_model(load_r, wind_r, solar_r, econ_base, phys_mod, solver_params)
    ev_time = time.time() - t0

    if not ev_res:
        print(f"    [FAIL] EV FAILED for rho={rho}")
        res_point["status"] = "EV_FAILED"
        results.append(res_point)
        continue

    print(f"    [OK] EV: Obj={ev_res['objval']:.2f}, Gap={ev_res['mipgap']:.4f}%, Time={ev_time:.1f}s")

    res_point["EV_Obj"] = ev_res["objval"]
    res_point["EV_Gap_pct"] = ev_res["mipgap"]
    res_point["EV_Time_s"] = ev_time
    res_point["EV_BESS_P_MW"] = ev_res["capacity"]["BESS_P_MW"]
    res_point["EV_BESS_E_MWh"] = ev_res["capacity"]["BESS_E_MWh"]
    res_point["EV_ELC_MW"] = ev_res["capacity"]["ELC_P_MW"]
    res_point["EV_FC_MW"] = ev_res["capacity"]["FC_P_MW"]

    # ---------- TSSP ----------
    t0 = time.time()
    print("    [TSSP] solving...")
    tssp_model, tssp_var_dict = build_two_stage_model(
        load_r, wind_sc, solar_sc, weights_sc, econ_base, phys_mod, solver_params
    )
    tssp_var_dict["x_bess_p"].Start = ev_res["capacity"]["BESS_P_MW"]
    tssp_var_dict["x_bess_e"].Start = ev_res["capacity"]["BESS_E_MWh"]
    tssp_var_dict["x_elc_p"].Start = ev_res["capacity"]["ELC_P_MW"]
    tssp_var_dict["x_h2_tank"].Start = H2_CAP
    tssp_var_dict["x_fc_p"].Start = ev_res["capacity"]["FC_P_MW"]

    tssp_res, _ = solve_and_extract(
        tssp_model, tssp_var_dict, load_r, wind_sc, solar_sc, weights_sc, phys_mod
    )
    tssp_time = time.time() - t0

    if not tssp_res:
        print(f"    [FAIL] TSSP FAILED for rho={rho}")
        res_point["status"] = "TSSP_FAILED"
        results.append(res_point)
        continue

    print(f"    [OK] TSSP: Obj={tssp_res['objval']:.2f}, Gap={tssp_res['mipgap']:.4f}%, Time={tssp_time:.1f}s")

    res_point["status"] = "OK"
    res_point["TSSP_Obj"] = tssp_res["objval"]
    res_point["TSSP_Gap_pct"] = tssp_res["mipgap"]
    res_point["TSSP_Time_s"] = tssp_time
    res_point["TSSP_BESS_P_MW"] = tssp_res["capacity"]["BESS_P_MW"]
    res_point["TSSP_BESS_E_MWh"] = tssp_res["capacity"]["BESS_E_MWh"]
    res_point["TSSP_ELC_MW"] = tssp_res["capacity"]["ELC_P_MW"]
    res_point["TSSP_FC_MW"] = tssp_res["capacity"]["FC_P_MW"]

    results.append(res_point)

    # Save partial results after each point
    partial_path = f"{PARTIAL_PREFIX}_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    pd.DataFrame(results).to_csv(partial_path, index=False, encoding="utf-8-sig")
    print(f"    [SAVED] partial: {partial_path}")

# ============================================================
# Summary output
# ============================================================
print(f"\n{'='*70}")
print("Copula sensitivity experiment (400 t) completed")
print(f"{'='*70}")

df = pd.DataFrame(results)
print(df.to_string(index=False))

df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
with open(OUTPUT_JSON, "w") as f:
    json.dump(results, f, indent=2, default=float)

print(f"\nSaved: {OUTPUT_CSV}")
print(f"Saved: {OUTPUT_JSON}")
