# -*- coding: utf-8 -*-
"""
Ablation: Copula vs Independent sampling for 400t base case.
Purpose: Defensively verify that core findings (SSE, carbon threshold, VSS)
are insensitive to the Copula module.
Only TSSP is re-run; EV uses existing results.
"""
import os, sys, time

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)
sys.path.insert(0, os.path.join(_project_root, "src"))
os.chdir(_project_root)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "2"

import numpy as np
import pandas as pd

from config import EconParams, PhysParams, ScenarioParams, SolverParams, DataPaths, build_load_profile, RepDayParams
from src.representative_days import run_representative_day_pipeline
from src.scenario_generator import generate_reduced_scenarios
from src.deterministic_model import build_deterministic_model
from src.stochastic_model import build_two_stage_model, solve_and_extract

print("=" * 70)
print("Ablation: Copula vs Independent sampling (400t base case)")
print("=" * 70)
print(f"Solver: MIPGap={SolverParams['MIPGap']}, TimeLimit={SolverParams['TimeLimit']}s")
print()

# === Load data ===
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

# BUG-01 fix: EV input uses KAN-mu
wind_actual = wind_mu_full
solar_actual = solar_mu_full
load_full = build_load_profile(8760, PhysParams["Load_Base"])

# Representative days
reps, wind_r, solar_r, load_r, weights_days = run_representative_day_pipeline(
    wind_actual, solar_actual, load_full,
    n_days=RepDayParams["n_days"], seed=RepDayParams["seed"]
)

# Extract sigma for representative days
wind_sigma_full = kan_df["wind_sigma"].values[:n_len]
solar_sigma_full = kan_df["solar_sigma"].values[:n_len]
wind_sigma_r, solar_sigma_r = [], []
for d in reps["day_indices"]:
    s = d * 24
    e = s + 24
    wind_sigma_r.extend(wind_sigma_full[s:e])
    solar_sigma_r.extend(solar_sigma_full[s:e])
wind_sigma_r = np.array(wind_sigma_r)
solar_sigma_r = np.array(solar_sigma_r)

# === Setup 400t ===
cap_h2 = 400_000
phys_mod = dict(PhysParams)
phys_mod["Cap_H2_Tank"] = cap_h2
phys_mod["Cap_H2_Tank_Max"] = cap_h2
phys_mod["T"] = len(load_r)
econ_base = dict(EconParams)

# ---------- EV (shared, run once) ----------
print("[EV] Solving...")
t0 = time.time()
ev_res, _ = build_deterministic_model(load_r, wind_r, solar_r, econ_base, phys_mod, SolverParams)
ev_time = time.time() - t0
print(f"  EV: Obj={ev_res['objval']:.2f}, Gap={ev_res['mipgap']:.4f}%, Time={ev_time:.1f}s")
print(f"     Cap: BESS_P={ev_res['capacity']['BESS_P_MW']:.0f}, "
      f"ELC={ev_res['capacity']['ELC_P_MW']:.0f}, FC={ev_res['capacity']['FC_P_MW']:.0f}")

results = []

# ---------- TSSP with Copula (baseline, rho=-0.30) ----------
print("\n[TSSP-Copula] rho=-0.30 (baseline)...")
wind_sc_cop, solar_sc_cop, weights_sc = generate_reduced_scenarios(
    wind_r, wind_sigma_r, solar_r, solar_sigma_r,
    n_sample=ScenarioParams["N_sample"], n_scenario=ScenarioParams["N_scenario"],
    seed=ScenarioParams["seed"], rho_override=-0.30
)

t0 = time.time()
tssp_model, tssp_var_dict = build_two_stage_model(
    load_r, wind_sc_cop, solar_sc_cop, weights_sc, econ_base, phys_mod, SolverParams
)
tssp_var_dict["x_bess_p"].Start = ev_res["capacity"]["BESS_P_MW"]
tssp_var_dict["x_bess_e"].Start = ev_res["capacity"]["BESS_E_MWh"]
tssp_var_dict["x_elc_p"].Start = ev_res["capacity"]["ELC_P_MW"]
tssp_var_dict["x_h2_tank"].Start = cap_h2
tssp_var_dict["x_fc_p"].Start = ev_res["capacity"]["FC_P_MW"]

tssp_cop, _ = solve_and_extract(tssp_model, tssp_var_dict, load_r, wind_sc_cop, solar_sc_cop, weights_sc, phys_mod)
tssp_cop_time = time.time() - t0
print(f"  TSSP-Copula: Obj={tssp_cop['objval']:.2f}, Gap={tssp_cop['mipgap']:.4f}%, Time={tssp_cop_time:.1f}s")
print(f"     Cap: BESS_P={tssp_cop['capacity']['BESS_P_MW']:.0f}, "
      f"ELC={tssp_cop['capacity']['ELC_P_MW']:.0f}, FC={tssp_cop['capacity']['FC_P_MW']:.0f}")

results.append({
    "Method": "Copula (rho=-0.30)",
    "BESS_P_MW": tssp_cop['capacity']['BESS_P_MW'],
    "BESS_E_MWh": tssp_cop['capacity']['BESS_E_MWh'],
    "ELC_MW": tssp_cop['capacity']['ELC_P_MW'],
    "FC_MW": tssp_cop['capacity']['FC_P_MW'],
    "Obj": tssp_cop['objval'],
    "Gap_pct": tssp_cop['mipgap'],
    "Time_s": tssp_cop_time,
})

# ---------- TSSP with Independent sampling ----------
print("\n[TSSP-Independent] use_copula=False...")
wind_sc_ind, solar_sc_ind, weights_sc_ind = generate_reduced_scenarios(
    wind_r, wind_sigma_r, solar_r, solar_sigma_r,
    n_sample=ScenarioParams["N_sample"], n_scenario=ScenarioParams["N_scenario"],
    seed=ScenarioParams["seed"], use_copula=False
)

t0 = time.time()
tssp_model2, tssp_var_dict2 = build_two_stage_model(
    load_r, wind_sc_ind, solar_sc_ind, weights_sc_ind, econ_base, phys_mod, SolverParams
)
tssp_var_dict2["x_bess_p"].Start = ev_res["capacity"]["BESS_P_MW"]
tssp_var_dict2["x_bess_e"].Start = ev_res["capacity"]["BESS_E_MWh"]
tssp_var_dict2["x_elc_p"].Start = ev_res["capacity"]["ELC_P_MW"]
tssp_var_dict2["x_h2_tank"].Start = cap_h2
tssp_var_dict2["x_fc_p"].Start = ev_res["capacity"]["FC_P_MW"]

tssp_ind, _ = solve_and_extract(tssp_model2, tssp_var_dict2, load_r, wind_sc_ind, solar_sc_ind, weights_sc_ind, phys_mod)
tssp_ind_time = time.time() - t0
print(f"  TSSP-Indep: Obj={tssp_ind['objval']:.2f}, Gap={tssp_ind['mipgap']:.4f}%, Time={tssp_ind_time:.1f}s")
print(f"     Cap: BESS_P={tssp_ind['capacity']['BESS_P_MW']:.0f}, "
      f"ELC={tssp_ind['capacity']['ELC_P_MW']:.0f}, FC={tssp_ind['capacity']['FC_P_MW']:.0f}")

results.append({
    "Method": "Independent",
    "BESS_P_MW": tssp_ind['capacity']['BESS_P_MW'],
    "BESS_E_MWh": tssp_ind['capacity']['BESS_E_MWh'],
    "ELC_MW": tssp_ind['capacity']['ELC_P_MW'],
    "FC_MW": tssp_ind['capacity']['FC_P_MW'],
    "Obj": tssp_ind['objval'],
    "Gap_pct": tssp_ind['mipgap'],
    "Time_s": tssp_ind_time,
})

# ---------- Summary ----------
print(f"\n{'='*70}")
print("ABLACTION SUMMARY (400t base case)")
print(f"{'='*70}")
print(f"{'Method':<22} {'BESS_P':>10} {'ELC':>10} {'FC':>10} {'Obj':>14} {'Gap':>8}")
print("-" * 80)
for r in results:
    print(f"{r['Method']:<22} {r['BESS_P_MW']:>10.0f} {r['ELC_MW']:>10.0f} "
          f"{r['FC_MW']:>10.0f} {r['Obj']:>14.2f} {r['Gap_pct']:>7.2f}%")

print(f"\nDifferences (Independent vs Copula):")
print(f"  BESS_P: {(results[1]['BESS_P_MW']-results[0]['BESS_P_MW'])/results[0]['BESS_P_MW']*100:+.2f}%")
print(f"  ELC:    {(results[1]['ELC_MW']-results[0]['ELC_MW'])/results[0]['ELC_MW']*100:+.2f}%")
print(f"  FC:     {(results[1]['FC_MW']-results[0]['FC_MW'])/results[0]['FC_MW']*100:+.2f}%")
print(f"  Obj:    {(results[1]['Obj']-results[0]['Obj'])/results[0]['Obj']*100:+.3f}%")

df_results = pd.DataFrame(results)
df_results.to_csv("results/tables/ablation_copula_independent_400t.csv", index=False)
print(f"\nResults saved to results/tables/ablation_copula_independent_400t.csv")
print(f"Total time: {time.time()-t0_total:.1f}s")
