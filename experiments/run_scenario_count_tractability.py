"""
Scenario-count comparison for the 400 t base case.

Purpose:
  Test whether an 8-scenario TSSP produces materially different first-stage
  capacities than the 4-scenario design used in the main manuscript. If the
  8-scenario run is tractable, the comparison strengthens the robustness claim
  for the 4-scenario design. If it is not, the failure statistics document the
  tractability boundary.

Protocol:
  - 400 t H2 tank, carbon price = 80 CNY/t (base case).
  - Representative days = 20, hours = 480.
  - n_scenario = 8 with the same Gaussian Copula (rho=-0.30).
  - EV solution used as MIPStart.
  - TimeLimit = 14,400 s (4 h), matching the main experiment.
  - Log MIPGap and incumbent/best-bound trajectory via Gurobi callback.

Output:
  results/tables/scenario_count_comparison.json
  results/tables/scenario_count_comparison.csv
"""
import os
import sys
import time
import json
import tracemalloc

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)
sys.path.insert(0, os.path.join(_project_root, "src"))
os.chdir(_project_root)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "2"

import numpy as np
import pandas as pd
import gurobipy as gp

from config import (
    EconParams, PhysParams, ScenarioParams, SolverParams,
    DataPaths, build_load_profile, RepDayParams
)
from src.representative_days import run_representative_day_pipeline
from src.scenario_generator import generate_reduced_scenarios
from src.deterministic_model import build_deterministic_model
from src.stochastic_model import build_two_stage_model, solve_and_extract

# ============================================================
# Configuration
# ============================================================
H2_CAP = 400_000
CARBON_PRICE = 80.0  # CNY/t
CARBON_PRICE_MODEL = CARBON_PRICE / 10000.0  # model units: 10^4 CNY/t
N_SCENARIOS = 8
TIME_LIMIT_TEST = 14_400  # 4 hours, matching the main hydrogen-sensitivity protocol

OUTPUT_JSON = "results/tables/scenario_count_comparison.json"
OUTPUT_CSV = "results/tables/scenario_count_comparison.csv"
PARTIAL_PREFIX = "results/tables/scenario_count_comparison_partial"

print("=" * 70)
print("Scenario-count comparison")
print("=" * 70)
print(f"H2 tank = {H2_CAP/1000:.0f} t, carbon price = {CARBON_PRICE} CNY/t")
print(f"Comparing n_scenario = {N_SCENARIOS} against main n_scenario = 4")
print(f"TimeLimit = {TIME_LIMIT_TEST}s")
print()

# ============================================================
# Data preparation
# ============================================================
print("[1/4] Data preparation...")
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

# ============================================================
# EV solution for MIPStart
# ============================================================
print("[2/4] EV solution (MIPStart)...")
phys_mod = dict(PhysParams)
phys_mod["Cap_H2_Tank"] = H2_CAP
phys_mod["Cap_H2_Tank_Max"] = H2_CAP
phys_mod["T"] = len(load_r)
econ_base = dict(EconParams)
econ_base["Carbon_price"] = CARBON_PRICE_MODEL

t0 = time.time()
ev_res, _, _ = build_deterministic_model(load_r, wind_r, solar_r, econ_base, phys_mod, SolverParams)
ev_time = time.time() - t0

if not ev_res:
    print("  [FAIL] EV FAILED")
    sys.exit(1)

print(f"  [OK] EV: Obj={ev_res['objval']:.2f}, Gap={ev_res['mipgap']:.4f}%, Time={ev_time:.1f}s")

# ============================================================
# 8-scenario generation
# ============================================================
print("[3/4] Generating 8 scenarios...")
t0 = time.time()
wind_sc, solar_sc, weights_sc = generate_reduced_scenarios(
    wind_r, wind_sigma_r, solar_r, solar_sigma_r,
    n_sample=ScenarioParams["N_sample"],
    n_scenario=N_SCENARIOS,
    seed=ScenarioParams["seed"],
    rho_override=-0.30
)
scen_time = time.time() - t0
print(f"  Scenarios generated: {scen_time:.1f}s")
print(f"  Scenario weights: {weights_sc}")

# ============================================================
# Build 8-scenario TSSP and record model size
# ============================================================
print("[4/4] Building 8-scenario TSSP...")
t0 = time.time()

test_solver_params = dict(SolverParams)
test_solver_params["TimeLimit"] = TIME_LIMIT_TEST
test_solver_params["MIPGap"] = 0.02

tssp_model, tssp_var_dict = build_two_stage_model(
    load_r, wind_sc, solar_sc, weights_sc, econ_base, phys_mod, test_solver_params
)
build_time = time.time() - t0

# Ensure model-size attributes are populated before reading them.
tssp_model.update()

n_vars = tssp_model.NumVars
n_intvars = tssp_model.NumIntVars
n_binvars = tssp_model.NumBinVars
n_constrs = tssp_model.NumConstrs
n_nz = tssp_model.NumNZs

print(f"  Model built: {build_time:.1f}s")
print(f"  Variables: {n_vars:,} (integer {n_intvars:,}, binary {n_binvars:,})")
print(f"  Constraints: {n_constrs:,}, Nonzeros: {n_nz:,}")

# Start memory tracing
tracemalloc.start()

# Progress log via callback
progress_log = []

def log_callback(model, where):
    if where == gp.GRB.Callback.MIP:
        nodecnt = model.cbGet(gp.GRB.Callback.MIP_NODCNT)
        objbst = model.cbGet(gp.GRB.Callback.MIP_OBJBST)
        objbnd = model.cbGet(gp.GRB.Callback.MIP_OBJBND)
        runtime = model.cbGet(gp.GRB.Callback.RUNTIME)

        # Avoid division by infinity / huge placeholder values before first incumbent
        if objbst >= gp.GRB.INFINITY or abs(objbst) < 1.0:
            gap = float("nan")
        else:
            gap = abs(objbst - objbnd) / (abs(objbst) + 1e-10)

        if len(progress_log) == 0 or runtime - progress_log[-1]["runtime"] >= 60.0:
            progress_log.append({
                "runtime": runtime,
                "nodecnt": nodecnt,
                "objbst": objbst,
                "objbnd": objbnd,
                "gap": gap
            })
            gap_str = f"{gap*100:6.2f}%" if not (gap != gap) else "   N/A"
            print(f"    {runtime:7.1f}s | nodes={nodecnt:>10,.0f} | gap={gap_str} | "
                  f"obj={objbst:14.2f} | bound={objbnd:14.2f}")

# MIPStart from EV
tssp_var_dict["x_bess_p"].Start = ev_res["capacity"]["BESS_P_MW"]
tssp_var_dict["x_bess_e"].Start = ev_res["capacity"]["BESS_E_MWh"]
tssp_var_dict["x_elc_p"].Start = ev_res["capacity"]["ELC_P_MW"]
tssp_var_dict["x_h2_tank"].Start = H2_CAP
tssp_var_dict["x_fc_p"].Start = ev_res["capacity"]["FC_P_MW"]

# Pre-initialise placeholders so the pre-solve checkpoint can run safely.
final_bess_p = None
final_bess_e = None
final_elc = None
final_fc = None
ref = {}
comparison = {}
solve_status = None
solve_time = None
final_gap = None
final_obj = None
final_bound = None

print("  Solving 8-scenario TSSP...")
t0 = time.time()
try:
    tssp_model.optimize(log_callback)
    solve_status = tssp_model.Status
    solve_time = time.time() - t0
    final_gap = tssp_model.MIPGap if tssp_model.SolCount > 0 else None
    final_obj = tssp_model.ObjVal if tssp_model.SolCount > 0 else None
    final_bound = tssp_model.ObjBound if tssp_model.SolCount > 0 else None
except Exception as e:
    solve_status = -1
    solve_time = time.time() - t0
    final_gap = None
    final_obj = None
    final_bound = None
    print(f"  [FAIL] Solver exception: {e}")

# Helper for status name
STATUS_NAMES = {
    1: "LOADED", 2: "OPTIMAL", 3: "INFEASIBLE", 4: "INF_OR_UNBD",
    5: "UNBOUNDED", 6: "CUTOFF", 7: "ITERATION_LIMIT", 8: "NODE_LIMIT",
    9: "TIME_LIMIT", 10: "SOLUTION_LIMIT", 11: "INTERRUPTED", 12: "NUMERIC",
    13: "SUBOPTIMAL", 14: "INPROGRESS", 15: "USER_OBJ_LIMIT"
}

def _save_checkpoint(label=""):
    """Save whatever results are available so far."""
    checkpoint = {
        "meta": {
            "H2_Tank_t": 400,
            "carbon_price_cny_t": CARBON_PRICE,
            "n_scenarios_attempted": N_SCENARIOS,
            "time_limit_s": TIME_LIMIT_TEST,
            "mipgap_target": test_solver_params["MIPGap"]
        },
        "ev": {
            "obj": ev_res["objval"] if ev_res else None,
            "gap_pct": ev_res["mipgap"] if ev_res else None,
            "time_s": ev_time,
            "capacity": ev_res["capacity"] if ev_res else None
        },
        "model_size": {
            "n_variables": n_vars,
            "n_integer_variables": n_intvars,
            "n_binary_variables": n_binvars,
            "n_constraints": n_constrs,
            "n_nonzeros": n_nz
        },
        "solve": {
            "build_time_s": build_time,
            "scenario_gen_time_s": scen_time,
            "solve_time_s": solve_time,
            "status": solve_status,
            "status_name": STATUS_NAMES.get(solve_status, str(solve_status)),
            "final_gap_pct": final_gap * 100 if final_gap is not None else None,
            "final_obj": final_obj,
            "final_bound": final_bound,
            "peak_memory_gb": peak_mem / (1024**3) if peak_mem else None,
            "final_capacity": {
                "BESS_P_MW": final_bess_p,
                "BESS_E_MWh": final_bess_e,
                "ELC_MW": final_elc,
                "FC_MW": final_fc
            }
        },
        "reference_4s": ref,
        "progress_log": progress_log
    }
    cp_path = f"{PARTIAL_PREFIX}_{time.strftime('%Y%m%d_%H%M%S')}_{label}.json"
    with open(cp_path, "w") as f:
        json.dump(checkpoint, f, indent=2, default=float)
    print(f"  [CHECKPOINT] saved: {cp_path}")

# Extract final capacity if incumbent exists
final_bess_p = None
final_bess_e = None
final_elc = None
final_fc = None
if tssp_model.SolCount > 0:
    final_bess_p = tssp_var_dict["x_bess_p"].X
    final_bess_e = tssp_var_dict["x_bess_e"].X
    final_elc = tssp_var_dict["x_elc_p"].X
    final_fc = tssp_var_dict["x_fc_p"].X

print(f"\n  Solve finished: status={solve_status}, time={solve_time:.1f}s")
if final_gap is not None:
    print(f"  Final MIPGap: {final_gap*100:.2f}%")
if final_obj is not None:
    print(f"  Final incumbent: {final_obj:.2f}")

current_mem, peak_mem = tracemalloc.get_traced_memory()
tracemalloc.stop()
print(f"  Peak memory: {peak_mem / (1024**3):.2f} GB")

if tssp_model.SolCount > 0:
    print(f"  Final capacity: BESS_P={final_bess_p:.0f}, BESS_E={final_bess_e:.0f}, "
          f"ELC={final_elc:.0f}, FC={final_fc:.0f}")

# Save intermediate checkpoint before comparison
_save_checkpoint("post_solve")

# ============================================================
# Compare against main 4-scenario reference
# ============================================================
print("[5/4] Comparing against main 4-scenario reference...")
ref_csv = "results/tables/h2_sensitivity_v3_rigorous_vss.csv"
ref = {}
if os.path.exists(ref_csv):
    df_ref = pd.read_csv(ref_csv)
    row = df_ref[df_ref["H2_Tank_t"] == 400].iloc[0]
    ref = {
        "TSSP_BESS_P_MW": row["TSSP_BESS_P_MW"],
        "TSSP_BESS_E_MWh": row["TSSP_BESS_E_MWh"],
        "TSSP_ELC_MW": row["TSSP_ELC_MW"],
        "TSSP_FC_MW": row["TSSP_FC_MW"],
        "TSSP_Obj": row["TSSP_Obj"],
        "TSSP_Gap_pct": row["TSSP_Gap_pct"]
    }
    print(f"  4-scenario ref: BESS_P={ref['TSSP_BESS_P_MW']:.0f}, "
          f"BESS_E={ref['TSSP_BESS_E_MWh']:.0f}, Obj={ref['TSSP_Obj']:.2f}")
else:
    print(f"  Warning: reference CSV not found at {ref_csv}")

comparison = {}
if ref and final_obj is not None:
    comparison = {
        "delta_BESS_P_MW": final_bess_p - ref["TSSP_BESS_P_MW"] if final_bess_p is not None else None,
        "delta_BESS_E_MWh": final_bess_e - ref["TSSP_BESS_E_MWh"] if final_bess_e is not None else None,
        "delta_ELC_MW": final_elc - ref["TSSP_ELC_MW"] if final_elc is not None else None,
        "delta_FC_MW": final_fc - ref["TSSP_FC_MW"] if final_fc is not None else None,
        "delta_Obj": final_obj - ref["TSSP_Obj"],
        "pct_BESS_P": (final_bess_p - ref["TSSP_BESS_P_MW"]) / ref["TSSP_BESS_P_MW"] * 100 if final_bess_p is not None else None,
    }
    print(f"  Difference (8s - 4s): BESS_P={comparison.get('delta_BESS_P_MW'):.1f} MW, "
          f"BESS_E={comparison.get('delta_BESS_E_MWh'):.1f} MWh, "
          f"Obj={comparison.get('delta_Obj'):.2f}")

# ============================================================
# Save results
# ============================================================
result = {
    "meta": {
        "H2_Tank_t": 400,
        "carbon_price_cny_t": CARBON_PRICE,
        "n_scenarios_attempted": N_SCENARIOS,
        "time_limit_s": TIME_LIMIT_TEST,
        "mipgap_target": test_solver_params["MIPGap"]
    },
    "ev": {
        "obj": ev_res["objval"],
        "gap_pct": ev_res["mipgap"],
        "time_s": ev_time,
        "capacity": ev_res["capacity"]
    },
    "model_size": {
        "n_variables": n_vars,
        "n_integer_variables": n_intvars,
        "n_binary_variables": n_binvars,
        "n_constraints": n_constrs,
        "n_nonzeros": n_nz
    },
    "solve": {
        "build_time_s": build_time,
        "scenario_gen_time_s": scen_time,
        "solve_time_s": solve_time,
        "status": solve_status,
        "status_name": STATUS_NAMES.get(solve_status, str(solve_status)),
        "final_gap_pct": final_gap * 100 if final_gap is not None else None,
        "final_obj": final_obj,
        "final_bound": final_bound,
        "peak_memory_gb": peak_mem / (1024**3),
        "final_capacity": {
            "BESS_P_MW": final_bess_p,
            "BESS_E_MWh": final_bess_e,
            "ELC_MW": final_elc,
            "FC_MW": final_fc
        }
    },
    "comparison_4s_vs_8s": comparison,
    "reference_4s": ref,
    "progress_log": progress_log
}

with open(OUTPUT_JSON, "w") as f:
    json.dump(result, f, indent=2, default=float)

# Flatten for CSV
flat = {
    "n_scenarios": N_SCENARIOS,
    "time_limit_s": TIME_LIMIT_TEST,
    "n_variables": n_vars,
    "n_binary_variables": n_binvars,
    "n_constraints": n_constrs,
    "build_time_s": build_time,
    "solve_time_s": solve_time,
    "final_status": solve_status,
    "final_gap_pct": final_gap * 100 if final_gap is not None else None,
    "final_obj": final_obj,
    "peak_memory_gb": peak_mem / (1024**3),
    "final_BESS_P_MW": final_bess_p,
    "final_BESS_E_MWh": final_bess_e,
    "delta_BESS_P_MW_8s_minus_4s": comparison.get("delta_BESS_P_MW"),
    "delta_BESS_E_MWh_8s_minus_4s": comparison.get("delta_BESS_E_MWh"),
    "delta_Obj_8s_minus_4s": comparison.get("delta_Obj"),
}
pd.DataFrame([flat]).to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

print(f"\nSaved: {OUTPUT_JSON}")

_save_checkpoint("final")
print(f"Saved: {OUTPUT_CSV}")
