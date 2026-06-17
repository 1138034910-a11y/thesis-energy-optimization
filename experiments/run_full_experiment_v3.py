"""
Full representative-day experiment v3 — authoritative 400t base case.

Runs: EV + TSSP (with op-vars export) + rigorous VSS (EEV with MIPStart)
       + optional carbon-cap sweep (TSSP) + JSON export for all figures archive.

Output: results/tables/full_experiment_v3.json
        results/tables/carbon_sweep_tssp.csv   (if RUN_CARBON_SWEEP=1)

This script uses the EXACT same data pipeline as run_h2_sensitivity_v3_FIXED.py
to ensure consistency with h2_sensitivity_v3_rigorous_vss.csv.
"""
import os, sys, time, json, argparse

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
    DataPaths, build_load_profile, RepDayParams,
)
from src.representative_days import run_representative_day_pipeline
from src.scenario_generator import generate_reduced_scenarios
from src.deterministic_model import build_deterministic_model
from src.stochastic_model import build_two_stage_model, solve_and_extract, build_eev_model

# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--carbon-sweep", action="store_true",
                    help="Also run TSSP carbon-cap sweep (6 points, ~6-9h)")
parser.add_argument("--skip-tssp", action="store_true",
                    help="Skip 400t TSSP re-solve (use CSV values instead)")
args = parser.parse_args()

H2_CAP = 400_000
OUT_JSON = "results/tables/full_experiment_v3.json"
OUT_CARBON = "results/tables/carbon_sweep_tssp.csv"

os.makedirs("results/tables", exist_ok=True)

print("=" * 70)
print("Full Experiment v3 — Authoritative 400t Base Case")
print("=" * 70)
print(f"H2 tank = {H2_CAP/1000:.0f} t")
print(f"Carbon sweep (TSSP) = {'YES' if args.carbon_sweep else 'NO'}")
print(f"Skip 400t TSSP = {'YES' if args.skip_tssp else 'NO'}")
print()

# ------------------------------------------------------------------------------
# 1. Load data (identical to h2_sensitivity_v3)
# ------------------------------------------------------------------------------
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

# BUG-01 fix: EV input uses KAN-predicted mu
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

wind_sc, solar_sc, weights_sc = generate_reduced_scenarios(
    wind_r, wind_sigma_r, solar_r, solar_sigma_r,
    n_sample=ScenarioParams["N_sample"], n_scenario=ScenarioParams["N_scenario"],
    seed=ScenarioParams["seed"], rho_override=-0.30
)

S, T = len(weights_sc), len(load_r)
print(f"[Setup] Scenarios={S}, Hours={T}, Time={time.time()-t0_total:.1f}s\n")

# ------------------------------------------------------------------------------
# 2. 400t EV
# ------------------------------------------------------------------------------
phys_mod = dict(PhysParams)
phys_mod["Cap_H2_Tank"] = H2_CAP
phys_mod["Cap_H2_Tank_Max"] = H2_CAP
phys_mod["T"] = T
econ_base = dict(EconParams)
econ_base["Carbon_price"] = 0.0  # Base case: zero carbon price

print("[1/4] 400t EV (deterministic)...")
t0 = time.time()
ev_res, ev_status, ev_op_vars = build_deterministic_model(load_r, wind_r, solar_r, econ_base, phys_mod, SolverParams)
ev_time = time.time() - t0

if not ev_res:
    print("  [FAIL] EV FAILED")
    sys.exit(1)

print(f"  [OK] EV: Obj={ev_res['objval']:.2f}, Gap={ev_res['mipgap']:.4f}%, Time={ev_time:.1f}s")
print(f"     Cap: BESS_P={ev_res['capacity']['BESS_P_MW']:.0f}, "
      f"BESS_E={ev_res['capacity']['BESS_E_MWh']:.0f}, "
      f"ELC={ev_res['capacity']['ELC_P_MW']:.0f}, "
      f"FC={ev_res['capacity']['FC_P_MW']:.0f}")

# ------------------------------------------------------------------------------
# 3. 400t TSSP
# ------------------------------------------------------------------------------
tssp_res = None
tssp_time = 0.0

if args.skip_tssp:
    # Load from authoritative CSV
    print("\n[2/4] 400t TSSP — loading from h2_sensitivity_v3_rigorous_vss.csv...")
    csv = pd.read_csv("results/tables/h2_sensitivity_v3_rigorous_vss.csv")
    row = csv[csv["H2_Tank_t"] == 400].iloc[0]
    tssp_res = {
        "objval": row["TSSP_Obj"],
        "mipgap": row["TSSP_Gap_pct"] / 100.0,
        "capacity": {
            "BESS_P_MW": row["TSSP_BESS_P_MW"],
            "BESS_E_MWh": row["TSSP_BESS_E_MWh"],
            "ELC_P_MW": row["TSSP_ELC_MW"],
            "H2_Tank_kg": 400_000.0,
            "FC_P_MW": row["TSSP_FC_MW"],
        },
        "costs": {},
        "scenarios": [],  # op-vars NOT available from CSV
    }
    print(f"  [WARN]  Loaded from CSV (no operational variables). "
          f"Obj={tssp_res['objval']:.2f}, Gap={tssp_res['mipgap']*100:.2f}%")
    print(f"     Cap: BESS_P={tssp_res['capacity']['BESS_P_MW']:.0f}, "
          f"BESS_E={tssp_res['capacity']['BESS_E_MWh']:.0f}")
else:
    print("\n[2/4] 400t TSSP (two-stage stochastic)...")
    t0 = time.time()
    tssp_model, tssp_var_dict = build_two_stage_model(
        load_r, wind_sc, solar_sc, weights_sc, econ_base, phys_mod, SolverParams
    )
    # Warm start from EV
    tssp_var_dict["x_bess_p"].Start = ev_res["capacity"]["BESS_P_MW"]
    tssp_var_dict["x_bess_e"].Start = ev_res["capacity"]["BESS_E_MWh"]
    tssp_var_dict["x_elc_p"].Start = ev_res["capacity"]["ELC_P_MW"]
    tssp_var_dict["x_h2_tank"].Start = H2_CAP
    tssp_var_dict["x_fc_p"].Start = ev_res["capacity"]["FC_P_MW"]

    tssp_res, tssp_status = solve_and_extract(
        tssp_model, tssp_var_dict, load_r, wind_sc, solar_sc, weights_sc, phys_mod
    )
    tssp_time = time.time() - t0

    if not tssp_res:
        print("  [FAIL] TSSP FAILED")
        sys.exit(1)

    print(f"  [OK] TSSP: Obj={tssp_res['objval']:.2f}, Gap={tssp_res['mipgap']*100:.2f}%, Time={tssp_time:.1f}s")
    print(f"     Cap: BESS_P={tssp_res['capacity']['BESS_P_MW']:.0f}, "
          f"BESS_E={tssp_res['capacity']['BESS_E_MWh']:.0f}, "
          f"ELC={tssp_res['capacity']['ELC_P_MW']:.0f}, "
          f"FC={tssp_res['capacity']['FC_P_MW']:.0f}")

# ------------------------------------------------------------------------------
# 4. Rigorous VSS (EEV)
# ------------------------------------------------------------------------------
print("\n[3/4] Rigorous VSS (EEV with EV-capacity fixation + MIPStart)...")
if not tssp_res:
    print("  Skipped (no TSSP)")
    eev_res = None
    vss_rigorous = None
    vss_pct = None
else:
    fixed_cap = ev_res["capacity"]
    eev_model, eev_var_dict = build_eev_model(
        load_r, wind_sc, solar_sc, weights_sc,
        fixed_cap, econ_base, phys_mod, SolverParams
    )

    # MIPStart: copy EV op-vars to all scenarios
    for s in range(S):
        for t in range(T):
            eev_var_dict["p_therm"][s, t].Start = float(ev_op_vars["p_therm"][t])
            eev_var_dict["u_therm"][s, t].Start = float(ev_op_vars["u_therm"][t])
            eev_var_dict["p_bess_ch"][s, t].Start = float(ev_op_vars["p_bess_ch"][t])
            eev_var_dict["p_bess_dis"][s, t].Start = float(ev_op_vars["p_bess_dis"][t])
            eev_var_dict["e_bess"][s, t].Start = float(ev_op_vars["e_bess"][t])
            eev_var_dict["p_elc"][s, t].Start = float(ev_op_vars["p_elc"][t])
            eev_var_dict["h_prod"][s, t].Start = float(ev_op_vars["h_prod"][t])
            eev_var_dict["h_supply"][s, t].Start = float(ev_op_vars["h_supply"][t])
            eev_var_dict["h_fc_use"][s, t].Start = float(ev_op_vars["h_fc_use"][t])
            eev_var_dict["p_fc"][s, t].Start = float(ev_op_vars["p_fc"][t])
            eev_var_dict["p_uhv"][s, t].Start = float(ev_op_vars["p_uhv"][t])
            eev_var_dict["p_curt"][s, t].Start = float(ev_op_vars["p_curt"][t])
            eev_var_dict["u_elc"][s, t].Start = float(ev_op_vars["u_elc"][t])
            if "y_start" in eev_var_dict:
                eev_var_dict["y_start"][s, t].Start = float(ev_op_vars["y_start"][t])
            if "z_stop" in eev_var_dict:
                eev_var_dict["z_stop"][s, t].Start = float(ev_op_vars["z_stop"][t])
            if "y_elc_start" in eev_var_dict:
                eev_var_dict["y_elc_start"][s, t].Start = float(ev_op_vars["y_elc_start"][t])
            if "z_elc_stop" in eev_var_dict:
                eev_var_dict["z_elc_stop"][s, t].Start = float(ev_op_vars["z_elc_stop"][t])
            if "u_bess_ch" in eev_var_dict:
                eev_var_dict["u_bess_ch"][s, t].Start = float(ev_op_vars["u_bess_ch"][t])
            if "u_fc" in eev_var_dict:
                eev_var_dict["u_fc"][s, t].Start = float(ev_op_vars["u_fc"][t])
            if "x_elc_on" in eev_var_dict:
                eev_var_dict["x_elc_on"][s, t].Start = float(ev_op_vars["x_elc_on"][t])

    t0 = time.time()
    eev_res, eev_status = solve_and_extract(
        eev_model, eev_var_dict, load_r, wind_sc, solar_sc, weights_sc, phys_mod
    )
    eev_time = time.time() - t0

    if eev_res:
        z_eev = eev_res["objval"]
        z_rp = tssp_res["objval"]
        vss_rigorous = z_eev - z_rp
        vss_pct = 100 * vss_rigorous / abs(z_rp) if z_rp != 0 else 0
        print(f"  [OK] EEV: Obj={z_eev:.2f}, Time={eev_time:.1f}s")
        print(f"     VSS = {vss_rigorous:.2f} ({vss_pct:.3f}%)")
    else:
        print("  [FAIL] EEV FAILED")
        vss_rigorous = None
        vss_pct = None

# ------------------------------------------------------------------------------
# 5. Optional: TSSP carbon-cap sweep
# ------------------------------------------------------------------------------
carbon_records = []
if args.carbon_sweep and tssp_res:
    print("\n[4/4] Carbon-cap sweep (TSSP, 6 points)...")
    carbon_caps = [5e6, 10e6, 15e6, 20e6, 25e6, 30e6]  # Mt -> kg
    for cap in carbon_caps:
        print(f"\n  [Carbon] Cap = {cap/1e6:.0f} Mt")
        econ_mod = dict(econ_base)
        econ_mod["Carbon_cap_annual"] = float(cap)
        t0 = time.time()
        cs_model, cs_var_dict = build_two_stage_model(
            load_r, wind_sc, solar_sc, weights_sc, econ_mod, phys_mod, SolverParams
        )
        # Warm start from 400t TSSP
        cs_var_dict["x_bess_p"].Start = tssp_res["capacity"]["BESS_P_MW"]
        cs_var_dict["x_bess_e"].Start = tssp_res["capacity"]["BESS_E_MWh"]
        cs_var_dict["x_elc_p"].Start = tssp_res["capacity"]["ELC_P_MW"]
        cs_var_dict["x_h2_tank"].Start = H2_CAP
        cs_var_dict["x_fc_p"].Start = tssp_res["capacity"]["FC_P_MW"]

        cs_res, cs_status = solve_and_extract(
            cs_model, cs_var_dict, load_r, wind_sc, solar_sc, weights_sc, phys_mod
        )
        rt = time.time() - t0
        if cs_res:
            print(f"    [OK] Obj={cs_res['objval']:.2f}, Gap={cs_res['mipgap']*100:.2f}%, Time={rt:.1f}s")
            print(f"       Cap: BESS={cs_res['capacity']['BESS_P_MW']:.0f}, "
                  f"ELC={cs_res['capacity']['ELC_P_MW']:.0f}, FC={cs_res['capacity']['FC_P_MW']:.0f}")
            carbon_records.append({
                "carbon_cap_Mt": cap / 1e6,
                "objval_M_CNY": cs_res["objval"] / 1e4,
                "bess_MW": cs_res["capacity"]["BESS_P_MW"],
                "bess_MWh": cs_res["capacity"]["BESS_E_MWh"],
                "elc_MW": cs_res["capacity"]["ELC_P_MW"],
                "fc_MW": cs_res["capacity"]["FC_P_MW"],
                "h2_kg": cs_res["capacity"]["H2_Tank_kg"],
                "gap": cs_res["mipgap"],
                "status": "OPTIMAL" if cs_res["mipgap"] <= SolverParams["MIPGap"] else "STATUS_9",
            })
        else:
            print(f"    [FAIL] FAILED")
            carbon_records.append({"carbon_cap_Mt": cap/1e6, "status": "FAILED"})

    df_carbon = pd.DataFrame(carbon_records)
    df_carbon.to_csv(OUT_CARBON, index=False)
    print(f"\n  Carbon sweep saved to {OUT_CARBON}")
else:
    print("\n[4/4] Carbon-cap sweep = SKIPPED")

# ------------------------------------------------------------------------------
# 6. Export JSON
# ------------------------------------------------------------------------------
print("\n" + "=" * 70)
print("Exporting JSON...")
print("=" * 70)

# Serialize numpy arrays to lists for JSON
scenario_list = []
if tssp_res and tssp_res.get("scenarios"):
    for sc in tssp_res["scenarios"]:
        scenario_list.append({k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in sc.items()})

output = {
    "meta": {
        "h2_tank_kg": H2_CAP,
        "h2_tank_t": H2_CAP / 1000,
        "solver_mipgap": SolverParams["MIPGap"],
        "solver_timelimit": SolverParams["TimeLimit"],
        "n_scenarios": S,
        "n_hours": T,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    },
    "ev": {
        "status": ev_status,
        "objval": ev_res["objval"],
        "mipgap": ev_res["mipgap"],
        "capacity": ev_res["capacity"],
        "costs": ev_res.get("costs", {}),
    },
    "tssp": {
        "status": tssp_status if not args.skip_tssp else "LOADED_FROM_CSV",
        "objval": tssp_res["objval"],
        "mipgap": tssp_res["mipgap"],
        "runtime": tssp_time,
        "capacity": tssp_res["capacity"],
        "costs": tssp_res.get("costs", {}),
        "scenarios": scenario_list,
    },
    "vss_rigorous": {
        "vss": vss_rigorous,
        "vss_pct": vss_pct,
        "z_eev": eev_res["objval"] if eev_res else None,
        "z_rp": tssp_res["objval"],
    } if vss_rigorous is not None else None,
    "carbon_sweep": carbon_records,
}

with open(OUT_JSON, "w") as f:
    json.dump(output, f, indent=2, default=str)

print(f"\n[OK] Saved: {OUT_JSON}")
print(f"   EV  Obj = {ev_res['objval']:.2f}")
print(f"   TSSP Obj = {tssp_res['objval']:.2f}")
if vss_rigorous is not None:
    print(f"   VSS = {vss_rigorous:.2f} ({vss_pct:.3f}%)")
print(f"   Total time = {time.time()-t0_total:.1f}s")
print("=" * 70)
