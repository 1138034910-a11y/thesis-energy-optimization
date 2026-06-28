"""Run a single (rho, H2 tank) Copula-sensitivity point and write a partial CSV.

Designed for short background tasks (<=1.5 h) so the job finishes before the
background worker timeout.
"""
import os
import sys
import time
import argparse
import threading

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


def _heartbeat(stop_event):
    while not stop_event.wait(30):
        print(f"[heartbeat] {time.strftime('%Y-%m-%d %H:%M:%S')} still running...", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rho", type=float, required=True)
    parser.add_argument("--h2", type=int, required=True, help="H2 tank capacity in tonnes")
    parser.add_argument("--timelimit", type=int, default=240, help="TSSP time limit in seconds")
    args = parser.parse_args()

    stop_event = threading.Event()
    threading.Thread(target=_heartbeat, args=(stop_event,), daemon=True).start()
    print(f"[heartbeat] startup at {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

    rho = args.rho
    cap_h2 = args.h2 * 1000
    cap_t = args.h2
    carbon_price = 80.0
    carbon_model = carbon_price / 10000.0

    solver = dict(SolverParams)
    solver["TimeLimit"] = args.timelimit
    solver["Threads"] = 4
    solver["OutputFlag"] = 0

    print("=" * 70)
    print(f"Copula point: rho={rho}, H2={cap_t} t, TimeLimit={args.timelimit}s")
    print("=" * 70)

    # Data prep
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

    wind_sc, solar_sc, weights_sc = generate_reduced_scenarios(
        wind_r, wind_sigma_r, solar_r, solar_sigma_r,
        n_sample=ScenarioParams["N_sample"],
        n_scenario=ScenarioParams["N_scenario"],
        seed=ScenarioParams["seed"],
        rho_override=rho
    )
    print(f"Data prep + scenarios: {time.time()-t0_total:.1f}s, weights={weights_sc}")

    phys_mod = dict(PhysParams)
    phys_mod["Cap_H2_Tank"] = cap_h2
    phys_mod["Cap_H2_Tank_Max"] = cap_h2
    phys_mod["T"] = len(load_r)

    econ_base = dict(EconParams)
    econ_base["Carbon_price"] = carbon_model

    res_point = {
        "rho": rho,
        "H2_Tank_t": cap_t,
        "H2_Tank_kg": cap_h2,
    }

    # EV
    t0 = time.time()
    print("[EV] solving...")
    ev_res, _, _ = build_deterministic_model(load_r, wind_r, solar_r, econ_base, phys_mod, solver)
    ev_time = time.time() - t0
    if not ev_res:
        print(f"[FAIL] EV FAILED")
        stop_event.set()
        return
    print(f"[OK] EV: Obj={ev_res['objval']:.2f}, Gap={ev_res['mipgap']:.4f}%, Time={ev_time:.1f}s")
    res_point.update({
        "EV_Obj": ev_res["objval"],
        "EV_Gap_pct": ev_res["mipgap"],
        "EV_Time_s": ev_time,
        "EV_BESS_P_MW": ev_res["capacity"]["BESS_P_MW"],
        "EV_BESS_E_MWh": ev_res["capacity"]["BESS_E_MWh"],
        "EV_ELC_MW": ev_res["capacity"]["ELC_P_MW"],
        "EV_FC_MW": ev_res["capacity"]["FC_P_MW"],
    })

    # TSSP
    t0 = time.time()
    print("[TSSP] solving...")
    tssp_model, tssp_var_dict = build_two_stage_model(
        load_r, wind_sc, solar_sc, weights_sc, econ_base, phys_mod, solver
    )
    tssp_var_dict["x_bess_p"].Start = ev_res["capacity"]["BESS_P_MW"]
    tssp_var_dict["x_bess_e"].Start = ev_res["capacity"]["BESS_E_MWh"]
    tssp_var_dict["x_elc_p"].Start = ev_res["capacity"]["ELC_P_MW"]
    tssp_var_dict["x_h2_tank"].Start = cap_h2
    tssp_var_dict["x_fc_p"].Start = ev_res["capacity"]["FC_P_MW"]

    tssp_res, _ = solve_and_extract(
        tssp_model, tssp_var_dict, load_r, wind_sc, solar_sc, weights_sc, phys_mod
    )
    tssp_time = time.time() - t0
    if not tssp_res:
        print(f"[FAIL] TSSP FAILED")
        stop_event.set()
        return
    print(f"[OK] TSSP: Obj={tssp_res['objval']:.2f}, Gap={tssp_res['mipgap']:.4f}%, Time={tssp_time:.1f}s")
    print(f"   Cap: BESS_P={tssp_res['capacity']['BESS_P_MW']:.0f}, "
          f"BESS_E={tssp_res['capacity']['BESS_E_MWh']:.0f}, "
          f"ELC={tssp_res['capacity']['ELC_P_MW']:.0f}, "
          f"FC={tssp_res['capacity']['FC_P_MW']:.0f}")
    res_point.update({
        "TSSP_Obj": tssp_res["objval"],
        "TSSP_Gap_pct": tssp_res["mipgap"],
        "TSSP_Time_s": tssp_time,
        "TSSP_BESS_P_MW": tssp_res["capacity"]["BESS_P_MW"],
        "TSSP_BESS_E_MWh": tssp_res["capacity"]["BESS_E_MWh"],
        "TSSP_ELC_MW": tssp_res["capacity"]["ELC_P_MW"],
        "TSSP_FC_MW": tssp_res["capacity"]["FC_P_MW"],
    })

    out_path = f"results/tables/copula_sensitivity_v3_point_rho{rho:.2f}_h2{cap_t}.csv"
    pd.DataFrame([res_point]).to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[saved] {out_path}")
    stop_event.set()


if __name__ == "__main__":
    main()
