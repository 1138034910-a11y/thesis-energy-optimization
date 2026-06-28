"""Deterministic EV solve for a single (rho, H2) point; appends to CSV."""
import os
import sys
import time
import argparse

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)
sys.path.insert(0, os.path.join(_project_root, "src"))
os.chdir(_project_root)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "4"

import numpy as np
import pandas as pd

from config import (
    EconParams, PhysParams, ScenarioParams, SolverParams,
    DataPaths, build_load_profile, RepDayParams
)
from src.representative_days import run_representative_day_pipeline
from src.scenario_generator import generate_reduced_scenarios
from src.deterministic_model import build_deterministic_model

CARBON_PRICE = 80.0
CARBON_PRICE_MODEL = CARBON_PRICE / 10000.0
OUTPUT_CSV = "results/tables/copula_sensitivity_ev.csv"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rho", type=float, required=True)
    parser.add_argument("--h2", type=int, required=True, help="H2 tank in tonnes")
    parser.add_argument("--timelimit", type=int, default=180, help="EV time limit in seconds")
    args = parser.parse_args()
    rho = args.rho
    cap_h2 = args.h2 * 1000
    cap_t = args.h2

    print(f"=== EV point: rho={rho}, H2={cap_t}t ===")

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

    solver = dict(SolverParams)
    solver["Threads"] = 4
    solver["TimeLimit"] = args.timelimit

    phys_mod = dict(PhysParams)
    phys_mod["Cap_H2_Tank"] = cap_h2
    phys_mod["Cap_H2_Tank_Max"] = cap_h2
    phys_mod["T"] = len(load_r)
    econ_base = dict(EconParams)
    econ_base["Carbon_price"] = CARBON_PRICE_MODEL

    t0 = time.time()
    ev_res, _, _ = build_deterministic_model(load_r, wind_r, solar_r, econ_base, phys_mod, solver)
    ev_time = time.time() - t0
    print(f"Obj={ev_res['objval']:.2f}, Gap={ev_res['mipgap']:.4f}%, Time={ev_time:.1f}s, BESS_P={ev_res['capacity']['BESS_P_MW']:.0f}")

    row = {
        "rho": rho,
        "H2_Tank_t": cap_t,
        "H2_Tank_kg": cap_h2,
        "EV_Obj": ev_res["objval"],
        "EV_Gap_pct": ev_res["mipgap"],
        "EV_Time_s": ev_time,
        "EV_BESS_P_MW": ev_res["capacity"]["BESS_P_MW"],
        "EV_BESS_E_MWh": ev_res["capacity"]["BESS_E_MWh"],
        "EV_ELC_MW": ev_res["capacity"]["ELC_P_MW"],
        "EV_FC_MW": ev_res["capacity"]["FC_P_MW"],
    }

    if os.path.exists(OUTPUT_CSV):
        df = pd.read_csv(OUTPUT_CSV)
        df = df[(df["rho"] != rho) | (df["H2_Tank_t"] != cap_t)]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df = df.sort_values(["rho", "H2_Tank_t"]).reset_index(drop=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"Saved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
