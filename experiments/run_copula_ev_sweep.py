"""Fast deterministic EV sweep for Copula correlation sensitivity.

Used to populate Table S9 when full TSSP runs are still pending.
"""
import os
import sys
import time

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)
sys.path.insert(0, os.path.join(_project_root, "src"))
os.chdir(_project_root)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "4"

import numpy as np
import pandas as pd

from config import (
    EconParams, PhysParams, ScenarioParams,
    DataPaths, build_load_profile, RepDayParams
)
from src.representative_days import run_representative_day_pipeline
from src.scenario_generator import generate_reduced_scenarios
from src.deterministic_model import build_deterministic_model

H2_CAPACITIES = [200_000, 400_000, 1_000_000]
RHO_VALUES = [-0.20, -0.41]
CARBON_PRICE = 80.0
CARBON_PRICE_MODEL = CARBON_PRICE / 10000.0

OUTPUT_CSV = "results/tables/copula_sensitivity_ev.csv"


def compute_sse(df_rho, bess_col="EV_BESS_P_MW"):
    df_rho = df_rho.sort_values("H2_Tank_t").reset_index(drop=True)
    sses = []
    for i, row in df_rho.iterrows():
        if i == 0:
            sses.append(None)
        else:
            prev = df_rho.iloc[i - 1]
            x0, x1 = prev["H2_Tank_t"], row["H2_Tank_t"]
            y0, y1 = prev[bess_col], row[bess_col]
            if x1 != x0 and (y0 + y1) > 0:
                sses.append((y1 - y0) / (x1 - x0) * (x0 + x1) / (y0 + y1))
            else:
                sses.append(None)
    return sses


def main():
    print("=" * 70)
    print("Copula correlation sensitivity: deterministic EV sweep")
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

    solver = dict(PhysParams.get("SolverParams", {})) if False else {"MIPGap": 0.02, "TimeLimit": 300, "Threads": 4, "MIPFocus": 1}
    from config import SolverParams
    solver = dict(SolverParams)
    solver["Threads"] = 4
    solver["TimeLimit"] = 300

    results = []
    for rho in RHO_VALUES:
        for cap_h2 in H2_CAPACITIES:
            cap_t = int(cap_h2 / 1000)
            print(f"\nrho={rho}, H2={cap_t} t")
            wind_sc, solar_sc, weights_sc = generate_reduced_scenarios(
                wind_r, wind_sigma_r, solar_r, solar_sigma_r,
                n_sample=ScenarioParams["N_sample"],
                n_scenario=ScenarioParams["N_scenario"],
                seed=ScenarioParams["seed"],
                rho_override=rho
            )
            phys_mod = dict(PhysParams)
            phys_mod["Cap_H2_Tank"] = cap_h2
            phys_mod["Cap_H2_Tank_Max"] = cap_h2
            phys_mod["T"] = len(load_r)
            econ_base = dict(EconParams)
            econ_base["Carbon_price"] = CARBON_PRICE_MODEL

            t0 = time.time()
            ev_res, _, _ = build_deterministic_model(load_r, wind_r, solar_r, econ_base, phys_mod, solver)
            ev_time = time.time() - t0
            print(f"  EV Obj={ev_res['objval']:.2f}, Gap={ev_res['mipgap']:.4f}%, Time={ev_time:.1f}s, BESS_P={ev_res['capacity']['BESS_P_MW']:.0f}")
            results.append({
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
            })

    df = pd.DataFrame(results)
    for rho in RHO_VALUES:
        mask = df["rho"] == rho
        df.loc[mask, "EV_SSE"] = compute_sse(df[mask])

    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\nTotal time: {time.time()-t0_total:.1f}s")
    print(f"Saved: {OUTPUT_CSV}")
    print(df[["rho", "H2_Tank_t", "EV_BESS_P_MW", "EV_SSE"]].to_string(index=False))


if __name__ == "__main__":
    main()
