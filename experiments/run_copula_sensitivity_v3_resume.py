"""
Resume the Copula correlation sensitivity experiment from any existing partial
results, run the remaining (rho, H2 tank) combinations, and write the final CSV.

This wrapper avoids repeating already-completed points and writes a checkpoint
after every successful solve so that an interrupted run can be resumed.
"""
import os
import sys
import time
import glob
import json
import threading

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)
sys.path.insert(0, os.path.join(_project_root, "src"))
os.chdir(_project_root)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "2"

# Keep the background worker heartbeat alive during long Gurobi solves
_heartbeat_stop = threading.Event()
def _heartbeat_printer():
    while not _heartbeat_stop.wait(60):
        print(f"[heartbeat] {time.strftime('%Y-%m-%d %H:%M:%S')} still running...", flush=True)
threading.Thread(target=_heartbeat_printer, daemon=True).start()

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

H2_CAPACITIES = [200_000, 400_000, 1_000_000]
RHO_VALUES = [-0.20, -0.41]
CARBON_PRICE = 80.0
CARBON_PRICE_MODEL = CARBON_PRICE / 10000.0

OUTPUT_CSV = "results/tables/copula_sensitivity_v3.csv"
OUTPUT_JSON = "results/tables/copula_sensitivity_v3.json"
PARTIAL_PREFIX = "results/tables/copula_sensitivity_v3_partial"

def load_existing_results():
    """Load any previously saved partial or final results, keeping the
    lowest-gap incumbent when duplicate (rho, H2) pairs exist."""
    results = {}
    for pattern in [OUTPUT_CSV, f"{PARTIAL_PREFIX}_*.csv"]:
        for path in glob.glob(pattern):
            try:
                df = pd.read_csv(path)
                print(f"[resume] Loaded {len(df)} row(s) from {path}")
                for _, row in df.iterrows():
                    key = (round(float(row["rho"]), 6), int(row["H2_Tank_kg"]))
                    cand = row.to_dict()
                    if key not in results:
                        results[key] = cand
                    else:
                        # Prefer the more-converged (lower gap) incumbent
                        prev_gap = float(results[key].get("TSSP_Gap_pct", 999))
                        new_gap = float(cand.get("TSSP_Gap_pct", 999))
                        if new_gap < prev_gap:
                            results[key] = cand
            except Exception as e:
                print(f"[resume] Could not read {path}: {e}")
    return list(results.values())


def already_done(results, rho, cap_h2):
    return any(
        abs(r["rho"] - rho) < 1e-6 and int(r["H2_Tank_kg"]) == int(cap_h2)
        for r in results
    )


def compute_sse(df_rho):
    df_rho = df_rho.sort_values("H2_Tank_t").reset_index(drop=True)
    sses = []
    for i, row in df_rho.iterrows():
        if i == 0:
            sses.append(None)
        else:
            prev = df_rho.iloc[i - 1]
            x0, x1 = prev["H2_Tank_t"], row["H2_Tank_t"]
            y0, y1 = prev["TSSP_BESS_P_MW"], row["TSSP_BESS_P_MW"]
            if x1 != x0 and (y0 + y1) > 0:
                sses.append((y1 - y0) / (x1 - x0) * (x0 + x1) / (y0 + y1))
            else:
                sses.append(None)
    return sses


def main():
    print("=" * 70)
    print("Copula correlation sensitivity experiment (v3 resume)")
    print("=" * 70)
    print(f"H2 capacities (t): {[c/1000 for c in H2_CAPACITIES]}")
    print(f"Gaussian Copula rho values: {RHO_VALUES}")
    print(f"Carbon price: {CARBON_PRICE} CNY/t")
    print(f"Solver: MIPGap={SolverParams['MIPGap']}, TimeLimit={SolverParams['TimeLimit']}s")
    print()

    results = load_existing_results()
    print(f"[resume] {len(results)} point(s) already completed")

    # Data preparation (shared)
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

    remaining = [
        (rho, cap_h2)
        for rho in RHO_VALUES
        for cap_h2 in H2_CAPACITIES
        if not already_done(results, rho, cap_h2)
    ]
    print(f"[resume] Remaining points to solve: {len(remaining)}")

    for rho, cap_h2 in remaining:
        cap_t = int(cap_h2 / 1000)
        print(f"\n{'='*70}")
        print(f"rho = {rho}, H2 tank = {cap_t} t")
        print(f"{'='*70}")

        # Scenarios for this rho
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

        phys_mod = dict(PhysParams)
        phys_mod["Cap_H2_Tank"] = cap_h2
        phys_mod["Cap_H2_Tank_Max"] = cap_h2
        phys_mod["T"] = len(load_r)

        econ_base = dict(EconParams)
        econ_base["Carbon_price"] = CARBON_PRICE_MODEL

        res_point = {
            "rho": rho,
            "H2_Tank_t": cap_t,
            "H2_Tank_kg": cap_h2,
        }

        # EV
        t0 = time.time()
        print("  [EV] solving...")
        ev_res, _, _ = build_deterministic_model(load_r, wind_r, solar_r, econ_base, phys_mod, SolverParams)
        ev_time = time.time() - t0
        if not ev_res:
            print(f"  [FAIL] EV FAILED for rho={rho}, H2={cap_t}t")
            continue
        print(f"  [OK] EV: Obj={ev_res['objval']:.2f}, Gap={ev_res['mipgap']:.4f}%, Time={ev_time:.1f}s")
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
        print("  [TSSP] solving...")
        tssp_model, tssp_var_dict = build_two_stage_model(
            load_r, wind_sc, solar_sc, weights_sc, econ_base, phys_mod, SolverParams
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
            print(f"  [FAIL] TSSP FAILED for rho={rho}, H2={cap_t}t")
            continue
        print(f"  [OK] TSSP: Obj={tssp_res['objval']:.2f}, Gap={tssp_res['mipgap']:.4f}%, Time={tssp_time:.1f}s")
        print(f"     Cap: BESS_P={tssp_res['capacity']['BESS_P_MW']:.0f}, "
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

        results.append(res_point)

        # Save partial checkpoint after every point
        partial_path = f"{PARTIAL_PREFIX}_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        pd.DataFrame(results).to_csv(partial_path, index=False, encoding="utf-8-sig")
        print(f"  [checkpoint] Saved {partial_path}")

    # Final output
    print(f"\n{'='*70}")
    print("Copula sensitivity experiment completed / resumed")
    print(f"{'='*70}")
    df = pd.DataFrame(results)
    for rho in RHO_VALUES:
        mask = df["rho"] == rho
        df.loc[mask, "SSE"] = compute_sse(df[mask])

    print("\n[SUMMARY]")
    print(df[["rho", "H2_Tank_t", "TSSP_BESS_P_MW", "TSSP_BESS_E_MWh", "TSSP_Gap_pct", "SSE"]].to_string(index=False))

    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved final: {OUTPUT_CSV}")
    print(f"Saved final: {OUTPUT_JSON}")
    _heartbeat_stop.set()


if __name__ == "__main__":
    main()
