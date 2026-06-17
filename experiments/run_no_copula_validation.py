"""
No-Copula validation: SSE sign-pattern robustness under independent sampling

Purpose:
  Verify that the three-stage SSE sign reversal (complementarity -> substitution
  -> strong substitution) is robust to the choice of dependence structure. The
  baseline experiment uses a Gaussian Copula (Kendall tau ~ -0.20) to preserve
  empirical wind-solar rank correlation. This script reruns the H2 sensitivity
  sweep with independent sampling (use_copula=False) at three key grid points.

Scope:
  Three H2 tank capacities: 200, 400, 1000 t (covering the full range from
  complementarity through substitution to strong substitution). EV, TSSP, and
  EEV are computed at each point to enable VSS comparison.

Expected duration:
  EV:  < 1 min per point
  TSSP: 1-4 h per point   (total ~8 h)
  EEV:  1-4 h per point   (total ~8 h)
  Total wall-clock: ~10-12 h (recommend overnight run)

Output:
  results/tables/no_copula_validation.json
  results/tables/no_copula_validation.csv

Usage:
  python experiments/run_no_copula_validation.py
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd

# ---- Project root and path setup ----
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)
sys.path.insert(0, os.path.join(_project_root, "src"))
os.chdir(_project_root)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "2"

from config import (
    EconParams, PhysParams, ScenarioParams, SolverParams,
    build_load_profile, RepDayParams
)
from src.representative_days import run_representative_day_pipeline
from src.scenario_generator import generate_reduced_scenarios
from src.deterministic_model import build_deterministic_model
from src.stochastic_model import build_two_stage_model, solve_and_extract, build_eev_model

# ---- Experiment configuration ----
# Three key grid points spanning the SSE sign reversal
H2_CAPACITIES = [200_000, 400_000, 1_000_000]

OUTPUT_JSON = "results/tables/no_copula_validation.json"
OUTPUT_CSV  = "results/tables/no_copula_validation.csv"

# Baseline results (with Copula) for comparison
BASELINE_CSV = "results/tables/h2_sensitivity_v3_rigorous_vss.csv"


def print_header():
    print("=" * 70)
    print("No-Copula Validation: SSE Sign-Pattern Robustness")
    print("=" * 70)
    print(f"  H2 capacities: {[f'{c // 1000}t' for c in H2_CAPACITIES]}")
    print(f"  Dependence:    independent sampling (use_copula=False)")
    print(f"  Solver:        MIPGap={SolverParams['MIPGap']}, TimeLimit={SolverParams['TimeLimit']}s")
    print(f"  Seed:          RepDays={RepDayParams['seed']}, Scenarios={ScenarioParams['seed']}")
    print()


def load_baseline():
    """Load Copula-based results for side-by-side comparison."""
    if not os.path.exists(BASELINE_CSV):
        print(f"  [WARNING] Baseline file not found: {BASELINE_CSV}")
        print(f"            Cross-comparison will be skipped.")
        return {}
    df = pd.read_csv(BASELINE_CSV)
    baseline = {}
    for _, row in df.iterrows():
        baseline[int(row["H2_Tank_t"])] = row.to_dict()
    print(f"  Loaded {len(baseline)} baseline (Copula) reference points.")
    return baseline


def prepare_data():
    """Load and prepare data: KAN forecasts, representative days, scenarios."""
    print("\n[Step 1] Loading data and generating scenarios (independent sampling)...")
    t0 = time.time()

    # 1.1 KAN forecasts
    kan_df = pd.read_csv("results/tables/kan_forecasts.csv")
    wind_mu  = kan_df["wind_mu"].bfill().values
    solar_mu = kan_df["solar_mu"].bfill().values

    # 1.2 EV input uses KAN-mu (BUG-01 fix)
    wind_actual  = wind_mu.copy()
    solar_actual = solar_mu.copy()

    # 1.3 Truncate to 8760 h
    n_len = min(len(wind_mu), 8760)
    wind_mu   = wind_mu[:n_len]
    solar_mu  = solar_mu[:n_len]
    wind_actual  = wind_actual[:n_len]
    solar_actual = solar_actual[:n_len]

    # 1.4 Load profile
    load_full = build_load_profile(8760, PhysParams["Load_Base"])

    # 1.5 Representative days (same seed as baseline)
    reps, wind_r, solar_r, load_r, weights_days = run_representative_day_pipeline(
        wind_actual, solar_actual, load_full,
        n_days=RepDayParams["n_days"], seed=RepDayParams["seed"]
    )

    # 1.6 Extract sigma for representative days
    wind_sigma_full  = kan_df["wind_sigma"].values[:n_len]
    solar_sigma_full = kan_df["solar_sigma"].values[:n_len]
    wind_sigma_r  = []
    solar_sigma_r = []
    for d in reps["day_indices"]:
        s = d * 24
        e = s + 24
        wind_sigma_r.extend(wind_sigma_full[s:e])
        solar_sigma_r.extend(solar_sigma_full[s:e])
    wind_sigma_r  = np.array(wind_sigma_r)
    solar_sigma_r = np.array(solar_sigma_r)

    # 1.7 Scenario generation: INDEPENDENT SAMPLING (no Copula)
    wind_sc, solar_sc, weights_sc = generate_reduced_scenarios(
        wind_r, wind_sigma_r, solar_r, solar_sigma_r,
        n_sample=ScenarioParams["N_sample"],
        n_scenario=ScenarioParams["N_scenario"],
        seed=ScenarioParams["seed"],
        use_copula=False            # <-- KEY CHANGE: independent sampling
    )

    elapsed = time.time() - t0
    print(f"  Data preparation complete: {elapsed:.1f}s")
    print(f"  Representative days: {RepDayParams['n_days']}, "
          f"scenarios: {len(weights_sc)}, "
          f"scenario weights: {np.round(weights_sc, 3)}")
    return load_r, wind_r, solar_r, wind_sc, solar_sc, weights_sc


def compute_sse(bess_200, bess_400, bess_1000):
    """Compute arc elasticities for two intervals from three grid points."""
    def arc_elast(y1, y2, x1, x2):
        y_bar = (y1 + y2) / 2.0
        x_bar = (x1 + x2) / 2.0
        if y_bar == 0 or x_bar == 0:
            return float('nan')
        return (y2 - y1) / y_bar / ((x2 - x1) / x_bar)

    sse_200_400 = arc_elast(bess_200, bess_400, 200, 400)
    sse_400_1000 = arc_elast(bess_400, bess_1000, 400, 1000)
    return sse_200_400, sse_400_1000


def run_single_point(cap_h2, load_r, wind_r, solar_r, wind_sc, solar_sc, weights_sc,
                     baseline, econ_base, phys_base):
    """Run EV -> TSSP -> EEV for one H2 tank capacity."""
    cap_t = int(cap_h2 / 1000)
    print(f"\n{'=' * 70}")
    print(f"H2 Tank = {cap_t} t (independent sampling)")
    print(f"{'=' * 70}")

    # Configure physical parameters for this H2 capacity
    phys_mod = dict(phys_base)
    phys_mod["Cap_H2_Tank"]     = cap_h2
    phys_mod["Cap_H2_Tank_Max"] = cap_h2
    phys_mod["T"] = len(load_r)

    result = {"H2_Tank_t": cap_t, "H2_Tank_kg": cap_h2}

    # ---- EV ----
    t0 = time.time()
    ev_res, _, _ = build_deterministic_model(
        load_r, wind_r, solar_r, econ_base, phys_mod, SolverParams
    )
    ev_time = time.time() - t0
    if not ev_res:
        print(f"  [FAIL] EV did not return a feasible solution.")
        return None

    ev_bess_p = ev_res["capacity"]["BESS_P_MW"]
    print(f"  EV:  obj={ev_res['objval']:,.0f}, gap={ev_res['mipgap']:.4f}%, "
          f"BESS_P={ev_bess_p:.0f} MW, time={ev_time:.0f}s")

    result["EV_Obj"]       = ev_res["objval"]
    result["EV_Gap_pct"]   = ev_res["mipgap"]
    result["EV_Time_s"]    = ev_time
    result["EV_BESS_P_MW"] = ev_bess_p
    result["EV_BESS_E_MWh"] = ev_res["capacity"]["BESS_E_MWh"]

    # ---- TSSP ----
    t0 = time.time()
    tssp_model, tssp_vars = build_two_stage_model(
        load_r, wind_sc, solar_sc, weights_sc, econ_base, phys_mod, SolverParams
    )
    # Warm start from EV
    tssp_vars["x_bess_p"].Start   = ev_res["capacity"]["BESS_P_MW"]
    tssp_vars["x_bess_e"].Start   = ev_res["capacity"]["BESS_E_MWh"]
    tssp_vars["x_elc_p"].Start    = ev_res["capacity"]["ELC_P_MW"]
    tssp_vars["x_h2_tank"].Start  = cap_h2
    tssp_vars["x_fc_p"].Start     = ev_res["capacity"]["FC_P_MW"]

    tssp_res, _ = solve_and_extract(
        tssp_model, tssp_vars, load_r, wind_sc, solar_sc, weights_sc, phys_mod
    )
    tssp_time = time.time() - t0
    if not tssp_res:
        print(f"  [FAIL] TSSP did not return a feasible solution.")
        return None

    tssp_bess_p = tssp_res["capacity"]["BESS_P_MW"]
    print(f"  TSSP: obj={tssp_res['objval']:,.0f}, gap={tssp_res['mipgap']:.4f}%, "
          f"BESS_P={tssp_bess_p:.0f} MW, time={tssp_time:.0f}s")

    result["TSSP_Obj"]         = tssp_res["objval"]
    result["TSSP_Gap_pct"]     = tssp_res["mipgap"]
    result["TSSP_Time_s"]      = tssp_time
    result["TSSP_BESS_P_MW"]   = tssp_bess_p
    result["TSSP_BESS_E_MWh"]  = tssp_res["capacity"]["BESS_E_MWh"]

    # ---- EEV ----
    t0 = time.time()
    eev_model, eev_vars = build_eev_model(
        load_r, wind_sc, solar_sc, weights_sc,
        ev_res["capacity"], econ_base, phys_mod, SolverParams
    )
    eev_res, _ = solve_and_extract(
        eev_model, eev_vars, load_r, wind_sc, solar_sc, weights_sc, phys_mod
    )
    eev_time = time.time() - t0

    if eev_res:
        z_eev = eev_res["objval"]
        z_rp  = tssp_res["objval"]
        vss   = z_eev - z_rp
        vss_pct = 100.0 * vss / abs(z_rp) if z_rp != 0 else 0.0
        print(f"  EEV: obj={z_eev:,.0f}, VSS={vss:,.0f} ({vss_pct:.2f}%), time={eev_time:.0f}s")
        result["EEV_Obj"]    = z_eev
        result["EEV_Gap_pct"] = eev_res["mipgap"]
        result["EEV_Time_s"] = eev_time
        result["VSS"]        = vss
        result["VSS_pct"]    = vss_pct
    else:
        print(f"  [WARN] EEV did not return a feasible solution.")
        result["EEV_Obj"] = None
        result["VSS"]     = None
        result["VSS_pct"] = None

    # ---- Cross-comparison with baseline (Copula) ----
    if cap_t in baseline:
        ref = baseline[cap_t]
        delta_bess = tssp_bess_p - ref.get("TSSP_BESS_P_MW", float('nan'))
        print(f"  [COMPARE] BESS_P (no-Copula - Copula) = {delta_bess:+.0f} MW")
        result["Baseline_TSSP_BESS_P_MW"] = ref.get("TSSP_BESS_P_MW")
        result["Delta_BESS_P_MW"] = delta_bess

    return result


def print_summary(results, sse_pair, baseline):
    """Print final comparison table and SSE diagnostics."""
    print(f"\n{'=' * 70}")
    print("Validation Summary: Independent vs Copula Sampling")
    print(f"{'=' * 70}")

    # Header
    print(f"{'H2(t)':>6} {'BESS_noCop':>10} {'BESS_Cop':>10} {'Delta':>8} "
          f"{'VSS_noCop':>10} {'VSS_Cop':>10}")
    print("-" * 65)

    for r in results:
        cap  = r["H2_Tank_t"]
        bp   = r.get("TSSP_BESS_P_MW", float('nan'))
        bp_c = r.get("Baseline_TSSP_BESS_P_MW", float('nan'))
        db   = r.get("Delta_BESS_P_MW", float('nan'))
        vss  = r.get("VSS_pct", float('nan'))
        vss_c = baseline.get(cap, {}).get("VSS_pct", float('nan'))
        print(f"{cap:>6} {bp:>10.0f} {bp_c:>10.0f} {db:>+8.0f} "
              f"{vss:>10.4f} {vss_c:>10.4f}")

    # SSE comparison
    sse_cop = sse_pair.get("copula", {})
    sse_ind = sse_pair.get("independent", {})
    print(f"\n  SSE (200-400 t):  Copula={sse_cop.get('sse_200_400', '?'):>8}  "
          f"Independent={sse_ind.get('sse_200_400', '?'):>8}")
    print(f"  SSE (400-1000 t): Copula={sse_cop.get('sse_400_1000', '?'):>8}  "
          f"Independent={sse_ind.get('sse_400_1000', '?'):>8}")

    # Sign-pattern check
    sign_ind_200 = sse_ind.get("sign_200_400")
    sign_cop_200 = sse_cop.get("sign_200_400")
    sign_ind_400 = sse_ind.get("sign_400_1000")
    sign_cop_400 = sse_cop.get("sign_400_1000")

    if sign_ind_200 == sign_cop_200 and sign_ind_400 == sign_cop_400:
        print(f"\n  >> SIGN PATTERN PRESERVED: Independent sampling confirms")
        print(f"     the same complementarity-to-substitution transition.")
    else:
        print(f"\n  >> SIGN PATTERN DIFFERS: See discussion in Supplementary S3.")


def main():
    print_header()

    # ---- Load baseline (Copula) for comparison ----
    baseline = load_baseline()

    # ---- Prepare data with independent sampling ----
    load_r, wind_r, solar_r, wind_sc, solar_sc, weights_sc = prepare_data()

    phys_base = dict(PhysParams)
    econ_base = dict(EconParams)

    # ---- Run EV -> TSSP -> EEV for each H2 capacity ----
    results = []
    t_start = time.time()

    for cap_h2 in H2_CAPACITIES:
        res = run_single_point(
            cap_h2, load_r, wind_r, solar_r,
            wind_sc, solar_sc, weights_sc,
            baseline, econ_base, phys_base
        )
        if res:
            results.append(res)

    wall_clock = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"All runs complete. Wall-clock: {wall_clock:.0f}s ({wall_clock/3600:.1f}h)")
    print(f"{'=' * 70}")

    # ---- Compute SSE for both sampling methods ----
    sse_pair = {}

    # Independent sampling SSE
    ind_map = {r["H2_Tank_t"]: r["TSSP_BESS_P_MW"] for r in results if "TSSP_BESS_P_MW" in r}
    if all(k in ind_map for k in [200, 400, 1000]):
        e200, e1000 = compute_sse(ind_map[200], ind_map[400], ind_map[1000])
        sse_pair["independent"] = {
            "sse_200_400":  round(e200, 4),
            "sse_400_1000": round(e1000, 4),
            "sign_200_400":  "positive" if e200 > 0 else "negative",
            "sign_400_1000": "negative" if e1000 < 0 else "positive",
        }

    # Copula SSE (from baseline CSV)
    cop_map = {int(baseline[k]["H2_Tank_t"]): baseline[k]["TSSP_BESS_P_MW"]
               for k in baseline if int(baseline[k]["H2_Tank_t"]) in [200, 400, 1000]}
    if all(k in cop_map for k in [200, 400, 1000]):
        e200c, e1000c = compute_sse(cop_map[200], cop_map[400], cop_map[1000])
        sse_pair["copula"] = {
            "sse_200_400":  round(e200c, 4),
            "sse_400_1000": round(e1000c, 4),
            "sign_200_400":  "positive" if e200c > 0 else "negative",
            "sign_400_1000": "negative" if e1000c < 0 else "positive",
        }

    # ---- Print summary ----
    print_summary(results, sse_pair, baseline)

    # ---- Save results ----
    with open(OUTPUT_JSON, "w") as f:
        json.dump({"results": results, "sse": sse_pair}, f, indent=2, default=float)
    print(f"\n  JSON saved: {OUTPUT_JSON}")

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"  CSV saved:  {OUTPUT_CSV}")

    print("\nDone. Compare with baseline (Copula) results in Supplementary S3.")


if __name__ == "__main__":
    main()
