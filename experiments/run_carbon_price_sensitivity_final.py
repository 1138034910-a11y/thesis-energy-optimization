#!/usr/bin/env python3
"""
================================================================================
碳价格敏感性实验 -- 最终优化版（选项B：固定H₂上限400t）
================================================================================
核心设计：
  • H₂储罐上限固定400t（与主实验完全一致）
  • 碳价范围：[0, 50, 80, 100, 150, 200, 300, 500] CNY/ton CO₂
  • MIPGap收紧至1.5%
  • 代表日和场景只生成一次，所有碳价复用
  • 断点恢复：支持从crash中断点继续（检测partial文件）
  • 碳价=0时自动延长TimeLimit至6h（对称性导致求解更困难）

政策含义：
  "当氢储能已达经济最优上限（400t）时，碳价上升如何重塑电力侧储能组合？"

使用方法：
  python run_carbon_price_sensitivity_final.py
  
  # 如果需要强制从头开始（忽略之前的partial文件）：
  # 手动删除 results/tables/carbon_price_partial_*.csv

预计时间：8个碳价 × ~35-90分钟 ≈ 5-10小时（建议overnight跑）
================================================================================
"""

import sys
import os
import time
import json
import glob

_project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _project_root)
sys.path.insert(0, os.path.join(_project_root, "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import (
    EconParams, PhysParams, SolverParams, DataPaths,
    build_load_profile, RepDayParams, ScenarioParams, PlotParams
)
from src.representative_days import run_representative_day_pipeline, scale_annual_constraints
from src.scenario_generator import generate_reduced_scenarios
from src.stochastic_model import build_two_stage_model, solve_and_extract


# ==============================================================================
# CONFIGURATION
# ==============================================================================
CARBON_PRICES_CNY_PER_TON = [0, 50, 80, 100, 150, 200, 300, 500]
CARBON_PRICES_INTERNAL = [p / 1e4 for p in CARBON_PRICES_CNY_PER_TON]

RANDOM_SEED = 42
OUTPUT_DIR = "results/tables"
FIGURE_DIR = "results/figures"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)


# ==============================================================================
# RESUME LOGIC: Detect partial results from previous run
# ==============================================================================
def check_resume():
    """
    Check for existing partial results. If found, return the set of completed
    carbon prices so we can skip them. This protects against crashes/power loss.
    """
    pattern = os.path.join(OUTPUT_DIR, "carbon_price_partial_*.csv")
    partial_files = sorted(glob.glob(pattern))
    
    if not partial_files:
        return set(), None
    
    # Use the most recent partial file
    latest = partial_files[-1]
    try:
        df_partial = pd.read_csv(latest)
        completed = set(df_partial["carbon_price_cny_per_ton"].dropna().astype(int))
        print(f"  [Resume] Found partial file: {os.path.basename(latest)}")
        print(f"  [Resume] Completed prices: {sorted(completed)}")
        return completed, latest
    except Exception as e:
        print(f"  [Resume] Warning: Could not read partial file ({e}). Starting fresh.")
        return set(), None


# ==============================================================================
# 1. LOAD DATA
# ==============================================================================
print("=" * 70)
print("Carbon Price Sensitivity -- Final Optimized (H₂ Tank capped at 400t)")
print("=" * 70)

df_w = pd.read_csv(DataPaths["wind_pred"])
df_s = pd.read_csv(DataPaths["solar_pred"])
kan_df = pd.read_csv("results/tables/kan_forecasts.csv")

wind_actual = kan_df["wind_mu"].bfill().values
solar_actual = kan_df["solar_mu"].bfill().values
load_full = build_load_profile(8760, PhysParams["Load_Base"])

wind_sigma_full = kan_df["wind_sigma"].values[:len(wind_actual)]
solar_sigma_full = kan_df["solar_sigma"].values[:len(solar_actual)]


# ==============================================================================
# 2. REPRESENTATIVE DAYS & SCENARIOS (compute once, reuse)
# ==============================================================================
print(f"\n[Step 1/4] Representative days (seed={RANDOM_SEED})...")
reps, wind_r, solar_r, load_r, weights_days = run_representative_day_pipeline(
    wind_actual, solar_actual, load_full,
    n_days=RepDayParams["n_days"], seed=RANDOM_SEED
)

econ_base, phys_mod = scale_annual_constraints(EconParams, PhysParams, reps)
phys_mod["T"] = len(load_r)

wind_sigma_r, solar_sigma_r = [], []
for d in reps["day_indices"]:
    s, e = d * 24, d * 24 + 24
    wind_sigma_r.extend(wind_sigma_full[s:e])
    solar_sigma_r.extend(solar_sigma_full[s:e])
wind_sigma_r = np.array(wind_sigma_r)
solar_sigma_r = np.array(solar_sigma_r)

print(f"      {RepDayParams['n_days']} days × 24h = {phys_mod['T']} time steps")

print(f"\n[Step 2/4] Scenario generation (seed={RANDOM_SEED})...")
wind_sc, solar_sc, weights_sc = generate_reduced_scenarios(
    wind_r, wind_sigma_r, solar_r, solar_sigma_r,
    n_sample=ScenarioParams["N_sample"],
    n_scenario=ScenarioParams["N_scenario"],
    seed=RANDOM_SEED,
    rho_override=-0.30
)
print(f"      {len(weights_sc)} scenarios | weights: {np.round(weights_sc, 4)}")


# ==============================================================================
# 3. CHECK RESUME STATE
# ==============================================================================
completed_prices, partial_path_old = check_resume()
if completed_prices:
    remaining = [p for p in CARBON_PRICES_CNY_PER_TON if p not in completed_prices]
    print(f"\n[Resume] Will skip {len(completed_prices)} completed prices.")
    print(f"[Resume] Remaining to run: {remaining}")
    if not remaining:
        print("[Resume] All prices already completed! Exiting.")
        sys.exit(0)
else:
    remaining = CARBON_PRICES_CNY_PER_TON[:]

timestamp = time.strftime("%Y%m%d_%H%M%S")


# ==============================================================================
# 4. CARBON PRICE SWEEP
# ==============================================================================
results = []
# Load existing partial results if resuming
if completed_prices and partial_path_old:
    try:
        df_existing = pd.read_csv(partial_path_old)
        # Convert DataFrame rows back to dicts
        for _, row in df_existing.iterrows():
            results.append(row.to_dict())
    except Exception:
        pass

print(f"\n[Step 3/4] Carbon price sweep ({len(remaining)} remaining out of {len(CARBON_PRICES_CNY_PER_TON)} total)...")

for i, (price_cny, price_internal) in enumerate(
    zip(CARBON_PRICES_CNY_PER_TON, CARBON_PRICES_INTERNAL), 1
):
    if price_cny in completed_prices:
        print(f"\n  -- {i}/{len(CARBON_PRICES_CNY_PER_TON)}: Carbon price = {price_cny} CNY/ton [SKIPPED - already done] --")
        continue

    print(f"\n  -- {i}/{len(CARBON_PRICES_CNY_PER_TON)}: Carbon price = {price_cny} CNY/ton --")

    econ_mod = dict(econ_base)
    econ_mod["Carbon_price"] = float(price_internal)

    # Tightened solver config with special handling for carbon_price = 0
    solver_cfg = dict(SolverParams)
    solver_cfg["MIPGap"] = 0.015  # 1.5%
    
    if price_cny == 0:
        # Carbon price = 0: carbon cost term vanishes from objective, increasing
        # symmetry. Extended TimeLimit and more aggressive heuristics help.
        solver_cfg["TimeLimit"] = 21600  # 6 hours
        solver_cfg["Heuristics"] = 0.7   # More aggressive heuristics
        print(f"      [Config] Carbon price = 0: extended TimeLimit=6h, Heuristics=0.7")
    else:
        solver_cfg["TimeLimit"] = 14400  # 4 hours (standard)

    t0 = time.time()
    try:
        tssp_model, tssp_var_dict = build_two_stage_model(
            load_r, wind_sc, solar_sc, weights_sc, econ_mod, phys_mod, solver_cfg
        )
        tssp_res, status = solve_and_extract(
            tssp_model, tssp_var_dict, load_r, wind_sc, solar_sc, weights_sc, phys_mod
        )
    except Exception as e:
        print(f"      X EXCEPTION during solve: {e}")
        tssp_res, status = None, f"EXCEPTION: {e}"

    solve_time = time.time() - t0

    if tssp_res is None:
        print(f"      X FAILED: status={status}")
        results.append({
            "carbon_price_cny_per_ton": price_cny,
            "status": f"FAILED ({status})" if isinstance(status, str) else "FAILED",
        })
        # Still save partial results so we don't lose progress
        partial_path = os.path.join(OUTPUT_DIR, f"carbon_price_partial_{timestamp}.csv")
        pd.DataFrame(results).to_csv(partial_path, index=False, encoding="utf-8-sig")
        print(f"      [Checkpoint] Saved partial: {partial_path}")
        continue

    caps = tssp_res["capacity"]
    costs = tssp_res["costs"]

    carbon_emission_ton = costs["carbon_annual"]
    carbon_cost_10k_cny = carbon_emission_ton * price_internal
    objval = tssp_res["objval"]

    bess_duration = caps["BESS_E_MWh"] / caps["BESS_P_MW"] if caps["BESS_P_MW"] > 0 else 0
    elc_fc_ratio = caps["ELC_P_MW"] / caps["FC_P_MW"] if caps["FC_P_MW"] > 0 else 0

    record = {
        "carbon_price_cny_per_ton": price_cny,
        "carbon_price_internal": price_internal,
        "status": "OPTIMAL" if status == 2 else f"STATUS_{status}",
        "BESS_P_MW": round(caps["BESS_P_MW"], 2),
        "BESS_E_MWh": round(caps["BESS_E_MWh"], 2),
        "BESS_Duration_h": round(bess_duration, 2),
        "ELC_P_MW": round(caps["ELC_P_MW"], 2),
        "FC_P_MW": round(caps["FC_P_MW"], 2),
        "ELC_FC_Ratio": round(elc_fc_ratio, 2),
        "H2_Tank_kg": round(caps["H2_Tank_kg"], 2),
        "H2_Tank_t": round(caps["H2_Tank_kg"] / 1000, 2),
        "objval_10k_cny": round(objval, 2),
        "objval_million_cny": round(objval / 100, 2),
        "mipgap_pct": round(tssp_res["mipgap"] * 100, 2),
        "solve_time_s": round(solve_time, 1),
        "solve_time_min": round(solve_time / 60, 1),
        "cost_inv_10k": round(costs["inv"], 2),
        "cost_om_fix_10k": round(costs["om_fix"], 2),
        "cost_op_10k": round(costs["op_exp"], 2),
        "cost_carbon_10k": round(carbon_cost_10k_cny, 2),
        "revenue_total_10k": round(costs["revenue"], 2),
        "net_cost_10k": round(costs["inv"] + costs["om_fix"] + costs["op_exp"] + carbon_cost_10k_cny - costs["revenue"], 2),
        "carbon_emission_kt": round(carbon_emission_ton / 1000, 2),
        "carbon_emission_Mt": round(carbon_emission_ton / 1e6, 4),
        "carbon_cap_kt": round(econ_mod.get("Carbon_cap_annual", EconParams["Carbon_cap_annual"]) / 1000, 2),
        "carbon_cap_binding": "YES" if carbon_emission_ton >= 0.999 * econ_mod.get("Carbon_cap_annual", EconParams["Carbon_cap_annual"]) else "NO",
    }

    results.append(record)

    # Checkpoint: save after every successful (or failed) solve
    partial_path = os.path.join(OUTPUT_DIR, f"carbon_price_partial_{timestamp}.csv")
    pd.DataFrame(results).to_csv(partial_path, index=False, encoding="utf-8-sig")

    print(f"      OK Obj={record['objval_million_cny']:.2f}M, Gap={record['mipgap_pct']:.2f}%, Time={record['solve_time_min']:.1f}min")
    print(f"      Cap: BESS={record['BESS_P_MW']:.0f}MW (Dur={record['BESS_Duration_h']:.1f}h), ELC={record['ELC_P_MW']:.0f}MW, FC={record['FC_P_MW']:.0f}MW, H2={record['H2_Tank_t']:.0f}t")
    print(f"      Carbon: {record['carbon_emission_kt']:.1f}kt (cap={record['carbon_cap_kt']:.0f}kt, {record['carbon_cap_binding']})")
    print(f"      [Checkpoint] Saved: {partial_path}")


# ==============================================================================
# 5. SAVE FINAL RESULTS
# ==============================================================================
print(f"\n[Step 4/4] Saving final results...")

df_results = pd.DataFrame(results)

csv_path = os.path.join(OUTPUT_DIR, f"carbon_price_sensitivity_final_{timestamp}.csv")
df_results.to_csv(csv_path, index=False, encoding="utf-8-sig")
print(f"      CSV: {csv_path}")

json_path = os.path.join(OUTPUT_DIR, f"carbon_price_sensitivity_final_{timestamp}.json")
meta = {
    "experiment": "carbon_price_sensitivity_final",
    "description": "H2_Tank capped at 400t, MIPGap=1.5%, with resume support",
    "timestamp": timestamp,
    "random_seed": RANDOM_SEED,
    "solver_mipgap_actual": 0.015,
    "carbon_prices_cny_per_ton": CARBON_PRICES_CNY_PER_TON,
    "h2_tank_max_kg": PhysParams["Cap_H2_Tank_Max"],
    "results": results,
}
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)
print(f"      JSON: {json_path}")

# Clean up partial file if all succeeded
if partial_path_old and len(results) == len(CARBON_PRICES_CNY_PER_TON):
    all_success = all(r.get("status", "").startswith("OPTIMAL") for r in results)
    if all_success:
        try:
            os.remove(partial_path_old)
            print(f"      Cleaned up old partial: {os.path.basename(partial_path_old)}")
        except OSError:
            pass


# ==============================================================================
# 6. FIGURES
# ==============================================================================
print(f"\nGenerating figures archive...")

df = df_results[df_results["status"].str.contains("OPTIMAL|STATUS_9")].copy()
if len(df) == 0:
    print("! No successful solves -- skipping figures archive.")
    sys.exit(0)

df = df.sort_values("carbon_price_cny_per_ton")
colors = PlotParams["color_palette"]

fp = {
    "font.family": PlotParams["font_family"],
    "font.sans-serif": PlotParams["font_sans"],
    "font.size": PlotParams["font_size"],
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
}
for k, v in fp.items():
    plt.rcParams[k] = v

# Figure 1: Power capacities + BESS Duration
fig, axes = plt.subplots(1, 2, figsize=(7.48, 2.8), dpi=PlotParams["dpi"])

ax = axes[0]
ax.plot(df["carbon_price_cny_per_ton"], df["BESS_P_MW"],
        marker="o", markersize=PlotParams["marker_size"], linewidth=PlotParams["line_width"],
        color=colors["bess"], label="BESS")
ax.plot(df["carbon_price_cny_per_ton"], df["ELC_P_MW"],
        marker="s", markersize=PlotParams["marker_size"], linewidth=PlotParams["line_width"],
        color=colors["h2"], label="Electrolyzer")
ax.plot(df["carbon_price_cny_per_ton"], df["FC_P_MW"],
        marker="^", markersize=PlotParams["marker_size"], linewidth=PlotParams["line_width"],
        color=colors["fc"], label="Fuel Cell")
ax.set_xlabel("Carbon Price (CNY/tCO₂)", fontsize=PlotParams["font_size"])
ax.set_ylabel("Capacity (MW)", fontsize=PlotParams["font_size"])
ax.set_title("(a) Power Capacity Decisions", fontsize=PlotParams["font_size_large"], fontweight="bold")
ax.legend(fontsize=PlotParams["font_size_small"], frameon=False, loc="best")
ax.grid(True, alpha=0.3, linewidth=0.4)
ax.tick_params(labelsize=PlotParams["font_size_small"])

ax = axes[1]
ax.plot(df["carbon_price_cny_per_ton"], df["BESS_Duration_h"],
        marker="D", markersize=PlotParams["marker_size"], linewidth=PlotParams["line_width"],
        color=colors["bess"])
ax.set_xlabel("Carbon Price (CNY/tCO₂)", fontsize=PlotParams["font_size"])
ax.set_ylabel("BESS Duration (h)", fontsize=PlotParams["font_size"])
ax.set_title("(b) BESS Storage Duration", fontsize=PlotParams["font_size_large"], fontweight="bold")
ax.grid(True, alpha=0.3, linewidth=0.4)
ax.tick_params(labelsize=PlotParams["font_size_small"])

plt.tight_layout()
fig_path = os.path.join(FIGURE_DIR, f"fig_carbon_price_capacity_{timestamp}.png")
fig.savefig(fig_path, dpi=PlotParams["dpi"], bbox_inches="tight")
plt.close()
print(f"      Fig 1: {fig_path}")

# Figure 2: Cost structure
fig, ax = plt.subplots(figsize=(3.54, 2.8), dpi=PlotParams["dpi"])

x = np.arange(len(df))
width = 0.55
labels = [f"{int(p)}" for p in df["carbon_price_cny_per_ton"]]

bottom = np.zeros(len(df))
cost_items = [
    ("cost_inv_10k", "Investment", "#4C78A8"),
    ("cost_om_fix_10k", "Fixed O&M", "#72B7B2"),
    ("cost_op_10k", "Operation", "#F58518"),
    ("cost_carbon_10k", "Carbon Cost", "#E45756"),
]
for col, name, color in cost_items:
    vals = df[col].values / 100
    ax.bar(x, vals, width, bottom=bottom, label=name, color=color, edgecolor="white", linewidth=0.3)
    bottom += vals

rev_vals = -df["revenue_total_10k"].values / 100
ax.bar(x, rev_vals, width, label="Revenue", color="#54A24B", edgecolor="white", linewidth=0.3)

ax.set_xlabel("Carbon Price (CNY/tCO₂)", fontsize=PlotParams["font_size"])
ax.set_ylabel("Annual Cost (Million CNY)", fontsize=PlotParams["font_size"])
ax.set_title("(c) Cost Structure Breakdown", fontsize=PlotParams["font_size_large"], fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=PlotParams["font_size_small"])
ax.axhline(y=0, color="black", linewidth=0.6)
ax.legend(fontsize=PlotParams["font_size_small"], frameon=False, loc="upper left")
ax.grid(True, alpha=0.3, linewidth=0.4, axis="y")
ax.tick_params(labelsize=PlotParams["font_size_small"])

plt.tight_layout()
fig_path2 = os.path.join(FIGURE_DIR, f"fig_carbon_price_costs_{timestamp}.png")
fig.savefig(fig_path2, dpi=PlotParams["dpi"], bbox_inches="tight")
plt.close()
print(f"      Fig 2: {fig_path2}")

# Figure 3: Carbon emissions
fig, ax = plt.subplots(figsize=(3.54, 2.8), dpi=PlotParams["dpi"])

ax.plot(df["carbon_price_cny_per_ton"], df["carbon_emission_kt"],
        marker="o", markersize=PlotParams["marker_size"], linewidth=PlotParams["line_width"],
        color="#E45756", label="Actual Emissions")
ax.axhline(y=df["carbon_cap_kt"].iloc[0], color="black", linestyle="--", linewidth=0.8,
           label=f"Cap = {df['carbon_cap_kt'].iloc[0]:.0f} kt")

ax.set_xlabel("Carbon Price (CNY/tCO₂)", fontsize=PlotParams["font_size"])
ax.set_ylabel("Annual CO₂ Emissions (kt)", fontsize=PlotParams["font_size"])
ax.set_title("(d) Carbon Emission Response", fontsize=PlotParams["font_size_large"], fontweight="bold")
ax.legend(fontsize=PlotParams["font_size_small"], frameon=False)
ax.grid(True, alpha=0.3, linewidth=0.4)
ax.tick_params(labelsize=PlotParams["font_size_small"])

plt.tight_layout()
fig_path3 = os.path.join(FIGURE_DIR, f"fig_carbon_price_emissions_{timestamp}.png")
fig.savefig(fig_path3, dpi=PlotParams["dpi"], bbox_inches="tight")
plt.close()
print(f"      Fig 3: {fig_path3}")


# ==============================================================================
# 7. SUMMARY TABLE
# ==============================================================================
print(f"\n{'=' * 70}")
print("Summary table for manuscript:")
print("=" * 70)

table_cols = [
    "carbon_price_cny_per_ton",
    "BESS_P_MW", "ELC_P_MW", "FC_P_MW", "H2_Tank_t",
    "BESS_Duration_h",
    "objval_million_cny", "mipgap_pct",
    "carbon_emission_kt", "carbon_cap_binding",
]
table = df[table_cols].copy()
table.columns = [
    "Carbon (CNY/t)",
    "BESS (MW)", "ELC (MW)", "FC (MW)", "H₂ (t)",
    "BESS Dur (h)",
    "Obj (M)", "Gap (%)",
    "CO₂ (kt)", "Binding?",
]
print(table.to_string(index=False))
print("=" * 70)

# LaTeX table with fallback
try:
    latex_table = table.to_latex(index=False, float_format="%.1f", escape=False)
    latex_path = os.path.join(OUTPUT_DIR, f"carbon_price_table_{timestamp}.tex")
    with open(latex_path, "w", encoding="utf-8") as f:
        f.write(latex_table)
    print(f"\nLaTeX table: {latex_path}")
except Exception as e:
    print(f"\n! LaTeX export skipped (missing dependency: {e})")
    print("  Tip: pip install jinja2  # if you need LaTeX tables")

print(f"\n{'=' * 70}")
print("CARBON PRICE SENSITIVITY -- COMPLETE")
print(f"{'=' * 70}")
print(f"Runs: {len(results)} | Success: {sum(1 for r in results if str(r.get('status','')).startswith('OPTIMAL'))}")
print(f"Failed: {sum(1 for r in results if 'FAIL' in str(r.get('status','')))}")
print(f"\nAll outputs in:")
print(f"  - {OUTPUT_DIR}/")
print(f"  - {FIGURE_DIR}/")
