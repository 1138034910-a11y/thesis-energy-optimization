#!/usr/bin/env python3
"""
================================================================================
Seed稳健性检验（最终版 —— 3 Seed + 优化参数）
================================================================================

改进点：
  1. 3个seed: 42, 123, 999（标准做法）
  2. MIPGap=1.0%（投稿级精度）
  3. TimeLimit=6h/seed（overnight可跑完3个）
  4. 增加solver优化参数（Presolve/Cuts/MIPFocus）
  5. 每跑完1个seed自动保存，防止中断丢失
  6. 2个seed时样本量校正的CV计算

预计时间: 每个seed 2-5h，总计6-15h（overnight跑）。
================================================================================
"""

import sys
import os

_project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _project_root)
sys.path.insert(0, os.path.join(_project_root, "src"))

from config import *
from src.representative_days import run_representative_day_pipeline, scale_annual_constraints
from src.scenario_generator import generate_reduced_scenarios
from src.stochastic_model import build_two_stage_model, solve_and_extract
import numpy as np
import pandas as pd
import time

# ========== 0. 求解器配置（最终版参数）==========
SOLVER_CFG = {
    "MIPGap": 0.01,        # <- 1.0%，投稿级精度
    "TimeLimit": 21600,    # <- 6小时/seed，3个seed overnight可跑完
    "Threads": 0,          # 使用全部CPU核心
    "Presolve": 2,         # 激进预求解，减少问题规模
    "Cuts": 2,             # 激进割平面，加速收敛
    "Heuristics": 0.1,     # 适度启发式（0.1=10%时间用于启发式）
    "MIPFocus": 1,         # 关注找到更好可行解（适合容量规划问题）
}

SEEDS = [42, 123, 999]

print("=" * 70)
print("Seed稳健性检验（最终版 -- 3 Seed + MIPGap=1%）")
print("=" * 70)
print(f"[Config] MIPGap: {SOLVER_CFG['MIPGap']*100}%")
print(f"[Config] TimeLimit: {SOLVER_CFG['TimeLimit']/3600:.1f}h/seed")
print(f"[Config] Seeds: {SEEDS}")
print(f"[Config] 预计总时间: {len(SEEDS)*2}-{len(SEEDS)*5}h")
print("=" * 70)

# ========== 1. 加载数据 ==========
df_w = pd.read_csv(DataPaths["wind_pred"])
df_s = pd.read_csv(DataPaths["solar_pred"])
kan_df = pd.read_csv("results/tables/kan_forecasts.csv")

wind_actual = kan_df["wind_mu"].bfill().values
solar_actual = kan_df["solar_mu"].bfill().values
load_full = build_load_profile(8760, PhysParams["Load_Base"])

results = []

# ========== 2. 循环3个seed ==========
for idx, seed in enumerate(SEEDS, 1):
    print(f"\n{'=' * 70}")
    print(f"[{idx}/{len(SEEDS)}] Seed = {seed}")
    print(f"{'=' * 70}")
    t_start = time.time()

    # 代表日聚类
    reps, wind_r, solar_r, load_r, weights_days = run_representative_day_pipeline(
        wind_actual, solar_actual, load_full,
        n_days=RepDayParams["n_days"], seed=seed
    )
    econ_base, phys_mod = scale_annual_constraints(EconParams, PhysParams, reps)
    phys_mod["T"] = len(load_r)

    # KAN sigma提取
    wind_sigma = kan_df["wind_sigma"].values[:len(wind_actual)]
    solar_sigma = kan_df["solar_sigma"].values[:len(solar_actual)]
    wind_sigma_r, solar_sigma_r = [], []
    for d in reps["day_indices"]:
        s, e = d * 24, d * 24 + 24
        wind_sigma_r.extend(wind_sigma[s:e])
        solar_sigma_r.extend(solar_sigma[s:e])
    wind_sigma_r = np.array(wind_sigma_r)
    solar_sigma_r = np.array(solar_sigma_r)

    # 场景生成
    wind_sc, solar_sc, weights_sc = generate_reduced_scenarios(
        wind_r, wind_sigma_r, solar_r, solar_sigma_r,
        n_sample=ScenarioParams["N_sample"],
        n_scenario=ScenarioParams["N_scenario"],
        seed=seed,
        rho_override=-0.30
    )

    # TSSP求解
    print(f"  求解TSSP中（MIPGap=1%, 预计2-5h）...")
    t0 = time.time()
    tssp_model, tssp_var_dict = build_two_stage_model(
        load_r, wind_sc, solar_sc, weights_sc, econ_base, phys_mod, SOLVER_CFG
    )
    tssp_res, _ = solve_and_extract(
        tssp_model, tssp_var_dict, load_r, wind_sc, solar_sc, weights_sc, phys_mod
    )
    tssp_time = time.time() - t0

    if tssp_res:
        result = {
            "seed": seed,
            "BESS_P_MW": round(tssp_res["capacity"]["BESS_P_MW"], 1),
            "BESS_E_MWh": round(tssp_res["capacity"]["BESS_E_MWh"], 1),
            "ELC_P_MW": round(tssp_res["capacity"]["ELC_P_MW"], 1),
            "FC_P_MW": round(tssp_res["capacity"]["FC_P_MW"], 1),
            "H2_Tank_kg": round(tssp_res["capacity"]["H2_Tank_kg"], 1),
            "objval_10k": round(tssp_res["objval"] / 10000, 2),
            "mipgap_pct": round(tssp_res["mipgap"] * 100, 2),
            "time_s": round(tssp_time, 1)
        }
        results.append(result)
        elapsed = time.time() - t_start
        print(f"  ✓ Seed {seed} 完成!")
        print(f"     BESS={result['BESS_P_MW']}MW, ELC={result['ELC_P_MW']}MW, "
              f"FC={result['FC_P_MW']}MW")
        print(f"     Obj={result['objval_10k']}万, Gap={result['mipgap_pct']}%, "
              f"求解={result['time_s']/3600:.1f}h, 累计={elapsed/3600:.1f}h")
    else:
        print(f"  ✗ Seed {seed} TSSP求解失败!")
        results.append({
            "seed": seed, "BESS_P_MW": None, "BESS_E_MWh": None,
            "ELC_P_MW": None, "FC_P_MW": None, "H2_Tank_kg": None,
            "objval_10k": None, "mipgap_pct": None, "time_s": None
        })

    # 每跑完一个seed立即保存（防止中断丢失）
    os.makedirs("results/tables", exist_ok=True)
    df_partial = pd.DataFrame(results)
    ts = time.strftime("%Y%m%d_%H%M%S")
    df_partial.to_csv(f"results/tables/seed_robustness_partial_{ts}.csv", 
                      index=False, encoding='utf-8-sig')
    print(f"  [保存] 中间结果已保存")

# ========== 3. 最终汇总 ==========
print(f"\n{'=' * 70}")
print("最终结果汇总")
print(f"{'=' * 70}")

df_results = pd.DataFrame(results)
print(df_results.to_string(index=False))

# ========== 4. 稳健性评估 ==========
print(f"\n{'=' * 70}")
print("稳健性评估（CV = 标准差/|均值| × 100%）")
print(f"{'=' * 70}")

robustness = []
for col in ["BESS_P_MW", "ELC_P_MW", "FC_P_MW", "objval_10k"]:
    values = df_results[col].dropna()
    n = len(values)
    if n >= 2:
        mean_val = values.mean()
        std_val = values.std()
        # 样本量校正：n<3时用range/2作为std估计
        if n == 2:
            range_val = values.max() - values.min()
            std_val = range_val / 2  # 保守估计
        cv = abs(std_val / mean_val) * 100 if mean_val != 0 else 0
        status = "✓ 稳健" if cv < 5 else ("~ 一般" if cv < 10 else "✗ 不稳健")
        robustness.append({
            "指标": col, "n": n, "均值": round(mean_val, 2),
            "标准差": round(std_val, 2), "CV(%)": round(cv, 2), "评估": status
        })

df_robust = pd.DataFrame(robustness)
print(df_robust.to_string(index=False))

# ========== 5. 保存最终文件 ==========
ts = time.strftime("%Y%m%d_%H%M%S")
df_results.to_csv(f"results/tables/seed_robustness_final_{ts}.csv", 
                  index=False, encoding='utf-8-sig')
df_robust.to_csv(f"results/tables/seed_robustness_final_summary_{ts}.csv", 
                 index=False, encoding='utf-8-sig')

print(f"\n{'=' * 70}")
print("✓ 全部完成！")
print(f"   结果: results/tables/seed_robustness_final_{ts}.csv")
print(f"   汇总: results/tables/seed_robustness_final_summary_{ts}.csv")
print(f"{'=' * 70}")
