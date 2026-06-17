"""
H2储罐敏感性实验——严格VSS补跑（基于BUG-01修复后的数据）

核心功能：
  1. 复现 h2_tank_sensitivity_v3.csv 的 EV 和 TSSP 结果（交叉验证）
  2. 补充计算严格 VSS（EEV - RP），使用 build_eev_model 确保约束一致性
  3. 保存完整结果到 JSON 和 CSV

数据逻辑：
  - EV 输入使用 KAN-mu（BUG-01修复）
  - 代表日聚合 seed=RepDayParams["seed"]
  - 场景生成 seed=ScenarioParams["seed"]
  - 与 v3 实验完全一致的数据路径和参数

预计耗时：
  - EV: 每个点 <1分钟
  - TSSP: 每个点 1-4小时（建议夜间挂机）
  - EEV: 每个点 1-4小时（与TSSP相近，但变量固定后可能更快收敛）

用法:
  python experiments/run_h2_sensitivity_v3_rigorous_vss.py
"""
import os
import sys
import time
import json

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
from src.stochastic_model import build_two_stage_model, solve_and_extract, build_eev_model

# ============================================================
# 实验配置
# ============================================================
H2_CAPACITIES = [200_000, 400_000, 800_000, 1_000_000]
OUTPUT_JSON = "results/tables/h2_sensitivity_v3_rigorous_vss.json"
OUTPUT_CSV = "results/tables/h2_sensitivity_v3_rigorous_vss.csv"
V3_CSV = "results/tables/h2_tank_sensitivity_v3.csv"

print("=" * 70)
print("H2储罐敏感性实验——严格VSS补跑（基于BUG-01修复后的数据）")
print("=" * 70)
print(f"测试容量点: {[f'{c/1000:.0f}t' for c in H2_CAPACITIES]}")
print(f"求解参数: MIPGap={SolverParams['MIPGap']}, TimeLimit={SolverParams['TimeLimit']}s")
print()

# ============================================================
# Step 0: 加载已有v3数据（用于交叉验证）
# ============================================================
print("[Step 0] 加载已有v3数据用于交叉验证...")
if os.path.exists(V3_CSV):
    df_v3 = pd.read_csv(V3_CSV)
    v3_ref = {}
    for _, row in df_v3.iterrows():
        v3_ref[int(row["H2_Tank_t"])] = row.to_dict()
    print(f"  已加载 {len(v3_ref)} 个参考点")
else:
    v3_ref = {}
    print(f"  警告: 未找到 {V3_CSV}，跳过交叉验证")

# ============================================================
# Step 1: 数据准备（与v3完全一致）
# ============================================================
print("\n[Step 1] 数据准备...")
t0_total = time.time()

# 1. 原始实际数据（仅用于对齐长度）
df_w = pd.read_csv(DataPaths["wind_pred"])
df_s = pd.read_csv(DataPaths["solar_pred"])
wind_actual_raw = df_w["actual_pu"].bfill().values
solar_actual_raw = df_s["actual_pu"].bfill().values

# 2. KAN预测的期望值(mu)和不确定性(sigma)
kan_df = pd.read_csv("results/tables/kan_forecasts.csv")
wind_mu_full = kan_df["wind_mu"].bfill().values
solar_mu_full = kan_df["solar_mu"].bfill().values

# 3. 对齐长度
n_len = min(len(wind_actual_raw), len(wind_mu_full))
wind_actual_raw = wind_actual_raw[:n_len]
solar_actual_raw = solar_actual_raw[:n_len]
wind_mu_full = wind_mu_full[:n_len]
solar_mu_full = solar_mu_full[:n_len]

# 4. BUG-01修复：EV输入用KAN-mu（与v3完全一致）
wind_actual = wind_mu_full
solar_actual = solar_mu_full

load_full = build_load_profile(8760, PhysParams["Load_Base"])

# 5. 代表日聚合
reps, wind_r, solar_r, load_r, weights_days = run_representative_day_pipeline(
    wind_actual, solar_actual, load_full,
    n_days=RepDayParams["n_days"], seed=RepDayParams["seed"]
)

# 6. 提取代表日对应的sigma
wind_sigma_full = kan_df["wind_sigma"].values[:n_len]
solar_sigma_full = kan_df["solar_sigma"].values[:n_len]
wind_sigma_r = []
solar_sigma_r = []
for d in reps["day_indices"]:
    s = d * 24
    e = s + 24
    wind_sigma_r.extend(wind_sigma_full[s:e])
    solar_sigma_r.extend(solar_sigma_full[s:e])
wind_sigma_r = np.array(wind_sigma_r)
solar_sigma_r = np.array(solar_sigma_r)

# 7. 场景生成（seed固定，确保与v3一致）
wind_sc, solar_sc, weights_sc = generate_reduced_scenarios(
    wind_r, wind_sigma_r, solar_r, solar_sigma_r,
    n_sample=ScenarioParams["N_sample"],
    n_scenario=ScenarioParams["N_scenario"],
    seed=ScenarioParams["seed"],
    rho_override=-0.30
)

print(f"  数据加载完成: {time.time()-t0_total:.1f}s")
print(f"  代表日: {RepDayParams['n_days']}天, 场景: {len(weights_sc)}个")

# ============================================================
# Step 2: 主循环——逐个H2容量点跑 EV → TSSP → EEV
# ============================================================
results = []

for cap_h2 in H2_CAPACITIES:
    cap_t = int(cap_h2 / 1000)
    print(f"\n{'='*70}")
    print(f"H2储罐容量 = {cap_t} t")
    print(f"{'='*70}")

    # 修改物理参数
    phys_mod = dict(PhysParams)
    phys_mod["Cap_H2_Tank"] = cap_h2
    phys_mod["Cap_H2_Tank_Max"] = cap_h2
    phys_mod["T"] = len(load_r)
    econ_base = dict(EconParams)

    res_point = {
        "H2_Tank_t": cap_t,
        "H2_Tank_kg": cap_h2,
    }

    # ---------- EV ----------
    t0 = time.time()
    print("\n  [EV] 求解中...")
    ev_res, _ = build_deterministic_model(load_r, wind_r, solar_r, econ_base, phys_mod, SolverParams)
    ev_time = time.time() - t0

    if not ev_res:
        print(f"  ❌ EV FAILED for H2={cap_t}t")
        continue

    print(f"  ✅ EV: Obj={ev_res['objval']:.2f}, Gap={ev_res['mipgap']:.4f}%, Time={ev_time:.1f}s")
    print(f"     Cap: BESS_P={ev_res['capacity']['BESS_P_MW']:.0f}, "
          f"BESS_E={ev_res['capacity']['BESS_E_MWh']:.0f}, "
          f"ELC={ev_res['capacity']['ELC_P_MW']:.0f}, "
          f"FC={ev_res['capacity']['FC_P_MW']:.0f}")

    # 交叉验证：与v3 CSV对比
    if cap_t in v3_ref:
        ref = v3_ref[cap_t]
        diff_bess = abs(ev_res['capacity']['BESS_P_MW'] - ref['EV_BESS_P_MW'])
        diff_obj = abs(ev_res['objval'] - ref['EV_Obj'])
        if diff_bess > 1.0 or diff_obj > 1000:
            print(f"  ⚠️  交叉验证警告: EV结果与v3参考差异较大!")
            print(f"     BESS_P diff={diff_bess:.1f} MW, Obj diff={diff_obj:.0f}")
        else:
            print(f"  ✅ 交叉验证通过: 与v3参考一致 (BESS_P diff={diff_bess:.2f}, Obj diff={diff_obj:.0f})")

    res_point["EV_Obj"] = ev_res["objval"]
    res_point["EV_Gap_pct"] = ev_res["mipgap"]
    res_point["EV_Time_s"] = ev_time
    res_point["EV_BESS_P_MW"] = ev_res["capacity"]["BESS_P_MW"]
    res_point["EV_BESS_E_MWh"] = ev_res["capacity"]["BESS_E_MWh"]
    res_point["EV_ELC_MW"] = ev_res["capacity"]["ELC_P_MW"]
    res_point["EV_FC_MW"] = ev_res["capacity"]["FC_P_MW"]

    # ---------- TSSP ----------
    t0 = time.time()
    print("\n  [TSSP] 求解中...")
    tssp_model, tssp_var_dict = build_two_stage_model(
        load_r, wind_sc, solar_sc, weights_sc, econ_base, phys_mod, SolverParams
    )
    # Warm start from EV
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
        print(f"  ❌ TSSP FAILED for H2={cap_t}t")
        continue

    print(f"  ✅ TSSP: Obj={tssp_res['objval']:.2f}, Gap={tssp_res['mipgap']:.4f}%, Time={tssp_time:.1f}s")
    print(f"     Cap: BESS_P={tssp_res['capacity']['BESS_P_MW']:.0f}, "
          f"BESS_E={tssp_res['capacity']['BESS_E_MWh']:.0f}, "
          f"ELC={tssp_res['capacity']['ELC_P_MW']:.0f}, "
          f"FC={tssp_res['capacity']['FC_P_MW']:.0f}")

    # 交叉验证：与v3 CSV对比
    if cap_t in v3_ref:
        ref = v3_ref[cap_t]
        diff_bess = abs(tssp_res['capacity']['BESS_P_MW'] - ref['TSSP_BESS_P_MW'])
        diff_obj = abs(tssp_res['objval'] - ref['TSSP_Obj'])
        if diff_bess > 1.0 or diff_obj > 1000:
            print(f"  ⚠️  交叉验证警告: TSSP结果与v3参考差异较大!")
            print(f"     BESS_P diff={diff_bess:.1f} MW, Obj diff={diff_obj:.0f}")
        else:
            print(f"  ✅ 交叉验证通过: 与v3参考一致 (BESS_P diff={diff_bess:.2f}, Obj diff={diff_obj:.0f})")

    res_point["TSSP_Obj"] = tssp_res["objval"]
    res_point["TSSP_Gap_pct"] = tssp_res["mipgap"]
    res_point["TSSP_Time_s"] = tssp_time
    res_point["TSSP_BESS_P_MW"] = tssp_res["capacity"]["BESS_P_MW"]
    res_point["TSSP_BESS_E_MWh"] = tssp_res["capacity"]["BESS_E_MWh"]
    res_point["TSSP_ELC_MW"] = tssp_res["capacity"]["ELC_P_MW"]
    res_point["TSSP_FC_MW"] = tssp_res["capacity"]["FC_P_MW"]

    # ---------- EEV (严格VSS) ----------
    t0 = time.time()
    print("\n  [EEV] 严格VSS计算中...")
    print("        使用 build_eev_model 固定EV容量，在随机场景下重新评估...")

    eev_model, eev_var_dict = build_eev_model(
        load_r, wind_sc, solar_sc, weights_sc,
        ev_res["capacity"], econ_base, phys_mod, SolverParams
    )

    eev_res, eev_status = solve_and_extract(
        eev_model, eev_var_dict, load_r, wind_sc, solar_sc, weights_sc, phys_mod
    )
    eev_time = time.time() - t0

    if not eev_res:
        print(f"  ❌ EEV FAILED for H2={cap_t}t (status={eev_status})")
        res_point["EEV_Obj"] = None
        res_point["VSS"] = None
        res_point["VSS_pct"] = None
    else:
        z_eev = eev_res["objval"]
        z_rp = tssp_res["objval"]
        vss_rigorous = z_eev - z_rp
        vss_pct = 100 * vss_rigorous / abs(z_rp) if z_rp != 0 else 0

        print(f"  ✅ EEV: Obj={z_eev:.2f}, Time={eev_time:.1f}s")
        print(f"     z_EEV = {z_eev:.2f}")
        print(f"     z_RP  = {z_rp:.2f}")
        print(f"     VSS   = {vss_rigorous:.2f} ({vss_pct:.2f}%)")

        res_point["EEV_Obj"] = z_eev
        res_point["EEV_Gap_pct"] = eev_res["mipgap"]
        res_point["EEV_Time_s"] = eev_time
        res_point["VSS"] = vss_rigorous
        res_point["VSS_pct"] = vss_pct

    results.append(res_point)

# ============================================================
# Step 3: 汇总输出与保存
# ============================================================
print(f"\n{'='*70}")
print("严格VSS补跑完成——汇总表")
print(f"{'='*70}")
print(f"{'H2(t)':>8} {'EV_Obj':>14} {'TSSP_Obj':>14} {'EEV_Obj':>14} {'VSS':>12} {'VSS(%)':>8}")
print("-" * 80)
for r in results:
    if r.get("VSS") is not None:
        print(f"{r['H2_Tank_t']:>8.0f} {r['EV_Obj']:>14.2f} {r['TSSP_Obj']:>14.2f} "
              f"{r['EEV_Obj']:>14.2f} {r['VSS']:>12.2f} {r['VSS_pct']:>8.2f}")
    else:
        print(f"{r['H2_Tank_t']:>8.0f} {r['EV_Obj']:>14.2f} {r['TSSP_Obj']:>14.2f} "
              f"{'N/A':>14} {'N/A':>12} {'N/A':>8}")

# 保存 JSON
with open(OUTPUT_JSON, "w") as f:
    json.dump(results, f, indent=2, default=float)
print(f"\n✅ JSON结果已保存: {OUTPUT_JSON}")

# 保存 CSV
df_results = pd.DataFrame(results)
df_results.to_csv(OUTPUT_CSV, index=False)
print(f"✅ CSV结果已保存: {OUTPUT_CSV}")

print(f"\n总耗时: {time.time()-t0_total:.1f}s")
print("=" * 70)
print("下一步操作提示:")
print("  1. 检查上述汇总表中的VSS数值是否合理（应在0-5%范围内）")
print("  2. 如果VSS出现负值或>10%的异常值，说明可能存在数据问题")
print("  3. 将CSV中的VSS_pct列用于重绘VSS-H2趋势图")
print("  4. 将结果发送给导师检查")
print("=" * 70)
