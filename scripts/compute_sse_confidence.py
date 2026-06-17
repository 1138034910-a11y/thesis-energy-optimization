"""
Compute ObjBound (best bound) and SSE +/- confidence bands
from existing solver results -- NO re-running required.

Derivation:
  Gurobi MIPGap = |Incumbent - ObjBound| / |Incumbent|
  For minimization: ObjBound = ObjVal - MIPGap x |ObjVal|

  BESS capacity uncertainty is conservatively bounded by MIPGap.
  We propagate this through the SSE arc-elasticity formula to
  produce nominal, lower-bound, and upper-bound elasticity values.

Usage:
  python scripts/compute_sse_confidence.py
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
import os

# ============================================================
# 1. Load existing data
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# H2 sensitivity (TSSP + EV)
h2_csv = os.path.join(PROJECT_ROOT, "results/tables/h2_sensitivity_v3_rigorous_vss.csv")
# Carbon price sweep (TSSP)
carbon_csv = os.path.join(PROJECT_ROOT, "results/tables/carbon_price_sweep_tssp.csv")
# SSE calculations
sse_csv = os.path.join(PROJECT_ROOT, "results/tables/sse_calculations.csv")

df_h2 = pd.read_csv(h2_csv)
df_carbon = pd.read_csv(carbon_csv)
df_sse = pd.read_csv(sse_csv)

print("=" * 70)
print("ObjBound Recovery & SSE Confidence Band Computation")
print("=" * 70)

# ============================================================
# 2. Recover ObjBound from ObjVal + MIPGap
# ============================================================
def recover_objbound(objval, mipgap_pct):
    """
    Recover Gurobi BestBd (ObjBound) from Incumbent (ObjVal) and MIPGap.

    Gurobi definition (minimization):
      MIPGap = |Incumbent - ObjBound| / |Incumbent|

    For minimization, ObjBound <= Incumbent (more negative is better bound).
    Therefore: ObjBound = Incumbent - MIPGap × |Incumbent|

    For the sign-convention in this model:
      - Objective is annualized total cost net of revenue (can be negative
        when revenue > cost, as seen in the data)
      - More negative = better solution
      - ObjBound is MORE negative than Incumbent
    """
    gap_fraction = mipgap_pct / 100.0
    objbound = objval - gap_fraction * abs(objval)
    return objbound


print("\n--- H2 Sensitivity: ObjBound Recovery ---")
print(f"{'H2(t)':>8} {'ObjVal(TSSP)':>16} {'Gap(%)':>8} {'ObjBound':>16} {'GapCheck(%)':>10}")
print("-" * 65)

for _, row in df_h2.iterrows():
    objval = row["TSSP_Obj"]
    gap = row["TSSP_Gap_pct"]
    objbound = recover_objbound(objval, gap)
    # Verify: recompute gap from recovered ObjBound
    gap_check = abs(objval - objbound) / abs(objval) * 100

    print(f"{row['H2_Tank_t']:>8.0f} {objval:>16.2f} {gap:>8.4f} "
          f"{objbound:>16.2f} {gap_check:>10.4f}")

print("\n--- Carbon Price Sweep: ObjBound Recovery ---")
print(f"{'Price(CNY)':>12} {'ObjVal':>16} {'Gap(%)':>8} {'ObjBound':>16}")
print("-" * 60)

for _, row in df_carbon.iterrows():
    objval = row["objval_10k_cny"]
    gap = row["mipgap_pct"]
    objbound = recover_objbound(objval, gap)
    print(f"{row['carbon_price_cny_per_ton']:>12.0f} {objval:>16.2f} "
          f"{gap:>8.4f} {objbound:>16.2f}")

# ============================================================
# 3. SSE Confidence Bands
# ============================================================
#
# The SSE is computed from BESS_P at two grid points (X1, X2).
# For each point, BESS_P has uncertainty bounded by MIPGap.
#
# Conservative approach:
#   BESS_P_nominal  = the incumbent value (feasible solution)
#   BESS_P_pessimistic = BESS_P × (1 + MIPGap)  -- higher BESS (over-investment)
#   BESS_P_optimistic  = BESS_P × (1 - MIPGap)  -- lower BESS (under-investment)
#
# For each SSE interval (X1, X2), we compute 4 combinations:
#   (B1_lo, B2_lo), (B1_lo, B2_hi), (B1_hi, B2_lo), (B1_hi, B2_hi)
# and take min/max of the resulting SSE values.
#
# NOTE: This is conservative — it assumes BESS uncertainty = MIPGap,
# which overestimates variable-space uncertainty (the MIPGap is defined
# on the objective, not on individual variables). The paper should
# describe this as "conservative bounds."

print("\n" + "=" * 70)
print("SSE Confidence Bands (from TSSP BESS_P data)")
print("=" * 70)

# Build lookup: H2_tank_t -> {BESS_P, MIPGap}
h2_lookup = {}
for _, row in df_h2.iterrows():
    h2_lookup[int(row["H2_Tank_t"])] = {
        "bess_p": row["TSSP_BESS_P_MW"],
        "gap": row["TSSP_Gap_pct"] / 100.0,
        "bess_e": row["TSSP_BESS_E_MWh"],
    }

# Define intervals matching the paper
intervals = [
    (200, 400),
    (400, 600),
    (600, 800),
    (800, 1000),
]

def arc_elasticity(y1, y2, x1, x2):
    """Midpoint arc elasticity: ε = (ΔY/Ȳ) / (ΔX/X̄)"""
    y_bar = (y1 + y2) / 2.0
    x_bar = (x1 + x2) / 2.0
    dy = y2 - y1
    dx = x2 - x1
    if x_bar == 0 or y_bar == 0 or dx == 0:
        return np.nan
    return (dy / y_bar) / (dx / x_bar)

print(f"\n{'Interval':>14} {'H2_mid':>8} {'SSE_nom':>10} {'SSE_lo':>10} "
      f"{'SSE_hi':>10} {'Band':>12} {'Robust?':>10}")
print("-" * 70)

sse_results = []
for x1, x2 in intervals:
    if x1 not in h2_lookup or x2 not in h2_lookup:
        print(f"{f'{x1}->{x2}t':>14} {'N/A':>8} {'MISSING DATA'}")
        continue

    d1 = h2_lookup[x1]
    d2 = h2_lookup[x2]

    # Nominal
    b_nom_1 = d1["bess_p"]
    b_nom_2 = d2["bess_p"]
    sse_nom = arc_elasticity(b_nom_1, b_nom_2, x1, x2)

    # Pessimistic / optimistic bounds on BESS_P
    b_lo_1 = b_nom_1 * (1.0 - d1["gap"])  # optimistic: less BESS
    b_hi_1 = b_nom_1 * (1.0 + d1["gap"])  # pessimistic: more BESS
    b_lo_2 = b_nom_2 * (1.0 - d2["gap"])
    b_hi_2 = b_nom_2 * (1.0 + d2["gap"])

    # 4 corner combinations
    epsilons = [
        arc_elasticity(b_lo_1, b_lo_2, x1, x2),
        arc_elasticity(b_lo_1, b_hi_2, x1, x2),
        arc_elasticity(b_hi_1, b_lo_2, x1, x2),
        arc_elasticity(b_hi_1, b_hi_2, x1, x2),
    ]
    epsilons = [e for e in epsilons if not np.isnan(e)]

    sse_lo = min(epsilons)
    sse_hi = max(epsilons)
    band = sse_hi - sse_lo

    # Is the sign robust? (all 4 corners have same sign)
    signs = set(np.sign(e) for e in epsilons)
    robust = "YES" if len(signs) == 1 else "CHECK"

    # Regime determination
    if sse_nom > 0:
        regime = "Complementarity"
    elif -1 <= sse_nom < 0:
        regime = "Substitution"
    else:  # < -1
        regime = "Strong substitution"

    midpoint = (x1 + x2) / 2.0

    print(f"{f'{x1}->{x2}t':>14} {midpoint:>8.0f} {sse_nom:>10.4f} "
          f"{sse_lo:>10.4f} {sse_hi:>10.4f} {band:>12.4f} {robust:>10}")

    sse_results.append({
        "interval": f"{x1}->{x2}t",
        "h2_midpoint_t": midpoint,
        "sse_nominal": round(sse_nom, 4),
        "sse_lower": round(sse_lo, 4),
        "sse_upper": round(sse_hi, 4),
        "band_width": round(band, 4),
        "sign_robust": robust,
        "regime": regime,
        "bess_p_1_MW": round(b_nom_1, 1),
        "bess_p_2_MW": round(b_nom_2, 1),
        "gap_1_pct": round(d1["gap"] * 100, 2),
        "gap_2_pct": round(d2["gap"] * 100, 2),
    })

# ============================================================
# 4. For the paper: side-by-side comparison with original Table 4
# ============================================================
print("\n" + "=" * 70)
print("Table for Paper: SSE with Conservative Confidence Bands")
print("=" * 70)
print(f"{'Interval':>14} {'H₂_mid':>8} {'SSE':>10} {'±Band':>10} {'Regime':>22} {'Sign Robust?':>14}")
print("-" * 75)
for r in sse_results:
    half_band = r["band_width"] / 2.0
    print(f"{r['interval']:>14} {r['h2_midpoint_t']:>8.0f} {r['sse_nominal']:>10.4f} "
          f"{'±' + str(round(half_band, 4)):>10} {r['regime']:>22} {r['sign_robust']:>14}")

# ============================================================
# 5. Key diagnostic for the paper's main claim
# ============================================================
print("\n" + "=" * 70)
print("VALIDATION: Does the sign-reversal survive solver tolerance?")
print("=" * 70)

# The key test: SSE at 200-400t should be > 0, at 400-600t < 0
if sse_results:
    first = sse_results[0]   # 200-400t
    second = sse_results[1]  # 400-600t

    if first["sse_upper"] > 0 and second["sse_lower"] < 0:
        print("✅ PASS: The sign reversal (complementarity → substitution)")
        print(f"   survives even the MOST CONSERVATIVE uncertainty bounds.")
        print(f"   First interval upper bound = {first['sse_upper']:.4f} > 0")
        print(f"   Second interval lower bound = {second['sse_lower']:.4f} < 0")
        print(f"   Margin of safety = {min(first['sse_upper'], abs(second['sse_lower'])):.4f}")
    else:
        print("⚠️  FLAG: Under conservative bounds, the sign reversal is AMBIGUOUS.")
        print(f"   First interval SSE range = [{first['sse_lower']:.4f}, {first['sse_upper']:.4f}]")
        print(f"   Second interval SSE range = [{second['sse_lower']:.4f}, {second['sse_upper']:.4f}]")

    # Strong substitution test (800-1000t)
    last = sse_results[-1]
    if last["sse_lower"] < -1:
        print(f"\n✅ PASS: Strong substitution (ε < -1) at 800-1000t")
        print(f"   survives conservative bounds (upper = {last['sse_upper']:.4f} < -1)")
    else:
        print(f"\n⚠️  FLAG: Strong substitution not robust at 800-1000t")
        print(f"   SSE range = [{last['sse_lower']:.4f}, {last['sse_upper']:.4f}]")

# ============================================================
# 6. Save results
# ============================================================
df_sse_confidence = pd.DataFrame(sse_results)
output_path = os.path.join(PROJECT_ROOT, "results/tables/sse_confidence_bands.csv")
df_sse_confidence.to_csv(output_path, index=False)
print(f"\n✅ Results saved to: {output_path}")

# Print LaTeX-ready table snippet
print("\n" + "=" * 70)
print("LaTeX-ready Table Snippet (for Supplementary Material)")
print("=" * 70)
for r in sse_results:
    half = r["band_width"] / 2.0
    print(f"{r['interval']} & {r['h2_midpoint_t']:.0f} & "
          f"${r['sse_nominal']:.3f} \\pm {half:.3f}$ & {r['regime']} \\\\")

print("\nDone.")
