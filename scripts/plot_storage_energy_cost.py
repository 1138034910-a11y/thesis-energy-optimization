"""
Levelized energy-storage cost comparison: BESS vs hydrogen.

Shows that hydrogen electrical-equivalent storage is roughly an order of magnitude
cheaper per MWh than BESS energy capacity, explaining why hydrogen becomes the
lower-cost option for long-duration deficit coverage at intermediate carbon prices.

Output:
  results/figures_paper/si_fig_s10_storage_energy_cost.pdf
  results/figures_paper/si_fig_s10_storage_energy_cost.png
  results/tables/storage_energy_cost_comparison.csv
"""
import os
import sys
import numpy as np
import pandas as pd

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.nature_style import apply, C_BESS, C_ELC, save_fig

apply()

# ============================================================
# Parameters
# ============================================================
r = 0.06
def crf(r, n):
    return r * (1 + r)**n / ((1 + r)**n - 1)

CRF_bess = crf(r, 15)
CRF_tank = crf(r, 20)

C_bess_e = 120.0      # 10^4 CNY / MWh
C_tank = 0.18         # 10^4 CNY / kg
eta_fc_mwh_per_kg = 0.0165  # MWh electricity per kg H2

# Hydrogen energy-equivalent cost per MWh stored electricity
# 1 kg H2 stores 0.0165 MWh electricity -> 1 MWh = 60.606 kg
C_h2_per_mwh = C_tank / eta_fc_mwh_per_kg  # 10^4 CNY / MWh electrical equivalent

# ============================================================
# Cost vs duration
# ============================================================
durations = np.array([2, 4, 8, 12, 24, 48, 72, 120, 168])  # hours

# BESS: cost = C_bess_e per MWh (energy cost dominates for daily-weekly duration)
bess_cost = C_bess_e * np.ones_like(durations)  # 10^4 CNY / MWh capacity
bess_annual = CRF_bess * bess_cost

# H2: cost per MWh of electrical-equivalent storage
h2_cost = C_h2_per_mwh * np.ones_like(durations)  # tank cost only
h2_annual = CRF_tank * h2_cost

# ============================================================
# Table
# ============================================================
df = pd.DataFrame({
    "duration_h": durations,
    "bess_capex_10k_cny_per_mwh": bess_cost,
    "bess_annual_10k_cny_per_mwh_per_year": bess_annual,
    "h2_capex_10k_cny_per_mwh": h2_cost,
    "h2_annual_10k_cny_per_mwh_per_year": h2_annual,
    "ratio_bess_to_h2": bess_cost / h2_cost,
})
df.to_csv("results/tables/storage_energy_cost_comparison.csv", index=False, encoding="utf-8-sig")
print(df.to_string(index=False))

# ============================================================
# Plot
# ============================================================
fig, ax = plt.subplots(figsize=(5.5, 4.0))

ax.plot(durations, bess_cost, marker='o', color=C_BESS, linewidth=1.5,
        markersize=5, label="BESS energy capacity")
ax.plot(durations, h2_cost, marker='s', color=C_ELC, linewidth=1.5,
        markersize=5, label="H2 electrical-equivalent storage")

ax.set_xlabel("Storage duration (h)")
ax.set_ylabel("Energy capacity cost (10$^4$ CNY / MWh)")
ax.set_title("Energy-storage cost: BESS vs hydrogen")
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(2, 168)
ax.set_ylim(5, 300)
ax.legend(frameon=False, loc="upper right")

# Annotate order-of-magnitude difference
ax.annotate("BESS energy cost is\n~11× hydrogen storage cost",
            xy=(48, C_bess_e), xytext=(12, 150),
            arrowprops=dict(arrowstyle="->", color="#333333", lw=0.6),
            fontsize=8, color="#333333")

plt.tight_layout()

note = ("Hydrogen cost is tank-only electrical-equivalent storage (1 kg H2 = 0.0165 MWh). "
        "Power conversion (electrolyzer, fuel cell) and round-trip efficiency are excluded; "
        "the comparison isolates the energy-capacity cost advantage that makes hydrogen attractive for long-duration storage.")
from scripts.nature_style import add_note
add_note(fig, note, ypos=-0.02)

save_fig(fig, "results/figures_paper", "si_fig_s10_storage_energy_cost")

print("Done.")
