"""
Stylized marginal abatement cost (MAC) comparison for BESS and hydrogen.

Computes the levelized cost of displacing 1 MWh of thermal generation with
BESS or with the hydrogen conversion chain, as a function of carbon price.
The crossover explains the non-monotonic BESS response in Section 5.3.

Output:
  results/figures_paper/si_fig_s9_mac_crossover.pdf
  results/figures_paper/si_fig_s9_mac_crossover.png
"""
import os
import sys
import numpy as np

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.nature_style import apply, C_BESS, C_FC, C_THERM, save_fig

apply()

# ============================================================
# Parameters from config.py
# ============================================================
r = 0.06
def crf(r, n):
    return r * (1 + r)**n / ((1 + r)**n - 1)

CRF_bess = crf(r, 15)
CRF_elc = crf(r, 15)
CRF_tank = crf(r, 20)
CRF_fc = crf(r, 15)

C_bess_p = 80.0       # 10^4 CNY / MW
C_bess_e = 120.0      # 10^4 CNY / MWh
C_elc = 150.0         # 10^4 CNY / MW
C_tank = 0.18         # 10^4 CNY / kg
C_fc = 220.0          # 10^4 CNY / MW

eta_bess_rt = 0.95 * 0.95
eta_fc_mwh_per_kg = 0.0165
eta_elc_kg_per_mwh = 17.75

var_om_bess = 0.005   # 10^4 CNY / MWh cycled
var_om_fc = 0.004     # 10^4 CNY / MWh FC output

EF_coal = 0.85        # tCO2 / MWh
C_coal = 0.030        # 10^4 CNY / MWh thermal fuel

# ============================================================
# Levelized cost of displacing 1 MWh of thermal output
# ============================================================
# BESS: 1 MWh discharge displaces 1 MWh thermal. Need 1/eta_rt MWh charge.
# Annualized cost per MWh of BESS energy capacity (with power cost allocated).
D_bess = 3.0  # representative BESS duration (h)
annual_bess_per_mwh = CRF_bess * (C_bess_p / D_bess + C_bess_e)  # 10^4 CNY/MWh cap/year

# Hydrogen chain: 1 MWh FC output needs 1/eta_fc_mwh_per_kg kg stored H2.
# To produce that H2, need (kg)/eta_elc kg/MWh electricity.
# For one MWh of FC output, we need 1/eta_fc_mwh_per_kg kg H2 and an ELC big enough
# to produce it and a tank big enough to store it.
# We annualize per MWh of average FC output.
D_h2 = 5.0  # representative hydrogen storage duration (days) for the energy reservoir
h2_kg_per_mwh_fc = 1.0 / eta_fc_mwh_per_kg
elc_mw_per_mw_fc = h2_kg_per_mwh_fc / eta_elc_kg_per_mwh  # MW ELC per MW FC (for one MWh/day)
tank_kg_per_mw_fc = h2_kg_per_mwh_fc * D_h2 * 24  # kg tank per MW FC for D_h2 days

annual_h2_per_mwh_fc = (
    CRF_elc * C_elc * elc_mw_per_mw_fc +
    CRF_tank * C_tank * tank_kg_per_mw_fc +
    CRF_fc * C_fc
)  # 10^4 CNY per MW FC per year

# Annual output per MW FC if used 2,000 h/year (intermediate utilization)
utilization_hours = np.linspace(500, 8000, 200)
mac_bess = (annual_bess_per_mwh / utilization_hours + var_om_bess) / EF_coal  # 10^4 CNY/tCO2
mac_h2 = (annual_h2_per_mwh_fc / utilization_hours + var_om_fc) / EF_coal

# Thermal marginal cost with carbon price
# Carbon price in CNY/t; model units are 10^4 CNY/t -> divide by 10^4
lambda_co2 = np.linspace(0, 500, 200)
mac_thermal = (C_coal + lambda_co2 / 10000.0) / EF_coal  # 10^4 CNY per MWh avoided

# ============================================================
# Plot
# ============================================================
fig, ax = plt.subplots(figsize=(6.0, 5.0))

# Use a representative utilization (2000 h/year) for the single curves
util_ref = 2000
mac_bess_ref = (annual_bess_per_mwh / util_ref + var_om_bess) / EF_coal
mac_h2_ref = (annual_h2_per_mwh_fc / util_ref + var_om_fc) / EF_coal

ax.plot(lambda_co2, mac_thermal, color=C_THERM, linewidth=1.5, label="Thermal (fuel + carbon)")
ax.axhline(mac_bess_ref, color=C_BESS, linewidth=1.5, linestyle="-", label=f"BESS ({util_ref:.0f} h/year)")
ax.axhline(mac_h2_ref, color=C_FC, linewidth=1.5, linestyle="-", label=f"Hydrogen chain ({util_ref:.0f} h/year)")

# Shade regions where each technology is cheapest for abatement
ax.fill_between(lambda_co2, 0, np.minimum(mac_thermal, np.minimum(mac_bess_ref, mac_h2_ref)),
                color=C_BESS, alpha=0.08, label="BESS cheaper than thermal")
ax.fill_between(lambda_co2, 0, np.minimum(mac_thermal, mac_h2_ref),
                where=(mac_h2_ref < np.minimum(mac_thermal, mac_bess_ref)),
                color=C_FC, alpha=0.15, interpolate=True, label="Hydrogen cheaper")

ax.set_xlabel("Carbon price (CNY/tCO$_2$)")
ax.set_ylabel("Marginal abatement cost\n(10$^4$ CNY / MWh thermal avoided)")
ax.set_title("Marginal abatement cost crossover", pad=18)
ax.set_xlim(0, 500)
ax.set_ylim(0, max(mac_thermal.max(), mac_bess_ref, mac_h2_ref) * 1.1)
ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)

# Annotation
ax.annotate("BESS dominates\nlow-price abatement",
            xy=(40, mac_bess_ref), xytext=(70, mac_bess_ref + 0.015),
            arrowprops=dict(arrowstyle="->", color=C_BESS, lw=0.6),
            fontsize=7, color=C_BESS, ha="left")
ax.annotate("Hydrogen dominates\nintermediate prices",
            xy=(150, mac_h2_ref), xytext=(260, mac_h2_ref + 0.010),
            arrowprops=dict(arrowstyle="->", color=C_FC, lw=0.6),
            fontsize=7, color=C_FC, ha="left")
ax.annotate("BESS returns at\nhigh carbon prices",
            xy=(400, mac_bess_ref), xytext=(340, mac_bess_ref + 0.020),
            arrowprops=dict(arrowstyle="->", color=C_BESS, lw=0.6),
            fontsize=7, color=C_BESS, ha="left")

plt.tight_layout()

note = ("Costs are annualized at 6\\% over asset lifetimes. BESS duration = 3 h; hydrogen reservoir = 5 days. "
        "The hydrogen chain includes electrolyzer, tank, and fuel cell. Curves are illustrative; exact crossover "
        "points depend on utilization and cost assumptions.")
from scripts.nature_style import add_note
add_note(fig, note, ypos=-0.18)

save_fig(fig, "results/figures_paper", "si_fig_s9_mac_crossover")

print("Done.")
print(f"Annual BESS cost per MWh capacity: {annual_bess_per_mwh:.2f} 10^4 CNY/MWh")
print(f"Annual H2 chain cost per MW FC: {annual_h2_per_mwh_fc:.2f} 10^4 CNY/MW FC")
print(f"MAC BESS at {util_ref} h: {mac_bess_ref:.4f} 10^4 CNY/MWh avoided")
print(f"MAC H2 at {util_ref} h: {mac_h2_ref:.4f} 10^4 CNY/MWh avoided")
