"""
Generate operational evidence for the battery-hydrogen encroachment mechanism.

Reads existing full-experiment JSON files (400 t and 1,000 t, zero carbon price)
and produces a multi-panel figure comparing:
  (a) BESS net-discharge duration curve
  (b) BESS SOC distribution
  (c) Fuel-cell discharge duration curve
  (d) Hydrogen tank level distribution

Output:
  results/figures_paper/si_fig_s8_encroachment_evidence.pdf
  results/figures_paper/si_fig_s8_encroachment_evidence.png
"""
import os
import sys
import json
import numpy as np

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.nature_style import apply, C_BESS, C_FC, C_ELC, save_fig

apply()

# ============================================================
# Load data
# ============================================================
FILE_400 = "results/tables/full_experiment_v3.json"
FILE_1000 = "results/tables/full_experiment_1000t.json"

with open(FILE_400) as f:
    data_400 = json.load(f)
with open(FILE_1000) as f:
    data_1000 = json.load(f)

def aggregate_scenarios(tssp_data, var_key, weight_key="weight"):
    """Aggregate a variable across scenarios using scenario weights."""
    scenarios = tssp_data["tssp"]["scenarios"]
    weights = np.array([s[weight_key] for s in scenarios])
    weights = weights / weights.sum()
    arrays = [np.array(s[var_key]) for s in scenarios]
    # Weighted average across scenarios
    agg = sum(w * arr for w, arr in zip(weights, arrays))
    return agg

def duration_curve(values):
    """Return sorted values (descending) and corresponding cumulative hours."""
    sorted_vals = np.sort(values)[::-1]
    hours = np.arange(1, len(sorted_vals) + 1)
    return sorted_vals, hours

# ============================================================
# Extract variables
# ============================================================
cap_400 = data_400["tssp"]["capacity"]
cap_1000 = data_1000["tssp"]["capacity"]

bess_e_400 = aggregate_scenarios(data_400, "bess_e")
bess_ch_400 = aggregate_scenarios(data_400, "bess_ch")
bess_dis_400 = aggregate_scenarios(data_400, "bess_dis")
net_bess_400 = bess_dis_400 - bess_ch_400
fc_400 = aggregate_scenarios(data_400, "fc")
h_tank_400 = aggregate_scenarios(data_400, "h_tank")

bess_e_1000 = aggregate_scenarios(data_1000, "bess_e")
bess_ch_1000 = aggregate_scenarios(data_1000, "bess_ch")
bess_dis_1000 = aggregate_scenarios(data_1000, "bess_dis")
net_bess_1000 = bess_dis_1000 - bess_ch_1000
fc_1000 = aggregate_scenarios(data_1000, "fc")
h_tank_1000 = aggregate_scenarios(data_1000, "h_tank")

# Normalize
soc_400 = bess_e_400 / cap_400["BESS_E_MWh"]
soc_1000 = bess_e_1000 / cap_1000["BESS_E_MWh"]
h_norm_400 = h_tank_400 / cap_400["H2_Tank_kg"]
h_norm_1000 = h_tank_1000 / cap_1000["H2_Tank_kg"]

# Duration curves
bess_dc_400, h_400 = duration_curve(net_bess_400)
bess_dc_1000, h_1000 = duration_curve(net_bess_1000)
fc_dc_400, _ = duration_curve(fc_400)
fc_dc_1000, _ = duration_curve(fc_1000)

# ============================================================
# Plot
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(7.48, 5.5))

# Panel (a): BESS net-discharge duration curve
ax = axes[0, 0]
ax.plot(h_400, bess_dc_400 / 1e3, color=C_BESS, linewidth=1.2, label="400 t H2")
ax.plot(h_1000, bess_dc_1000 / 1e3, color=C_BESS, linewidth=1.2, linestyle="--", label="1,000 t H2")
ax.axhline(0, color="#888888", linewidth=0.5, linestyle="-")
ax.set_xlabel("Hours (descending)")
ax.set_ylabel("BESS net discharge (GW)")
ax.set_title("(a) BESS net-discharge duration curve")
ax.legend(frameon=False)
ax.set_xlim(0, 480)

# Panel (b): BESS SOC distribution
ax = axes[0, 1]
ax.hist(soc_400, bins=30, density=True, alpha=0.6, color=C_BESS, label="400 t H2")
ax.hist(soc_1000, bins=30, density=True, alpha=0.6, color=C_BESS, edgecolor="white",
        linestyle="--", histtype="step", linewidth=1.5, label="1,000 t H2")
ax.set_xlabel("BESS SOC / energy capacity")
ax.set_ylabel("Density")
ax.set_title("(b) BESS SOC distribution")
ax.legend(frameon=False)

# Panel (c): Fuel-cell discharge duration curve
ax = axes[1, 0]
ax.plot(h_400, fc_dc_400 / 1e3, color=C_FC, linewidth=1.2, label="400 t H2")
ax.plot(h_1000, fc_dc_1000 / 1e3, color=C_FC, linewidth=1.2, linestyle="--", label="1,000 t H2")
ax.set_xlabel("Hours (descending)")
ax.set_ylabel("Fuel-cell output (GW)")
ax.set_title("(c) Fuel-cell discharge duration curve")
ax.legend(frameon=False)
ax.set_xlim(0, 480)

# Panel (d): Hydrogen tank level distribution
ax = axes[1, 1]
ax.hist(h_norm_400, bins=30, density=True, alpha=0.6, color=C_ELC, label="400 t H2")
ax.hist(h_norm_1000, bins=30, density=True, alpha=0.6, color=C_ELC, edgecolor="white",
        linestyle="--", histtype="step", linewidth=1.5, label="1,000 t H2")
ax.set_xlabel("Hydrogen tank level / tank capacity")
ax.set_ylabel("Density")
ax.set_title("(d) Hydrogen tank level distribution")
ax.legend(frameon=False)

plt.tight_layout()

note = ("Dispatch profiles are scenario-weighted averages from the zero-carbon-price TSSP solutions. "
        "Capacities: 400 t H2 (BESS {0:.0f} MW / {1:.0f} MWh), 1,000 t H2 (BESS {2:.0f} MW / {3:.0f} MWh)."
        .format(cap_400["BESS_P_MW"], cap_400["BESS_E_MWh"],
                cap_1000["BESS_P_MW"], cap_1000["BESS_E_MWh"]))

from scripts.nature_style import add_note
add_note(fig, note, ypos=-0.02)

save_fig(fig, "results/figures_paper", "si_fig_s8_encroachment_evidence")

print("Done.")
