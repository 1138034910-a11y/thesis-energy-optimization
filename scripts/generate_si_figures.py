#!/usr/bin/env python3
"""Generate Supplementary Information figures archive with unified Nature style."""
import os, sys, json, numpy as np, pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)

from nature_style import apply, save_fig, C_BESS, C_ELC, C_FC, C_CARBON, C_REF, C_EV, C_CAP, C_WIND, C_PV, C_THERM

apply()

BASE = _project_root
TAB = os.path.join(BASE, "results", "tables")
FIG = os.path.join(BASE, "results", "figures_paper")
os.makedirs(FIG, exist_ok=True)

def mm_inch(w, h=None, ratio=0.618):
    wi = w / 25.4
    return (wi, wi * ratio if h is None else h / 25.4)

# Load data
cp = pd.read_csv(os.path.join(TAB, "carbon_price_sensitivity_CONSOLIDATED.csv"))
with open(os.path.join(TAB, "full_experiment_1000t.json"), 'r') as f:
    t1000 = json.load(f)

prices = cp["carbon_price_cny_per_ton"].values

# ================================================================
# SI Fig. S2: Absolute capacity vs carbon price
# ================================================================
fig, ax = plt.subplots(figsize=mm_inch(165, 95))

# BESS power
l1 = ax.plot(prices, cp["BESS_P_MW"], "s-", color=C_BESS, markersize=5,
             markerfacecolor="white", markeredgewidth=1.0, linewidth=1.5,
             label="BESS power", zorder=3)
# ELC
l2 = ax.plot(prices, cp["ELC_P_MW"], "^-", color=C_ELC, markersize=4.5,
             markerfacecolor="white", markeredgewidth=0.8, linewidth=1.2,
             label="Electrolyzer", zorder=3)
# FC
l3 = ax.plot(prices, cp["FC_P_MW"], "v-", color=C_FC, markersize=4.5,
             markerfacecolor="white", markeredgewidth=0.8, linewidth=1.2,
             label="Fuel cell", zorder=3)

ax.set_xlabel("Carbon price (CNY/t CO$_2$)", fontsize=9)
ax.set_ylabel("Power capacity (MW)", fontsize=9, color=C_CAP)
ax.tick_params(axis="y", labelcolor=C_CAP)
ax.set_ylim(0, max(cp["BESS_P_MW"].max(), cp["ELC_P_MW"].max(), cp["FC_P_MW"].max()) * 1.15)

# Twin axis for BESS energy
if "BESS_E_MWh" in cp.columns:
    ax2 = ax.twinx()
    ax2.spines["top"].set_visible(False)
    l4 = ax2.plot(prices, cp["BESS_E_MWh"], "o--", color=C_EV, markersize=4.5,
                  markerfacecolor="white", markeredgewidth=0.8, linewidth=1.2,
                  label="BESS energy", zorder=3)
    ax2.set_ylabel("BESS energy (MWh)", fontsize=9, color=C_EV)
    ax2.tick_params(axis="y", labelcolor=C_EV)
    ax2.set_ylim(0, cp["BESS_E_MWh"].max() * 1.15)

    # Merge legends and place outside right
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="center left",
              bbox_to_anchor=(1.18, 0.5), frameon=False)
else:
    ax.legend(fontsize=7, loc="upper left", frameon=False)

ax.set_title("SI Fig. S2: Absolute capacity vs carbon price", fontsize=10,
             fontweight="bold", pad=6, loc="left")
plt.tight_layout(pad=0.8)
save_fig(fig, FIG, "si_fig_s2_carbon_capacity_absolute")
plt.close()
print("  [OK] si_fig_s2_carbon_capacity_absolute")

# ================================================================
# SI Fig. S3: Emission trajectory
# ================================================================
fig, ax = plt.subplots(figsize=mm_inch(165, 95))

emissions = cp["carbon_emission_Mt"].values
ax.plot(prices, emissions, "o-", color=C_CARBON, markersize=5, linewidth=1.5,
        label="CO$_2$ emissions", zorder=3)
ax.axhline(y=15.0, color=C_EV, linewidth=0.8, linestyle="--", label="Carbon cap (15 Mt/yr)")

# Fill cap-binding vs non-binding
# Only highlight cap-binding region (emissions above cap)
ax.fill_between(prices, emissions, 15.0, where=(emissions >= 15.0),
                alpha=0.15, color=C_CARBON, interpolate=True, label="Cap-binding")

ax.set_xlabel("Carbon price (CNY/t CO$_2$)", fontsize=9)
ax.set_ylabel("Annual CO$_2$ emissions (Mt)", fontsize=9)
ax.set_ylim(0, max(emissions.max(), 16) * 1.08)
ax.legend(fontsize=7, loc="upper right", frameon=False)
ax.set_title("SI Fig. S3: Emission trajectory vs carbon price", fontsize=10,
             fontweight="bold", pad=6, loc="left")
plt.tight_layout(pad=0.8)
save_fig(fig, FIG, "si_fig_s3_emission_trajectory")
plt.close()
print("  [OK] si_fig_s3_emission_trajectory")

# ================================================================
# SI Fig. S4: BESS duration vs carbon price
# ================================================================
if "BESS_E_MWh" in cp.columns and "BESS_P_MW" in cp.columns:
    fig, ax = plt.subplots(figsize=mm_inch(165, 95))
    duration = cp["BESS_E_MWh"] / cp["BESS_P_MW"]

    ax.plot(prices, duration, "s-", color=C_BESS, markersize=5,
            markerfacecolor="white", markeredgewidth=1.0, linewidth=1.5,
            label="BESS duration", zorder=3)

    # Regime background zones (no overlapping text)
    ax.axvspan(0, 50, alpha=0.06, color=C_CARBON)
    ax.axvspan(50, prices.max(), alpha=0.06, color=C_BESS)
    ax.axvline(x=50, color=C_REF, linestyle=":", linewidth=0.7, alpha=0.5)

    # Zone labels placed at figure edges with enough padding
    ax.text(25, ax.get_ylim()[1] * 0.96, "Cap-driven\nregime", fontsize=7,
            color=C_CARBON, ha="center", va="top", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.8))
    ax.text(275, ax.get_ylim()[1] * 0.96, "Price-driven\nregime", fontsize=7,
            color=C_BESS, ha="center", va="top", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.8))

    ax.set_xlabel("Carbon price (CNY/t CO$_2$)", fontsize=9)
    ax.set_ylabel("BESS duration (h)", fontsize=9)
    ax.set_title("SI Fig. S4: BESS duration vs carbon price", fontsize=10,
                 fontweight="bold", pad=6, loc="left")
    plt.tight_layout(pad=0.8)
    save_fig(fig, FIG, "si_fig_s4_bess_duration")
    plt.close()
    print("  [OK] si_fig_s4_bess_duration")

# ================================================================
# SI Fig. S5: Weekly dispatch heatmap (168 h, Scenario 1)
# ================================================================
sc = t1000["tssp"]["scenarios"][0]
ds, de = 0, 168  # full week

# Compute BESS net
bess_net = (np.array(sc["bess_dis"][ds:de]) - np.array(sc["bess_ch"][ds:de])) / 1000.0

components = {
    "Wind": np.array(sc["wind_avail"][ds:de]) / 1000,
    "Solar": np.array(sc["pv_avail"][ds:de]) / 1000,
    "Thermal": np.array(sc["therm"][ds:de]) / 1000,
    "ELC": np.array(sc["elc"][ds:de]) / 1000,
    "FC": np.array(sc["fc"][ds:de]) / 1000,
    "BESS net": bess_net,
    "UHV": np.array(sc["uhv"][ds:de]) / 1000,
    "Curt.": np.array(sc["curt"][ds:de]) / 1000,
}

fig, ax = plt.subplots(figsize=mm_inch(180, 110))
mat = np.array(list(components.values()))

# Custom colormap: white -> warm tones
cmap = LinearSegmentedColormap.from_list("si_heatmap", ["#FFFFFF", "#FDE725", "#F58518", "#E45756"])
im = ax.imshow(mat, aspect="auto", cmap=cmap, interpolation="nearest")

ax.set_yticks(np.arange(len(components)))
ax.set_yticklabels(list(components.keys()), fontsize=8)

# X-axis: every 24 hours, label as Day
ax.set_xticks(np.arange(0, 168, 24))
ax.set_xticklabels([f"D{i+1}" for i in range(7)], fontsize=8)
ax.set_xlabel("Hour of week (Day 1–7)", fontsize=9)

ax.set_title("SI Fig. S5: Weekly dispatch heatmap (Scenario 1, 168 h)", fontsize=10,
             fontweight="bold", pad=6, loc="left")

# Colorbar with generous padding to avoid overlap
cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.04)
cbar.set_label("Power (GW)", fontsize=8)
cbar.ax.tick_params(labelsize=7)

plt.tight_layout(pad=0.8)
save_fig(fig, FIG, "si_fig_s5_dispatch_heatmap")
plt.close()
print("  [OK] si_fig_s5_dispatch_heatmap")

print("\nAll SI figures archive generated!")
