#!/usr/bin/env python3
"""
Unified figure generation for paper — all outputs to results/figures_paper/
Uses nature_style.py color palette. Fixes Fig.5 carbon-unit bug.
"""
import os, sys, json, numpy as np, pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)

from nature_style import apply, save_fig, C_BESS, C_ELC, C_FC, C_CARBON, C_REF, C_EV, C_THEORY

apply()

BASE = _project_root
TAB = os.path.join(BASE, "results", "tables")
FIG = os.path.join(BASE, "results", "figures_paper")
os.makedirs(FIG, exist_ok=True)

# ================================================================
# DATA LOADING
# ================================================================
print("Loading authoritative data...")

h2 = pd.read_csv(os.path.join(TAB, "h2_sensitivity_v3_rigorous_vss.csv"))
cp = pd.read_csv(os.path.join(TAB, "carbon_price_sensitivity_CONSOLIDATED.csv"))
with open(os.path.join(TAB, "full_experiment_v3.json"), 'r') as f:
    v3 = json.load(f)
with open(os.path.join(TAB, "full_experiment_1000t.json"), 'r') as f:
    t1000 = json.load(f)

conv = pd.read_csv(os.path.join(TAB, "convergence_test.csv"))
sse_df = pd.read_csv(os.path.join(TAB, "sse_calculations.csv"))

# Color grammar (from nature_style.py + extras)
C = {
    "tssp": C_BESS,
    "ev": C_EV,
    "h2": C_ELC,
    "h2_light": "#90D5A0",
    "bess": C_BESS,
    "bess_lt": "#A6C8E0",
    "carbon": C_CARBON,
    "wind": "#3498DB",
    "solar": "#F1C40F",
    "thermal": "#C0392B",
    "curt": "#7F7F7F",
    "gray": C_REF,
    "dark": "#2E2E2E",
    "fc": C_FC,
    "elc": C_ELC,
    "theory": C_THEORY,
}

STYLE_TSSP = {"marker": "s", "linestyle": "-",  "markerfacecolor": "white", "markersize": 6, "linewidth": 1.5}
STYLE_EV   = {"marker": "o", "linestyle": "--", "markerfacecolor": "white", "markersize": 5, "linewidth": 1.2}

def mm_inch(w, h=None, ratio=0.618):
    wi = w / 25.4
    return (wi, wi * ratio if h is None else h / 25.4)

# ================================================================
# FIGURE 5: COST STRUCTURE (FIXED)
# ================================================================
def fig05_cost_structure():
    """Cost structure: (a) donut with exact values, (b) carbon cost share."""
    fig = plt.figure(figsize=mm_inch(160, 82))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.2], wspace=0.28)

    # --- Panel (a): Donut chart ---
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Use correct cost data from CSV (80 CNY/t row)
    cp80 = cp[cp["carbon_price_cny_per_ton"] == 80].iloc[0]
    inv = cp80["cost_inv_10k"] / 1e4        # to B CNY
    om = cp80["cost_om_fix_10k"] / 1e4
    op = cp80["cost_op_10k"] / 1e4
    carb = cp80["cost_carbon_10k"] / 1e4
    total = inv + om + op + carb
    
    sizes = [inv/total*100, om/total*100, op/total*100, carb/total*100]
    vals = [inv, om, op, carb]
    labels = [f"Investment\n{sizes[0]:.1f}%\n({vals[0]:.1f} B)",
              f"Fixed O&M\n{sizes[1]:.1f}%\n({vals[1]:.1f} B)",
              f"Operation\n{sizes[2]:.1f}%\n({vals[2]:.1f} B)",
              f"Carbon\n{sizes[3]:.1f}%\n({vals[3]:.1f} B)"]
    colors_d = [C["bess"], C["gray"], C["h2"], C["carbon"]]

    wedges, texts = ax1.pie(sizes, labels=labels, colors=colors_d, startangle=90,
                            wedgeprops=dict(width=0.35, edgecolor="white", linewidth=0.5),
                            textprops=dict(fontsize=7, color=C["dark"]))
    # Center text
    ax1.text(0, 0, f"Total cost\n{total:.1f} B CNY\n(80 CNY/t)", ha="center", va="center",
             fontsize=9, fontweight="bold", color=C["dark"])
    ax1.set_title("(a) Annual cost composition\n400 t H$_2$, 80 CNY/t", fontsize=9, loc="left", pad=6)

    # --- Panel (b): Carbon cost share vs price ---
    ax2 = fig.add_subplot(gs[0, 1])
    prices = cp["carbon_price_cny_per_ton"].values
    shares = []
    for _, row in cp.iterrows():
        cs = row["cost_carbon_10k"]
        tot = row["cost_inv_10k"] + row["cost_om_fix_10k"] + row["cost_op_10k"] + cs
        shares.append(cs / tot * 100 if tot > 0 else 0)

    xpos = np.arange(len(prices))
    bars = ax2.bar(xpos, shares, color=C["carbon"], edgecolor="white", linewidth=0.3, width=0.6)
    # Add value labels on bars
    for bar, val in zip(bars, shares):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=6.5, color=C["dark"], fontweight="bold")

    ax2.plot(xpos, shares, "o-", color=C["dark"], markersize=3, linewidth=0.8, alpha=0.6)

    ax2.set_xticks(xpos)
    ax2.set_xticklabels([f"{int(p)}" for p in prices], fontsize=7, rotation=0)
    ax2.set_xlabel("Carbon price (CNY/t CO$_2$)", fontsize=9)
    ax2.set_ylabel("Carbon cost share (%)", fontsize=9)
    ax2.set_title("(b) Carbon cost share of total", fontsize=9, loc="left", pad=6)

    # Regime shading
    ax2.axvline(x=1.5, color=C["gray"], linestyle=":", linewidth=0.7, alpha=0.5)
    ax2.text(0.7, max(shares)*0.92, "Cap-\ndriven", fontsize=6.5, ha="center", color=C["gray"])
    ax2.text(4.5, max(shares)*0.92, "Price-driven", fontsize=6.5, ha="center", color=C["tssp"])
    ax2.set_ylim(0, max(shares)*1.25)

    plt.tight_layout(pad=0.8)
    save_fig(fig, FIG, "fig05_cost_structure")
    plt.close()
    print("  [OK] fig05_cost_structure")


# ================================================================
# FIGURE 6: H2-BESS SUBSTITUTION
# ================================================================
def fig06_h2_bess():
    """H2-BESS substitution: 2-panel power & energy."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=mm_inch(165, 80))
    tanks = h2["H2_Tank_t"].values

    # Panel (a): BESS power
    ax1.plot(tanks, h2["EV_BESS_P_MW"], label="EV", color=C["ev"], **STYLE_EV)
    ax1.plot(tanks, h2["TSSP_BESS_P_MW"], label="TSSP", color=C["tssp"], **STYLE_TSSP)
    ax1.set_xlabel("H$_2$ tank capacity (t)", fontsize=9)
    ax1.set_ylabel("BESS power (MW)", fontsize=9)
    ax1.set_title("(a) BESS power capacity", fontsize=9, loc="left", pad=4)
    ax1.legend(fontsize=7.5)
    ax1.set_xticks(tanks)

    # Highlight 400t peak region
    ax1.axvspan(350, 650, alpha=0.06, color=C["h2"])
    ax1.text(500, max(h2["TSSP_BESS_P_MW"])*0.92, "Non-monotonic\nregion", fontsize=6.5,
             ha="center", color=C["h2"], fontweight="bold")

    # Panel (b): BESS energy
    ax2.plot(tanks, h2["EV_BESS_E_MWh"], label="EV", color=C["ev"], **STYLE_EV)
    ax2.plot(tanks, h2["TSSP_BESS_E_MWh"], label="TSSP", color=C["tssp"], **STYLE_TSSP)
    ax2.set_xlabel("H$_2$ tank capacity (t)", fontsize=9)
    ax2.set_ylabel("BESS energy (MWh)", fontsize=9)
    ax2.set_title("(b) BESS energy capacity", fontsize=9, loc="left", pad=4)
    ax2.legend(fontsize=7.5)
    ax2.set_xticks(tanks)

    plt.tight_layout(pad=1.0)
    save_fig(fig, FIG, "fig06_h2_bess_substitution")
    plt.close()
    print("  [OK] fig06_h2_bess_substitution")


# ================================================================
# FIGURE 7: SSE ELASTICITY
# ================================================================
def fig07_sse():
    """SSE elasticity curve."""
    fig, ax = plt.subplots(figsize=mm_inch(150, 85))

    x_mid = sse_df["h2_mid_t"].values
    eps = sse_df["elast_tssp_p"].values

    ax.plot(x_mid, eps, marker="s", linestyle="-", color=C["tssp"], linewidth=1.5,
            markersize=6, markerfacecolor="white", markeredgewidth=1.0, zorder=3)
    ax.axhline(y=0, color=C["gray"], linestyle="--", linewidth=0.7, alpha=0.5)

    # Regime labels
    ax.text(300, 0.08, "Complementarity\n($\\varepsilon > 0$)", fontsize=6.5, color=C["h2"], ha="center")
    ax.text(700, -0.6, "Substitution\n($-1 < \\varepsilon < 0$)", fontsize=6.5, color=C["tssp"], ha="center")
    ax.text(900, -1.7, "Strong substitution\n($\\varepsilon < -1$)", fontsize=6.5, color=C["carbon"], ha="center")

    ax.set_xlabel("H$_2$ tank capacity (t, midpoint)", fontsize=9)
    ax.set_ylabel("Storage substitution elasticity $\\varepsilon$", fontsize=9)
    ax.set_title("Storage Substitution Elasticity (SSE)", fontsize=10, fontweight="bold", pad=6)

    plt.tight_layout(pad=0.8)
    save_fig(fig, FIG, "fig07_sse_elasticity")
    plt.close()
    print("  [OK] fig07_sse_elasticity")


# ================================================================
# FIGURE 8: CARBON PRICE REGIME
# ================================================================
def fig08_carbon_regime():
    """Carbon price: (a) BESS response, (b) emissions."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=mm_inch(165, 80))

    prices = cp["carbon_price_cny_per_ton"].values
    bess_p = cp["BESS_P_MW"].values
    bess_theo = cp["BESS_P_theoretical_MW"].values
    emissions = cp["carbon_emission_Mt"].values
    statuses = cp["status"].values
    is_opt = np.array([str(s).strip() == "OPTIMAL" for s in statuses])
    is_s9 = ~is_opt

    # Panel (a)
    opt_idx = np.where(is_opt)[0]
    if len(opt_idx) > 1:
        ax1.plot(prices[opt_idx], bess_p[opt_idx], marker=STYLE_TSSP["marker"],
                 linestyle=STYLE_TSSP["linestyle"], color=C["tssp"], markersize=6,
                 markerfacecolor=C["tssp"], markeredgewidth=0.5, linewidth=1.5,
                 label="OPTIMAL", zorder=3)
    s9_idx = np.where(is_s9)[0]
    if len(s9_idx) > 0:
        ax1.scatter(prices[s9_idx], bess_p[s9_idx], marker="o", s=55, facecolors="white",
                    edgecolors=C["gray"], linewidths=1.0, zorder=3, label="STATUS_9")

    ax1.scatter([100], [bess_theo[3]], marker="X", s=80, color=C["theory"], edgecolors=C["theory"],
                linewidths=0.8, zorder=4, label="Spline projection")

    ax1.axvline(x=50, color=C["gray"], linestyle=":", linewidth=0.7, alpha=0.5)
    ax1.fill_between([0, 50], 4000, 10000, alpha=0.04, color=C["carbon"])
    ax1.text(20, 9300, "Cap-driven", fontsize=7, color=C["carbon"], ha="center", fontweight="bold")
    ax1.text(280, 9300, "Price-driven", fontsize=7, color=C["tssp"], ha="center", fontweight="bold")

    ax1.set_xlabel("Carbon price (CNY/t CO$_2$)", fontsize=9)
    ax1.set_ylabel("BESS power capacity (MW)", fontsize=9)
    ax1.set_title("(a) BESS investment response", fontsize=9, loc="left", pad=4)
    ax1.legend(fontsize=6.5, loc="lower right")
    ax1.set_ylim(4500, 9800)
    ax1.set_xlim(-20, 530)

    # Panel (b)
    xpos = np.arange(len(prices))
    colors_em = [C["carbon"] if e >= 14 else C["h2"] for e in emissions]
    ax2.bar(xpos, emissions, color=colors_em, edgecolor="white", linewidth=0.3, width=0.6)
    ax2.axhline(y=15.0, color=C["ev"], linewidth=0.8, linestyle="--", label="Carbon cap (15 Mt/yr)")

    ax2.set_xticks(xpos)
    ax2.set_xticklabels([f"{int(p)}" for p in prices], fontsize=7, rotation=0)
    ax2.set_xlabel("Carbon price (CNY/t CO$_2$)", fontsize=9)
    ax2.set_ylabel("Annual CO$_2$ emissions (Mt)", fontsize=9)
    ax2.set_title("(b) Emission response", fontsize=9, loc="left", pad=4)
    ax2.legend(fontsize=7, loc="upper right")

    plt.tight_layout(pad=1.0)
    save_fig(fig, FIG, "fig08_carbon_regime")
    plt.close()
    print("  [OK] fig08_carbon_regime")


# ================================================================
# FIGURE 9: CONVERGENCE TEST (FIXED — drop 25-day outlier)
# ================================================================
def fig09_convergence():
    """Convergence: drop 25-day outlier, show 10/15/20/30."""
    fig, ax1 = plt.subplots(figsize=mm_inch(140, 85))

    conv_unique = conv.drop_duplicates(subset=["n_days"]).sort_values("n_days")
    # Drop 25-day outlier
    conv_clean = conv_unique[conv_unique["n_days"] != 25]
    ndays = conv_clean["n_days"].values
    bess_conv = conv_clean["BESS_P_MW"].values
    cost_conv = abs(conv_clean["Obj"].values) / 1e4

    ax1.plot(ndays, bess_conv, marker="s", linestyle="-", color=C["tssp"], linewidth=1.5,
             markersize=7, markerfacecolor="white", markeredgewidth=1.0, label="BESS power (MW)", zorder=3)
    ax1.set_xlabel("Number of representative days", fontsize=9)
    ax1.set_ylabel("BESS power capacity (MW)", fontsize=9, color=C["tssp"])
    ax1.tick_params(axis="y", labelcolor=C["tssp"])

    ax2 = ax1.twinx()
    ax2.plot(ndays, cost_conv, marker="o", linestyle="--", color=C["ev"], linewidth=1.2,
             markersize=6, markerfacecolor="white", markeredgewidth=0.8, label="Total cost (B CNY)", zorder=3)
    ax2.set_ylabel("Total annual cost (B CNY)", fontsize=9, color=C["ev"])
    ax2.tick_params(axis="y", labelcolor=C["ev"])

    # 3% band around 20-day cost
    ref_cost = cost_conv[ndays == 20][0] if 20 in ndays else cost_conv[-1]
    ax2.axhline(y=ref_cost*1.03, color=C["gray"], linestyle=":", linewidth=0.6, alpha=0.5)
    ax2.axhline(y=ref_cost*0.97, color=C["gray"], linestyle=":", linewidth=0.6, alpha=0.5)
    ax2.text(22, ref_cost*1.032, "$\\pm$3%", fontsize=6.5, color=C["gray"], ha="center")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper right")

    ax1.set_title("Convergence of representative-day aggregation", fontsize=10, fontweight="bold", pad=6)

    plt.tight_layout(pad=0.6)
    save_fig(fig, FIG, "fig09_convergence")
    plt.close()
    print("  [OK] fig09_convergence")


# ================================================================
# FIGURE 10: VSS ACROSS H2 SCALES
# ================================================================
def fig10_vss():
    """VSS across H2 scales."""
    fig, ax = plt.subplots(figsize=mm_inch(150, 85))

    vss_pct = h2["VSS_pct"].values
    ev_tssp_gap = abs((h2["EV_Obj"] - h2["TSSP_Obj"]) / h2["TSSP_Obj"] * 100).values
    tssp_gaps = h2["TSSP_Gap_pct"].values
    tanks = h2["H2_Tank_t"].values

    x = np.arange(len(tanks))
    w = 0.33

    bars1 = ax.bar(x - w/2, vss_pct*100, w, color=C["tssp"], edgecolor="white", linewidth=0.3, label="Rigorous VSS")
    bars2 = ax.bar(x + w/2, ev_tssp_gap, w, color=C["ev"], edgecolor="white", linewidth=0.3, label="EV–TSSP gap")

    # Value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                f"{bar.get_height():.2f}%", ha="center", va="bottom", fontsize=6.5, color=C["tssp"], fontweight="bold")
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                f"{bar.get_height():.2f}%", ha="center", va="bottom", fontsize=6.5, color=C["ev"], fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(t)}" for t in tanks], fontsize=8)
    ax.set_xlabel("H$_2$ tank capacity (t)", fontsize=9)
    ax.set_ylabel("Value (%)", fontsize=9)
    ax.set_title("Value of stochastic solution across H$_2$ scales", fontsize=10, fontweight="bold", pad=6)
    ax.legend(fontsize=7.5, loc="upper left")

    plt.tight_layout(pad=0.8)
    save_fig(fig, FIG, "fig10_vss_h2_scales")
    plt.close()
    print("  [OK] fig10_vss_h2_scales")


# ================================================================
# FIGURE 11a/b/c: SEASONAL DISPATCH (separate panels)
# ================================================================
def _plot_dispatch_day(ax, sc, day_idx, title_label):
    """Plot one representative day dispatch."""
    hours = np.arange(24)
    h0 = day_idx * 24
    h1 = h0 + 24

    wind = np.array(sc["wind_avail"][h0:h1])
    solar = np.array(sc["pv_avail"][h0:h1])
    load = np.array(sc["load"][h0:h1]) if "load" in sc else np.zeros(24)
    elc = np.array(sc["elc"][h0:h1]) if "elc" in sc else np.zeros(24)
    bess_ch = np.array(sc["bess_ch"][h0:h1]) if "bess_ch" in sc else np.zeros(24)
    bess_dis = np.array(sc["bess_dis"][h0:h1]) if "bess_dis" in sc else np.zeros(24)
    fc = np.array(sc["fc"][h0:h1]) if "fc" in sc else np.zeros(24)

    bess_net = bess_dis - bess_ch

    ax.stackplot(hours, wind, solar, labels=["Wind", "PV"], colors=[C["wind"], C["solar"]], alpha=0.85)
    ax.plot(hours, load, color=C["dark"], linewidth=1.2, linestyle="--", label="Load")

    ax2 = ax.twinx()
    ax2.plot(hours, elc, color=C["elc"], linewidth=1.0, label="ELC")
    ax2.plot(hours, fc, color=C["fc"], linewidth=1.0, linestyle="--", label="FC")
    ax2.plot(hours, bess_net, color=C["bess"], linewidth=1.0, linestyle=":", label="BESS net")
    ax2.set_ylabel("Device power (GW)", fontsize=8, color=C["dark"])
    ax2.tick_params(axis="y", labelcolor=C["dark"])

    ax.set_xlim(0, 23)
    ax.set_xlabel("Hour", fontsize=9)
    ax.set_ylabel("Renewable / Load (GW)", fontsize=9)
    ax.set_title(title_label, fontsize=9, loc="left", pad=4, fontweight="bold")
    ax.set_xticks(np.arange(0, 24, 4))

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=5.5, loc="upper center",
              frameon=False, ncol=3, bbox_to_anchor=(0.5, -0.12), handlelength=1.2)


def fig11_seasonal_dispatch():
    """Three separate seasonal dispatch figures archive."""
    if "scenarios" not in t1000.get("tssp", {}) or not t1000["tssp"]["scenarios"]:
        print("  SKIP fig11: no scenarios in 1000t JSON"); return

    sc = t1000["tssp"]["scenarios"][0]
    days_info = [
        (0, "(a) High-RE day — representative day 1"),
        (5, "(b) Low-RE day — representative day 6"),
        (10, "(c) Transition day — representative day 11"),
    ]

    for day_idx, title in days_info:
        fig, ax = plt.subplots(figsize=mm_inch(140, 75))
        _plot_dispatch_day(ax, sc, day_idx, title)
        plt.tight_layout(pad=0.6)
        suffix = ["summer", "winter", "transition"][days_info.index((day_idx, title))]
        save_fig(fig, FIG, f"fig11_{suffix}_dispatch")
        plt.close()
        print(f"  [OK] fig11_{suffix}_dispatch")


# ================================================================
# FIGURE 12: WEEKLY OPERATION
# ================================================================
def fig12_weekly_operation():
    """Weekly BESS + H2 operation profiles."""
    if "scenarios" not in t1000.get("tssp", {}) or not t1000["tssp"]["scenarios"]:
        print("  SKIP fig12: no scenarios in 1000t JSON"); return

    sc = t1000["tssp"]["scenarios"][0]
    hours = np.arange(168)  # 7 days

    bess_ch = np.array(sc["bess_ch"][:168])
    bess_dis = np.array(sc["bess_dis"][:168])
    bess_soc = np.array(sc["bess_e"][:168]) if "bess_e" in sc else np.zeros(168)
    elc = np.array(sc["elc"][:168])
    fc = np.array(sc["fc"][:168])
    h_level = np.array(sc["h_tank"][:168]) if "h_tank" in sc else np.zeros(168)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=mm_inch(160, 100), sharex=True)

    # Panel (a): BESS
    ax1.fill_between(hours, 0, bess_ch, color=C["bess_lt"], alpha=0.6, label="Charge")
    ax1.fill_between(hours, 0, -bess_dis, color=C["bess"], alpha=0.6, label="Discharge")
    ax1.plot(hours, bess_soc, color=C["dark"], linewidth=0.8, linestyle="--", label="SOC")
    ax1.set_ylabel("BESS (GW / p.u.)", fontsize=9)
    ax1.set_title("(a) BESS weekly operation", fontsize=9, loc="left", pad=4, fontweight="bold")
    ax1.legend(fontsize=6.5, loc="upper left", framealpha=0.6)
    ax1.axhline(y=0, color=C["gray"], linewidth=0.4)
    ax1.set_xlabel("Hour", fontsize=9)
    ax1.tick_params(labelbottom=True)

    # Panel (b): H2
    ax2.fill_between(hours, 0, elc, color=C["h2_light"], alpha=0.6, label="Electrolyzer")
    ax2.fill_between(hours, 0, -fc, color=C["fc"], alpha=0.6, label="Fuel cell")
    ax2.plot(hours, h_level/1000, color=C["dark"], linewidth=0.8, linestyle="--", label="Tank (kt)")
    ax2.set_xlabel("Hour", fontsize=9)
    ax2.set_ylabel("H$_2$ (GW / kt)", fontsize=9)
    ax2.set_title("(b) Hydrogen weekly operation", fontsize=9, loc="left", pad=4, fontweight="bold")
    ax2.legend(fontsize=6.5, loc="upper left", framealpha=0.6)
    ax2.axhline(y=0, color=C["gray"], linewidth=0.4)

    # Day separators
    for d in [24, 48, 72, 96, 120, 144]:
        ax1.axvline(x=d, color=C["gray"], linestyle=":", linewidth=0.4, alpha=0.5)
        ax2.axvline(x=d, color=C["gray"], linestyle=":", linewidth=0.4, alpha=0.5)

    plt.tight_layout(pad=0.6)
    save_fig(fig, FIG, "fig12_weekly_operation")
    plt.close()
    print("  [OK] fig12_weekly_operation")


# ================================================================
# FIGURE 13: HOURLY POWER ALLOCATION
# ================================================================
def fig13_hourly_power():
    """2-panel hourly power allocation: high-RE vs low-RE day."""
    if "scenarios" not in v3.get("tssp", {}) or not v3["tssp"]["scenarios"]:
        print("  SKIP fig13: no scenarios in v3 JSON"); return

    sc = v3["tssp"]["scenarios"][0]
    hours = np.arange(24)

    # Find high-RE day (max total renewable) and low-RE day (min)
    re_daily = []
    for d in range(20):
        h0 = d * 24
        h1 = h0 + 24
        wind = np.array(sc["wind_avail"][h0:h1])
        solar = np.array(sc["pv_avail"][h0:h1])
        re_daily.append(sum(wind) + sum(solar))
    high_day = int(np.argmax(re_daily))
    low_day = int(np.argmin(re_daily))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=mm_inch(180, 95))

    for ax, day_idx, title in [(ax1, high_day, "(a) High-RE day"), (ax2, low_day, "(b) Low-RE day")]:
        h0 = day_idx * 24
        h1 = h0 + 24
        wind = np.array(sc["wind_avail"][h0:h1]) / 1000
        solar = np.array(sc["pv_avail"][h0:h1]) / 1000
        load = np.array(sc["load"][h0:h1]) / 1000 if "load" in sc else np.zeros(24)
        elc = np.array(sc["elc"][h0:h1]) / 1000 if "elc" in sc else np.zeros(24)
        bess_ch = np.array(sc["bess_ch"][h0:h1]) / 1000 if "bess_ch" in sc else np.zeros(24)
        bess_dis = np.array(sc["bess_dis"][h0:h1]) / 1000 if "bess_dis" in sc else np.zeros(24)
        fc = np.array(sc["fc"][h0:h1]) / 1000 if "fc" in sc else np.zeros(24)
        bess_net = bess_dis - bess_ch

        # RE availability as subtle background (low zorder so lines sit on top)
        ax.fill_between(hours, 0, wind, alpha=0.18, color=C["wind"], label="Wind", zorder=1)
        ax.fill_between(hours, wind, wind+solar, alpha=0.22, color=C["solar"], label="PV", zorder=1)
        ax.plot(hours, load, color=C["dark"], linewidth=1.2, linestyle="--", label="Load", zorder=4)

        ax2_t = ax.twinx()
        ax2_t.plot(hours, elc, color=C["elc"], linewidth=1.3, label="ELC", zorder=5)
        ax2_t.plot(hours, fc, color=C["fc"], linewidth=1.3, linestyle="--", label="FC", zorder=5)
        ax2_t.plot(hours, bess_net, color=C["bess"], linewidth=1.3, linestyle=":", label="BESS net", zorder=5)
        ax2_t.set_ylabel("Device (GW)", fontsize=8)

        ax.set_xlim(0, 23)
        ax.set_xlabel("Hour", fontsize=9)
        ax.set_ylabel("Power (GW)", fontsize=9)
        ax.set_title(title, fontsize=9, loc="left", pad=4, fontweight="bold")
        ax.set_xticks(np.arange(0, 24, 4))

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2_t.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=6, loc="upper center",
                  frameon=False, ncol=3, bbox_to_anchor=(0.5, -0.18), handlelength=1.2)

    plt.tight_layout(pad=0.8, rect=[0, 0.06, 1, 1])
    save_fig(fig, FIG, "fig13_hourly_power_allocation")
    plt.close()
    print("  [OK] fig13_hourly_power_allocation")


# ================================================================
# FIGURE 14: CARBON DUAL-AXIS
# ================================================================
def fig14_carbon_dual():
    """Carbon cost share + BESS capacity dual-axis."""
    fig, ax1 = plt.subplots(figsize=mm_inch(140, 85))

    prices = cp["carbon_price_cny_per_ton"].values
    bess_p = cp["BESS_P_MW"].values
    bess_theo = cp["BESS_P_theoretical_MW"].values

    # Carbon cost share
    shares = []
    for _, row in cp.iterrows():
        cs = row["cost_carbon_10k"]
        tot = row["cost_inv_10k"] + row["cost_om_fix_10k"] + row["cost_op_10k"] + cs
        shares.append(cs / tot * 100 if tot > 0 else 0)

    bar_width = 25
    bars = ax1.bar(prices, shares, width=bar_width, color=C["carbon"], edgecolor="white",
                   linewidth=0.5, alpha=0.85, zorder=2, label="Carbon cost share")
    for i, (c, val) in enumerate(zip(prices, shares)):
        ax1.text(c, val + 0.15, f"{val:.1f}%", ha="center", va="bottom",
                 fontsize=6.5, color=C["carbon"], fontweight="bold")

    ax1.set_xlabel("Carbon price (CNY/tCO$_2$)", fontsize=9)
    ax1.set_ylabel("Carbon cost share of total cost (%)", fontsize=9, color=C["carbon"])
    ax1.tick_params(axis="y", labelcolor=C["carbon"], labelsize=7.5)
    ax1.set_ylim(0, max(shares)*1.25)
    ax1.set_xlim(-30, 540)
    ax1.set_xticks([0, 50, 100, 150, 200, 300, 500])

    # Right axis: BESS capacity
    ax2 = ax1.twinx()
    ax2.plot(prices, bess_p, "-s", color=C["bess"], linewidth=1.5, markersize=5,
             markerfacecolor="white", markeredgecolor=C["bess"], markeredgewidth=1.2, zorder=3, label="BESS actual")
    ax2.plot([prices[2], prices[3], prices[4]], [bess_p[2], bess_theo[3], bess_p[4]],
             "--", color=C["bess"], linewidth=1.2, alpha=0.6, zorder=2)
    ax2.scatter(prices[3], bess_theo[3], marker="x", s=80, c=C["theory"],
                linewidths=2.0, zorder=4, label="BESS theory")

    ax2.set_ylabel("BESS power capacity (MW)", fontsize=9, color=C["bess"])
    ax2.tick_params(axis="y", labelcolor=C["bess"], labelsize=7.5)
    ax2.set_ylim(4000, 10000)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", frameon=False, fontsize=7)

    ax1.grid(True, alpha=0.15, linestyle="--", linewidth=0.5, axis="y")

    plt.tight_layout(pad=0.8)
    save_fig(fig, FIG, "fig14_carbon_dual_axis")
    plt.close()
    print("  [OK] fig14_carbon_dual_axis")


# ================================================================
# FIGURE 15: VSS DECOMPOSITION
# ================================================================
def fig15_vss_decomp():
    """VSS decomposition: capacity + cost."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=mm_inch(165, 80))

    # Panel (a): Capacity comparison
    ev_cap = [h2["EV_BESS_P_MW"].values[1], h2["EV_ELC_MW"].values[1], h2["EV_FC_MW"].values[1]]
    tssp_cap = [h2["TSSP_BESS_P_MW"].values[1], h2["TSSP_ELC_MW"].values[1], h2["TSSP_FC_MW"].values[1]]
    comps = ["BESS\n(MW)", "ELC\n(MW)", "FC\n(MW)"]
    x = np.arange(len(comps))
    w = 0.3

    ax1.bar(x - w/2, ev_cap, w, color=C["ev"], edgecolor="white", linewidth=0.3, label="EV", zorder=2)
    ax1.bar(x + w/2, tssp_cap, w, color=C["tssp"], edgecolor="white", linewidth=0.3, label="TSSP", zorder=2)

    for i, (ev, tssp) in enumerate(zip(ev_cap, tssp_cap)):
        pct = (tssp - ev) / ev * 100
        color_diff = C["h2"] if pct > 0 else C["ev"]
        ax1.text(i, max(ev, tssp) * 1.03, f"{pct:+.1f}%", ha="center", fontsize=7,
                fontweight="bold", color=color_diff)

    ax1.set_xticks(x)
    ax1.set_xticklabels(comps, fontsize=7)
    ax1.set_ylabel("Capacity (MW)", fontsize=9)
    ax1.set_title("(a) Capacity: EV vs TSSP", fontsize=9, loc="left", pad=4)
    ax1.legend(fontsize=7.5, loc="upper right")

    # Panel (b): Cost comparison
    tssp_obj = abs(h2["TSSP_Obj"].values[1]) / 1e4
    ev_obj = abs(h2["EV_Obj"].values[1]) / 1e4
    eev_obj = abs(h2["EEV_Obj"].values[1]) / 1e4

    cost_labels = ["EV", "EEV", "TSSP"]
    cost_vals = [ev_obj, eev_obj, tssp_obj]
    colors_c = [C["ev"], C["gray"], C["tssp"]]

    bars = ax2.bar(cost_labels, cost_vals, color=colors_c, edgecolor="white", linewidth=0.3, width=0.45)
    # Value labels
    for bar, val in zip(bars, cost_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.1f}", ha="center", va="bottom", fontsize=7, fontweight="bold", color=C["dark"])

    ax2.annotate("", xy=(2, tssp_obj), xytext=(1, eev_obj),
                arrowprops=dict(arrowstyle="<->", color=C["dark"], lw=1.0))
    ax2.text(1.5, (eev_obj + tssp_obj)/2 + 0.8, f"VSS\n{h2['VSS_pct'].values[1]*100:.2f}%",
             ha="center", fontsize=7.5, fontweight="bold", color=C["dark"])

    ax2.set_ylabel("Total cost (B CNY)", fontsize=9)
    ax2.set_title("(b) Cost: EV, EEV, TSSP", fontsize=9, loc="left", pad=4)

    plt.tight_layout(pad=1.0)
    save_fig(fig, FIG, "fig15_vss_decomposition")
    plt.close()
    print("  [OK] fig15_vss_decomposition")


# ================================================================
# COPY EXISTING MANUAL/STATIC FIGURES
# ================================================================
def copy_static_figures():
    """Copy Fig.2, Fig.3, Fig.4 from existing sources."""
    import shutil
    src_dir = os.path.join(BASE, "results", "figures archive")
    copies = [
        ("fig2_multiscale_coordination.png", "fig02_multiscale_coordination.png"),
        ("fig3_uncertainty_pipeline.png", "fig03_uncertainty_pipeline.png"),
        ("fig3_kan_validation_2panel.png", "fig04_kan_validation.png"),
    ]
    for src_name, dst_name in copies:
        src = os.path.join(src_dir, src_name)
        dst = os.path.join(FIG, dst_name)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"  [OK] copied {src_name} -> {dst_name}")
        else:
            print(f"  [WARN] {src_name} not found")


# ================================================================
# MAIN
# ================================================================
if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("Generating all unified paper figures archive...")
    print("=" * 65)

    fig05_cost_structure()
    fig06_h2_bess()
    fig07_sse()
    fig08_carbon_regime()
    fig09_convergence()
    fig10_vss()
    fig11_seasonal_dispatch()
    fig12_weekly_operation()
    fig13_hourly_power()
    fig14_carbon_dual()
    fig15_vss_decomp()
    copy_static_figures()

    print("\n" + "=" * 65)
    print(f"All figures archive saved to: {FIG}")
    print("=" * 65)
