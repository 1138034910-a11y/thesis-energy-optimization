#!/usr/bin/env python3
"""Standalone KAN forecast — identical to fig04 panels (a)+(b), no CRPS panel."""
import os, sys, numpy as np, pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_project_root)
sys.path.insert(0, _project_root)

from nature_style import apply, save_fig, C_WIND, C_PV, C_CAP

apply()

# ------------------------------------------------------------------
# Load data — identical to original fig04
# ------------------------------------------------------------------
kan = pd.read_csv("results/tables/kan_forecasts.csv")
wind_pred = pd.read_csv("data/甘肃_风电_prediction_result.csv")
solar_pred = pd.read_csv("data/甘肃_光伏_prediction_result.csv")

# Use a representative 168h (7-day) window starting from hour 120
# to capture both high and low wind/solar periods
START = 1000
H = 168
hours = np.arange(H)

aw = wind_pred["actual_pu"].bfill().values[START:START+H]
mw = kan["wind_mu"].values[START:START+H]
sw = kan["wind_sigma"].values[START:START+H]

as_ = solar_pred["actual_pu"].bfill().values[START:START+H]
ms = kan["solar_mu"].values[START:START+H]
ss = kan["solar_sigma"].values[START:START+H]

# ------------------------------------------------------------------
# Layout: 2 rows × 1 column (same style, no CRPS panel)
# ------------------------------------------------------------------
fig, (ax_w, ax_s) = plt.subplots(2, 1, figsize=(10, 4.8))
fig.subplots_adjust(hspace=0.32, left=0.08, right=0.97, top=0.94, bottom=0.08)

# ---- Panel (a): Wind forecast — IDENTICAL to original ----
ax_w.fill_between(hours, np.clip(mw - 2*sw, 0, 1), np.clip(mw + 2*sw, 0, 1),
                  color=C_WIND, alpha=0.10, zorder=1)
ax_w.fill_between(hours, np.clip(mw - sw, 0, 1), np.clip(mw + sw, 0, 1),
                  color=C_WIND, alpha=0.22, zorder=1)
ax_w.plot(hours, aw, color=C_CAP, lw=0.7, label="Actual", alpha=0.8, zorder=4)
ax_w.plot(hours, mw, color=C_WIND, lw=1.6, label="KAN mean", zorder=5)
ax_w.set_ylabel("Wind power (p.u.)", fontsize=8)
ax_w.set_title("(a) Wind — KAN probabilistic forecast", loc="left", fontsize=9, fontweight="bold", pad=4)
ax_w.set_xlim(0, H-1)
ax_w.set_ylim(-0.02, 1.05)
ax_w.set_xticks(np.arange(0, H+1, 24))
ax_w.set_xticklabels([])
ax_w.legend(loc="upper right", frameon=False, fontsize=6.5, handlelength=1.2)

# ---- Panel (b): Solar forecast — IDENTICAL to original ----
ax_s.fill_between(hours, np.clip(ms - 2*ss, 0, 1), np.clip(ms + 2*ss, 0, 1),
                  color=C_PV, alpha=0.10, zorder=1)
ax_s.fill_between(hours, np.clip(ms - ss, 0, 1), np.clip(ms + ss, 0, 1),
                  color=C_PV, alpha=0.22, zorder=1)
ax_s.plot(hours, as_, color=C_CAP, lw=0.7, label="Actual", alpha=0.8, zorder=4)
ax_s.plot(hours, ms, color=C_PV, lw=1.6, label="KAN mean", zorder=5)
ax_s.set_ylabel("Solar power (p.u.)", fontsize=8)
ax_s.set_xlabel("Hour of year", fontsize=8)
ax_s.set_title("(b) Solar — KAN probabilistic forecast", loc="left", fontsize=9, fontweight="bold", pad=4)
ax_s.set_xlim(0, H-1)
ax_s.set_ylim(-0.02, 1.05)
ax_s.set_xticks(np.arange(0, H+1, 24))
ax_s.legend(loc="upper left", frameon=False, fontsize=6.5, handlelength=1.2)

# ------------------------------------------------------------------
# Save — new filename, does NOT overwrite fig04
# ------------------------------------------------------------------
outdir = os.path.join(_project_root, "results", "figures_paper")
os.makedirs(outdir, exist_ok=True)
for fmt, dpi in [("png", 600), ("pdf", None), ("svg", None)]:
    kw = dict(bbox_inches="tight", facecolor="white", edgecolor="none")
    if dpi: kw["dpi"] = dpi
    fig.savefig(f"{outdir}/fig_kan_forecast_standalone.{fmt}", **kw)

print("[OK] fig_kan_forecast_standalone.png + .pdf + .svg")
plt.close(fig)
