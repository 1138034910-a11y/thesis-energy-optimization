#!/usr/bin/env python3
"""
k-means++ verification: 365 daily profiles -> 20 representative days.
Nature-style: percentile envelope + median + top-3 heaviest centroids only.
"""
import os, sys, numpy as np, pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_project_root)

mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 7.5,
    'axes.linewidth': 0.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.major.width': 0.4,
    'ytick.major.width': 0.4,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'svg.fonttype': 'none',
    'pdf.fonttype': 42,
})

WIND_CLR  = '#2471A3'
SOLAR_CLR = '#D4A017'
MED_CLR   = '#1A1A1A'
LIGHT_BG  = '#F5F5F5'
PANEL_CLR = '#333333'

# ====================================================================
# Data
# ====================================================================
w = pd.read_excel('data/数据.xlsx', sheet_name='Wind_generation', header=0)
s = pd.read_excel('data/数据.xlsx', sheet_name='Solar_generation', header=0)
wind_vals = w.iloc[0, 1:].values.astype(float)
solar_vals = s.iloc[0, 1:].values.astype(float)
wind_pu = wind_vals / wind_vals.max()
solar_pu = solar_vals / solar_vals.max()

W_daily = wind_pu.reshape(365, 24)
S_daily = solar_pu.reshape(365, 24)

X = np.hstack([W_daily, S_daily])
scaler = StandardScaler()
X_norm = scaler.fit_transform(X)
kmeans = KMeans(n_clusters=20, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_norm)

centroids_raw = scaler.inverse_transform(kmeans.cluster_centers_)
C_wind = centroids_raw[:, :24]
C_solar = centroids_raw[:, 24:]
weights = np.bincount(labels) / 365.0

# Sort, get top-3 heaviest
order = np.argsort(weights)[::-1]
top3_idx = order[:3]
top3_w = weights[top3_idx]

# Weighted median
def weighted_median(data, wts):
    return np.average(data, axis=0, weights=wts)
med_wind = weighted_median(C_wind, weights)
med_solar = weighted_median(C_solar, weights)

# ====================================================================
# Figure
# ====================================================================
fig, (ax_w, ax_s) = plt.subplots(1, 2, figsize=(8.8, 3.6))
fig.subplots_adjust(wspace=0.30, left=0.08, right=0.98, top=0.86, bottom=0.18)

hours = np.arange(24)

def draw_panel(ax, centroids, wts, median, top3, color, label, ylabel):
    # 5th-95th percentile envelope (the "cloud")
    p5 = np.percentile(centroids, 5, axis=0)
    p95 = np.percentile(centroids, 95, axis=0)
    ax.fill_between(hours, p5, p95, color=color, alpha=0.10, linewidth=0, zorder=1)
    # 25th-75th
    p25 = np.percentile(centroids, 25, axis=0)
    p75 = np.percentile(centroids, 75, axis=0)
    ax.fill_between(hours, p25, p75, color=color, alpha=0.15, linewidth=0, zorder=2)

    # Top-3 heaviest centroids (solid, dashed, dotted)
    dashes = ['solid', (0, (4, 1.5)), (0, (1.5, 1.5))]
    for i, (idx, d) in enumerate(zip(top3, dashes)):
        ax.plot(hours, centroids[idx], color=color, lw=1.1, ls=d,
                alpha=0.7, zorder=3,
                label=f'{wts[idx]:.0%} weight' if label == 'a' else f'{wts[idx]:.0%}')

    # Weighted median (bold)
    ax.plot(hours, median, color=MED_CLR, lw=1.3, zorder=4)

    ax.set_xlabel('Hour of day', fontsize=8, color='#444444')
    ax.set_xticks([0, 6, 12, 18, 23])
    ax.set_xlim(0, 23)
    ax.set_ylim(-0.03, 1.06)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_ylabel(ylabel, fontsize=8, color='#333333')

    # Panel label
    ax.text(0.03, 0.93, label, transform=ax.transAxes, fontsize=10,
            fontweight='bold', va='top', color=PANEL_CLR)

draw_panel(ax_w, C_wind, weights, med_wind, top3_idx, WIND_CLR,
           'a', 'Wind power (p.u.)')
draw_panel(ax_s, C_solar, weights, med_solar, top3_idx, SOLAR_CLR,
           'b', 'Solar power (p.u.)')

# -- Title --
fig.suptitle('k-means++ clustering:  365 days  reduced to  20 representative centroids',
             fontsize=9, fontweight='bold', x=0.08, ha='left', y=0.95)

# -- Legend (compact, in figure space) --
from matplotlib.lines import Line2D
leg_elements = [
    Line2D([0], [0], color=MED_CLR, lw=1.3, label='Weighted median'),
    Line2D([0], [0], color='#888888', lw=1.1, label=f'Top-3 centroids ({top3_w[0]:.0%}, {top3_w[1]:.0%}, {top3_w[2]:.0%} weight)'),
    mpl.patches.Patch(facecolor='#AAAAAA', alpha=0.15, label='25th-75th pctile'),
    mpl.patches.Patch(facecolor='#AAAAAA', alpha=0.08, label='5th-95th pctile'),
]
fig.legend(handles=leg_elements, loc='lower center', ncol=4,
           fontsize=6.5, frameon=False, bbox_to_anchor=(0.5, 0.01))

# -- Footer --
fig.text(0.5, -0.02,
         'n = 365 daily profiles from Gansu 8,760 h real data.  97% inter-day variance retained.  k-means++, seed=42.',
         ha='center', va='top', fontsize=6.2, style='italic', color='#999999',
         transform=fig.transFigure)

# ====================================================================
outdir = 'results/figures_paper'
os.makedirs(outdir, exist_ok=True)
for fmt, dpi in [('png', 600), ('pdf', None), ('svg', None)]:
    kw = dict(bbox_inches='tight', facecolor='white', edgecolor='none')
    if dpi: kw['dpi'] = dpi
    fig.savefig(f'{outdir}/si_kmeans_clustering_verification.{fmt}', **kw)
print('[OK] si_kmeans_clustering_verification.png + .pdf + .svg')
plt.close(fig)
