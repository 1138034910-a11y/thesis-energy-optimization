"""
Generate Supplementary Figure: Copula vs Independent Sampling comparison.
Side-by-side: (a) BESS power curves + (b) SSE bar comparison.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os, sys

sys.stdout.reconfigure(encoding='utf-8')

# ---- Data ----
h2_cap = np.array([200, 400, 600, 800, 1000])

# Copula (baseline from h2_sensitivity_v3_rigorous_vss.csv)
bess_cop = np.array([5519, 5758, 4061, 3710, 2646])

# Independent sampling (from no_copula_validation.json: 200, 400, 1000 only)
# Fill 600 and 800 with NaN since not run
bess_ind = np.array([6114, 5543, np.nan, np.nan, 5809])

# SSE values
sse_cop  = np.array([0.064,  -0.864, -0.316, -1.507])
sse_ind  = np.array([-0.147, np.nan,  np.nan,  0.055])  # only 200-400 and 400-1000 computable
intervals = ['200-400 t', '400-600 t', '600-800 t', '800-1000 t']

# ---- Plot ----
plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 9,
    'axes.titlesize': 10, 'axes.labelsize': 9,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'legend.fontsize': 8, 'figure.dpi': 300,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.05,
})

fig, axes = plt.subplots(1, 2, figsize=(7.48, 3.3))

# ---- (a) BESS Power Curves ----
ax = axes[0]

# Copula: all 5 points, solid line
ax.plot(h2_cap, bess_cop, 's-', color='#4C78A8', linewidth=1.8, markersize=8,
        markerfacecolor='white', markeredgewidth=1.5,
        label='Copula (baseline)', zorder=4)

# Independent: only 3 points (200, 400, 1000), dashed line with gaps for 600, 800
ax.plot([200, 400, 1000], [6114, 5543, 5809], 'o--', color='#E45756',
        linewidth=1.8, markersize=9, markerfacecolor='white', markeredgewidth=1.5,
        label='Independent sampling', zorder=3)

# Mark N/A for missing 600, 800 points
ax.text(600, 3200, 'N/A', ha='center', fontsize=7, color='gray', fontstyle='italic')
ax.text(800, 3200, 'N/A', ha='center', fontsize=7, color='gray', fontstyle='italic')

ax.set_xlabel('H$_2$ tank capacity (t)')
ax.set_ylabel('BESS power capacity (MW)')
ax.set_title('(a) BESS power: Copula vs Independent sampling', fontweight='bold')
ax.legend(frameon=False, loc='upper right')
ax.set_xlim(150, 1050)
ax.set_ylim(2000, 7000)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1000:.1f}k'))
ax.grid(True, alpha=0.2)

# Annotate key divergences with offset to avoid overlap
ax.annotate('Independent:\n6,114 MW', xy=(200, 6114),
            xytext=(130, 6650), fontsize=7, color='#E45756', ha='center',
            arrowprops=dict(arrowstyle='->', color='#E45756', lw=0.8))
ax.annotate('Copula:\n5,519 MW', xy=(200, 5519),
            xytext=(270, 4950), fontsize=7, color='#4C78A8', ha='center',
            arrowprops=dict(arrowstyle='->', color='#4C78A8', lw=0.8))

ax.annotate('Independent:\n5,809 MW', xy=(1000, 5809),
            xytext=(920, 6650), fontsize=7, color='#E45756', ha='center',
            arrowprops=dict(arrowstyle='->', color='#E45756', lw=0.8))
ax.annotate('Copula:\n2,646 MW', xy=(1000, 2646),
            xytext=(1070, 2200), fontsize=7, color='#4C78A8', ha='center',
            arrowprops=dict(arrowstyle='->', color='#4C78A8', lw=0.8))

# ---- (b) SSE Comparison ----
ax = axes[1]
x = np.arange(len(intervals))
w = 0.35

bars1 = ax.bar(x - w/2, sse_cop, w, color='#4C78A8', alpha=0.85,
               label='Copula (baseline)', edgecolor='white', linewidth=0.5, zorder=3)

# Independent: only 1st and 4th intervals have data; show 2nd/3rd as transparent placeholders
ind_plot = np.array([-0.147, 0, 0, 0.055])  # zeros as placeholders for N/A
alphas  = [0.85, 0.15, 0.15, 0.85]
bars2 = ax.bar(x + w/2, ind_plot, w, color='#E45756', alpha=0.85,
               label='Independent sampling', edgecolor='white', linewidth=0.5, zorder=3)
# Overlay N/A hatch on missing intervals
for i in [1, 2]:
    bars2[i].set_alpha(0.15)
    bars2[i].set_hatch('//')
    bars2[i].set_edgecolor('gray')

# Regime boundaries
ax.axhline(y=0, color='black', linewidth=0.6, linestyle='-')
ax.axhline(y=-1, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)

# Regime labels
ax.text(3.75, 0.20, 'Complementarity\n($\\varepsilon > 0$)', fontsize=7, ha='right', color='gray')
ax.text(3.75, -0.50, 'Substitution\n($-1 < \\varepsilon < 0$)', fontsize=7, ha='right', color='gray')
ax.text(3.75, -1.35, 'Strong sub.\n($\\varepsilon < -1$)', fontsize=7, ha='right', color='gray')

ax.set_xticks(x)
ax.set_xticklabels(intervals, rotation=30, ha='right')
ax.set_ylabel('Storage Substitution Elasticity')
ax.set_title('(b) SSE: Copula vs Independent sampling', fontweight='bold')
ax.legend(frameon=False, loc='lower left', fontsize=7)
ax.grid(True, alpha=0.2, axis='y')
ax.set_ylim(-1.8, 0.4)

# Value labels on Copula bars
for bar in bars1:
    h = bar.get_height()
    offset = 0.06 if h >= 0 else -0.14
    ax.text(bar.get_x() + bar.get_width()/2., h + offset,
            f'{h:+.3f}', ha='center', fontsize=6.5, color='#4C78A8', fontweight='bold')

# Value labels for Independent (only where data exists)
for i in [0, 3]:
    bar = bars2[i]
    val = ind_plot[i]
    offset = 0.06 if val >= 0 else -0.14
    ax.text(bar.get_x() + bar.get_width()/2., val + offset,
            f'{val:+.3f}', ha='center', fontsize=6.5, color='#E45756', fontweight='bold')

# N/A labels
for i in [1, 2]:
    ax.text(x[i] + w/2, -0.06, 'not\nrun', ha='center', fontsize=6, color='gray', fontstyle='italic')

plt.tight_layout()

# Save
outdir = 'results/figures_paper'
os.makedirs(outdir, exist_ok=True)
for fmt in ['png', 'pdf']:
    path = os.path.join(outdir, f'si_fig_s1_copula_vs_independent.{fmt}')
    fig.savefig(path, dpi=300)
    print(f'Saved: {path}')

plt.close()
print('Done.')
