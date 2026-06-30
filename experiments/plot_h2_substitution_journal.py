"""
H2-BESS Substitution Curve — Journal-Grade Visualization
Compatible with: Journal of Energy Storage, Applied Energy, Energy, Nature Energy, IEEE TPWRS
Author: [Your Name]
Date: 2026-05-08
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np

# ============================================================
# 实验数据 (从 h2_tank_sensitivity_v3.csv 提取)
# ============================================================
h2_cap = np.array([200, 400, 600, 800, 1000])
tssp_bess = np.array([5519, 5758, 4061, 3710, 2646])
ev_bess = np.array([5908, 5428, 3786, 3846, 2725])

# ============================================================
# 全局字体与样式设置 (Nature/Science 兼容)
# ============================================================
plt.rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.linewidth': 1.0,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.major.size': 5,
    'ytick.major.size': 5,
    'axes.spines.top': False,      # 极简边框
    'axes.spines.right': False,
})

fig = plt.figure(figsize=(7.5, 5.8))
ax = fig.add_axes([0.12, 0.14, 0.78, 0.72])  # 手动布局 [left, bottom, width, height]

# ============================================================
# 配色方案
# ============================================================
color_tssp = '#C0392B'      # 深红
color_ev = '#2980B9'        # 深蓝
color_fill = '#FADBD8'      # 浅红填充
color_annot = '#922B21'     # 注释深红
color_green = '#1E8449'     # 趋势绿

# ============================================================
# 主图：双折线 + 填充
# ============================================================
line_ev, = ax.plot(h2_cap, ev_bess, '--', color=color_ev, linewidth=1.8,
                   marker='s', markersize=7, markerfacecolor='white',
                   markeredgecolor=color_ev, markeredgewidth=1.5,
                   label='EV (Expected Value)', zorder=3, alpha=0.85)

line_tssp, = ax.plot(h2_cap, tssp_bess, '-', color=color_tssp, linewidth=2.5,
                     marker='o', markersize=9, markerfacecolor=color_tssp,
                     markeredgecolor='white', markeredgewidth=2.0,
                     label='TSSP (Two-Stage Stochastic)', zorder=5)

ax.fill_between(h2_cap, tssp_bess, ev_bess, alpha=0.12, color=color_tssp,
                interpolate=True, zorder=1)

# ============================================================
# 数值标签
# ============================================================
for i, (h, tssp_val, ev_val) in enumerate(zip(h2_cap, tssp_bess, ev_bess)):
    offset_tssp = 160 if i != 1 else -220
    va_tssp = 'bottom' if offset_tssp > 0 else 'top'
    ax.annotate(f'{tssp_val:,}', xy=(h, tssp_val), xytext=(h, tssp_val + offset_tssp),
                fontsize=9.5, fontweight='bold', color=color_tssp,
                ha='center', va=va_tssp, zorder=6)

    offset_ev = 160 if i != 1 else 220
    ax.annotate(f'{ev_val:,}', xy=(h, ev_val), xytext=(h, ev_val + offset_ev),
                fontsize=8.5, color=color_ev, ha='center', va='bottom',
                style='italic', alpha=0.85, zorder=4)

# ============================================================
# Portfolio Insurance Effect 标注 (400t)
# ============================================================
ax.annotate('Portfolio Insurance\nEffect at 400t',
            xy=(400, 5758), xytext=(280, 6400),
            fontsize=10, fontweight='bold', color=color_annot,
            ha='center', va='center',
            arrowprops=dict(arrowstyle='->', color=color_annot, lw=1.5,
                            connectionstyle='arc3,rad=0.25'),
            bbox=dict(boxstyle='round,pad=0.45', facecolor='#FDEDEC',
                      edgecolor=color_annot, linewidth=1.2, alpha=0.95),
            zorder=7)

# ============================================================
# 整体趋势标注
# ============================================================
ax.annotate('', xy=(1000, 2646), xytext=(200, 5519),
            arrowprops=dict(arrowstyle='->', color=color_green, lw=2.2,
                            connectionstyle='arc3,rad=-0.12'),
            zorder=2)
ax.text(650, 4600, '↓ 52.1%', fontsize=13, fontweight='bold',
        color=color_green, ha='center', zorder=6)

# ============================================================
# 临界阈值虚线
# ============================================================
ax.axvline(x=600, color='gray', linestyle=':', linewidth=1.0, alpha=0.6, zorder=0)
ax.text(600, 1800, 'Critical\nThreshold\n(~600t)', fontsize=8, color='gray',
        ha='center', va='bottom', style='italic')

# ============================================================
# 坐标轴
# ============================================================
ax.set_xlabel('H₂ Tank Capacity (t)', fontsize=13, fontweight='bold', labelpad=8)
ax.set_ylabel('BESS Power Capacity (MW)', fontsize=13, fontweight='bold', labelpad=8)
ax.set_title('H₂ Storage — BESS Substitution Curve Under Uncertainty',
             fontsize=14, fontweight='bold', pad=12)

ax.set_xlim(0, 1200)
ax.set_ylim(1500, 7000)
ax.set_xticks([0, 200, 400, 600, 800, 1000])
ax.set_xticklabels(['0', '200', '400', '600', '800', '1,000'], fontsize=10.5)
ax.set_yticks([2000, 3000, 4000, 5000, 6000])
ax.set_yticklabels(['2,000', '3,000', '4,000', '5,000', '6,000'], fontsize=10.5)

ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.6, zorder=0)
ax.set_axisbelow(True)

# ============================================================
# 图例
# ============================================================
legend = ax.legend(handles=[line_tssp, line_ev],
                   loc='upper right',
                   bbox_to_anchor=(0.98, 0.72),
                   fontsize=10.5,
                   framealpha=0.95,
                   edgecolor='gray',
                   fancybox=True,
                   borderpad=0.6)
legend.get_frame().set_linewidth(0.8)

# ============================================================
# 局部放大子图 (Inset) — 200t-600t
# ============================================================
ax_inset = inset_axes(ax, width="38%", height="36%", loc='upper right',
                       bbox_to_anchor=(0.96, 0.96, 1, 1),
                       bbox_transform=ax.transAxes)

h2_zoom = np.array([200, 400, 600, 800])
tssp_zoom = np.array([5519, 5758, 4061, 3710])
ev_zoom = np.array([5908, 5428, 3786, 3846])

ax_inset.plot(h2_zoom, tssp_zoom, '-o', color=color_tssp, linewidth=2.2,
              markersize=7, markerfacecolor=color_tssp,
              markeredgecolor='white', markeredgewidth=1.8, zorder=5)
ax_inset.plot(h2_zoom, ev_zoom, '--s', color=color_ev, linewidth=1.5,
              markersize=5, markerfacecolor='white',
              markeredgecolor=color_ev, markeredgewidth=1.2, zorder=3, alpha=0.85)
ax_inset.fill_between(h2_zoom, tssp_zoom, ev_zoom, alpha=0.12, color=color_tssp, zorder=1)

ax_inset.set_xlim(150, 650)
ax_inset.set_ylim(5000, 6200)
ax_inset.set_xticks([200, 400, 600])
ax_inset.set_xticklabels(['200', '400', '600'], fontsize=7.5)
ax_inset.set_yticks([5200, 5600, 6000])
ax_inset.set_yticklabels(['5,200', '5,600', '6,000'], fontsize=7.5)
ax_inset.tick_params(axis='both', which='major', labelsize=7.5, direction='out')
ax_inset.set_xlabel('H₂ (t)', fontsize=8, labelpad=2)
ax_inset.set_ylabel('BESS (MW)', fontsize=8, labelpad=2)

for spine in ['top', 'right']:
    ax_inset.spines[spine].set_visible(False)

# 子图400t标注
ax_inset.annotate('400t\npeak', xy=(400, 5758), xytext=(480, 5980),
                 fontsize=8, fontweight='bold', color=color_annot,
                 ha='center', va='center',
                 arrowprops=dict(arrowstyle='->', color=color_annot, lw=1.2),
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#FDEDEC',
                           edgecolor=color_annot, linewidth=1.0, alpha=0.9))

# 子图背景
rect = FancyBboxPatch((0.02, 0.02), 0.96, 0.96, boxstyle="round,pad=0.02",
                       facecolor='white', edgecolor='gray', linewidth=1.2,
                       alpha=0.95, transform=ax_inset.transAxes, zorder=0)
ax_inset.add_patch(rect)
ax_inset.set_facecolor('white')

# ============================================================
# 底部注释
# ============================================================
fig.text(0.5, 0.02,
         'Note: The non-monotonicity at 400 t reflects a portfolio insurance effect under uncertainty. '
         'At moderate hydrogen levels, TSSP co-invests in BESS and H₂ storage to hedge tail-risk scenarios, '
         'consistent with two-stage stochastic capacity planning theory (Birge & Louveaux, 2011). '
         'The substitutive relationship dominates only beyond the critical threshold (~600 t).',
         ha='center', fontsize=9, style='italic', color='#555555',
         linespacing=1.4)

# ============================================================
# 保存
# ============================================================
fig.savefig('results/figures archive/h2_bess_substitution_journal.png',
            dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none',
            pad_inches=0.18)
plt.show()
print("Saved: results/figures archive/h2_bess_substitution_journal.png")
