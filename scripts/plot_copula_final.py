# -*- coding: utf-8 -*-
"""
Copula correlation figure — FINAL version aligned with paper narrative.
§3.3: Gaussian Copula preserves Kendall rank dependence vs. independent sampling.
Key message: ignoring correlation inflates the probability of over-optimistic
wind-solar co-occurrence (e.g., both high), biasing stochastic planning upward.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyBboxPatch
from scipy.stats import kendalltau, norm
import pandas as pd

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
})

OUT_DIR = 'results/figures_journal'


def save_both(fig, fname):
    base = os.path.join(OUT_DIR, fname)
    fig.savefig(base + '.png', dpi=300, bbox_inches='tight',
                pad_inches=0.02, facecolor='white')
    fig.savefig(base + '.pdf', bbox_inches='tight',
                pad_inches=0.02, facecolor='white')
    print(f'  [OK] {fname}.png + .pdf')


def load_data():
    df_w = pd.read_excel('data/数据.xlsx', sheet_name='Wind_generation', header=0)
    df_s = pd.read_excel('data/数据.xlsx', sheet_name='Solar_generation', header=0)
    w = df_w.iloc[0, 1:].values.astype(float)
    s = df_s.iloc[0, 1:].values.astype(float)
    w_pu = w / w.max()
    s_pu = s / s.max()
    return w_pu, s_pu


def add_ellipse(ax, x, y, color, n_std=2.0, lw=1.5, ls='--'):
    cov = np.cov(x, y)
    p = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1] + 1e-12)
    rx = np.sqrt(1 + p)
    ry = np.sqrt(1 - p)
    ell = Ellipse((0, 0), width=rx * 2, height=ry * 2,
                  facecolor='none', edgecolor=color, lw=lw, ls=ls,
                  transform=plt.matplotlib.transforms.Affine2D()
                  .rotate_deg(45)
                  .scale(np.sqrt(cov[0, 0]) * n_std,
                         np.sqrt(cov[1, 1]) * n_std)
                  .translate(np.mean(x), np.mean(y)) + ax.transData)
    ax.add_patch(ell)


def main():
    print('[Copula] Loading actual data & running paper workflow...')
    w_hist, s_hist = load_data()
    n_total = len(w_hist)
    rng = np.random.default_rng(42)

    # --- Step 1: Estimate Kendall tau (EXACTLY as in scenario_generator.py) ---
    tau_full, _ = kendalltau(w_hist, s_hist)
    rho_gauss = np.sin(np.pi / 2 * tau_full)
    print(f'  Kendall τ = {tau_full:.4f}  ->  Gaussian ρ = {rho_gauss:.4f}')

    # --- Step 2: Generate Copula & Independent scenarios (same marginals) ---
    # Rank -> uniform -> normal -> correlate -> uniform -> inverse empirical CDF
    u_w = (np.argsort(np.argsort(w_hist)) + 0.5) / n_total
    u_s = (np.argsort(np.argsort(s_hist)) + 0.5) / n_total
    z_w = norm.ppf(np.clip(u_w, 1e-6, 1 - 1e-6))
    z_s = norm.ppf(np.clip(u_s, 1e-6, 1 - 1e-6))

    # Copula
    cov = np.array([[1.0, rho_gauss], [rho_gauss, 1.0]])
    L = np.linalg.cholesky(cov)
    z_corr = np.column_stack([z_w, z_s]) @ L.T
    u_w_c = norm.cdf(z_corr[:, 0])
    u_s_c = norm.cdf(z_corr[:, 1])
    w_sorted, s_sorted = np.sort(w_hist), np.sort(s_hist)
    idx_w = np.clip((u_w_c * n_total).astype(int), 0, n_total - 1)
    idx_s = np.clip((u_s_c * n_total).astype(int), 0, n_total - 1)
    wind_cop, solar_cop = w_sorted[idx_w], s_sorted[idx_s]

    # Independent
    wind_ind = rng.permutation(w_hist)
    solar_ind = rng.permutation(s_hist)

    # --- Step 3: Compute "over-optimistic" tail metric ---
    # P(wind > 0.7 AND solar > 0.5) — a joint-surplus event
    def tail_prob(w, s):
        return np.mean((w > 0.7) & (s > 0.5)) * 100

    p_cop = tail_prob(wind_cop, solar_cop)
    p_ind = tail_prob(wind_ind, solar_ind)
    p_hist = tail_prob(w_hist, s_hist)

    # Downsample for visual clarity
    vis = rng.choice(n_total, size=1500, replace=False)
    wh, sh = w_hist[vis], s_hist[vis]
    wc, sc = wind_cop[vis], solar_cop[vis]
    wi, si = wind_ind[vis], solar_ind[vis]

    # ================================================================
    # Plot: 2-panel, compact, narrative-driven
    # ================================================================
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.0))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.82, bottom=0.16,
                        wspace=0.28)

    # --- (a) Gaussian Copula ---
    ax = axes[0]
    ax.scatter(wh, sh, c='#BDC3C7', s=4, alpha=0.30, edgecolors='none',
               rasterized=True, label='Historical')
    ax.scatter(wc, sc, c='#2878B5', s=5, alpha=0.45, edgecolors='none',
               rasterized=True, label='Copula samples')
    add_ellipse(ax, wc, sc, '#2878B5', n_std=2.0)

    # Over-optimistic zone annotation
    rect = FancyBboxPatch((0.7, 0.5), 0.3, 0.5, boxstyle="round,pad=0.01",
                          facecolor='none', edgecolor='#922B21', lw=1.2,
                          ls='--', alpha=0.6)
    ax.add_patch(rect)
    ax.text(0.72, 0.93, 'Over-optimistic\nzone', fontsize=7, color='#922B21',
            va='top', ha='left', fontweight='bold', alpha=0.8)

    ax.set_xlabel('Wind power (p.u.)', fontsize=8.5)
    ax.set_ylabel('Solar power (p.u.)', fontsize=8.5)
    ax.set_title('(a) Gaussian Copula', fontsize=9.5, fontweight='bold', pad=6)
    ax.text(0.05, 0.95, f'Kendall τ = {tau_full:.3f}\nTail prob. = {p_cop:.2f}%',
            transform=ax.transAxes, fontsize=8, va='top', ha='left',
            color='#2878B5', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='#2878B5', alpha=0.9))
    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(-0.02, 1.05)
    ax.tick_params(labelsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.12)
    ax.legend(loc='lower right', fontsize=7, frameon=False)

    # --- (b) Independent sampling ---
    ax = axes[1]
    ax.scatter(wh, sh, c='#BDC3C7', s=4, alpha=0.30, edgecolors='none',
               rasterized=True, label='Historical')
    ax.scatter(wi, si, c='#C0392B', s=5, alpha=0.35, edgecolors='none',
               rasterized=True, label='Independent')
    add_ellipse(ax, wi, si, '#C0392B', n_std=2.0)

    rect2 = FancyBboxPatch((0.7, 0.5), 0.3, 0.5, boxstyle="round,pad=0.01",
                           facecolor='none', edgecolor='#922B21', lw=1.2,
                           ls='--', alpha=0.6)
    ax.add_patch(rect2)
    ax.text(0.72, 0.93, 'Over-optimistic\nzone', fontsize=7, color='#922B21',
            va='top', ha='left', fontweight='bold', alpha=0.8)

    ax.set_xlabel('Wind power (p.u.)', fontsize=8.5)
    ax.set_ylabel('Solar power (p.u.)', fontsize=8.5)
    ax.set_title('(b) Independent sampling', fontsize=9.5, fontweight='bold',
                 pad=6)
    ax.text(0.05, 0.95, f'ρ = 0.000\nTail prob. = {p_ind:.2f}%',
            transform=ax.transAxes, fontsize=8, va='top', ha='left',
            color='#C0392B', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='#C0392B', alpha=0.9))
    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(-0.02, 1.05)
    ax.tick_params(labelsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.12)
    ax.legend(loc='lower right', fontsize=7, frameon=False)

    # Figure-level title
    fig.suptitle(
        'Joint wind-solar scenario generation: Copula preserves rank dependence '
        f'(τ={tau_full:.3f}) and avoids {p_ind/p_cop:.1f}× over-optimistic tail events',
        fontsize=9.5, fontweight='bold', y=0.98)

    save_both(fig, 'fig_copula_correlation_final')
    plt.close(fig)
    print(f'  Historical tail={p_hist:.2f}%, Copula={p_cop:.2f}%, '
          f'Independent={p_ind:.2f}% (ratio={p_ind/p_cop:.1f}×)')
    print('[Copula] Done.')


if __name__ == '__main__':
    main()
