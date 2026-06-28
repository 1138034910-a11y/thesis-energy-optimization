# -*- coding: utf-8 -*-
"""
Copula correlation scatter plot using ACTUAL Gansu wind-solar data.
§3.3: Proves "ignoring correlation leads to over-optimistic scenarios".
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import kendalltau, norm

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
})

OUT_DIR = 'results/figures_journal'
os.makedirs(OUT_DIR, exist_ok=True)


def save_both(fig, fname):
    base = os.path.join(OUT_DIR, fname)
    fig.savefig(base + '.png', dpi=300, bbox_inches='tight',
                pad_inches=0.02, facecolor='white')
    fig.savefig(base + '.pdf', bbox_inches='tight',
                pad_inches=0.02, facecolor='white')
    print(f'  [OK] {fname}.png + .pdf')


def load_actual_data():
    """Load Gansu wind-solar hourly data and normalize to p.u."""
    import pandas as pd
    df_wind = pd.read_excel('data/数据.xlsx', sheet_name='Wind_generation', header=0)
    df_solar = pd.read_excel('data/数据.xlsx', sheet_name='Solar_generation', header=0)

    # First row = Gansu province
    wind_raw = df_wind.iloc[0, 1:].values.astype(float)
    solar_raw = df_solar.iloc[0, 1:].values.astype(float)

    # Normalize to [0, 1] using own max (capacity factor proxy)
    wind_pu = wind_raw / wind_raw.max()
    solar_pu = solar_raw / solar_raw.max()
    return wind_pu, solar_pu


def add_confidence_ellipse(ax, x, y, color, n_std=2.0, lw=1.5, ls='--'):
    """Add 95% confidence ellipse based on covariance."""
    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])

    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor='none', edgecolor=color, lw=lw, ls=ls,
                      transform=plt.matplotlib.transforms.Affine2D()
                      .rotate_deg(45)
                      .scale(np.sqrt(cov[0, 0]) * n_std,
                             np.sqrt(cov[1, 1]) * n_std)
                      .translate(np.mean(x), np.mean(y))
                      + ax.transData)
    ax.add_patch(ellipse)


def main():
    print('[Copula] Loading actual Gansu data...')
    wind_hist, solar_hist = load_actual_data()
    n_total = len(wind_hist)

    # Downsample for visualization (scatter with 8760 points is too dense)
    rng = np.random.default_rng(42)
    idx_vis = rng.choice(n_total, size=min(2000, n_total), replace=False)
    w_vis, s_vis = wind_hist[idx_vis], solar_hist[idx_vis]

    # Compute Kendall tau from FULL data
    tau_full, _ = kendalltau(wind_hist, solar_hist)
    rho_gauss = np.sin(np.pi / 2 * tau_full)
    print(f'  Actual Kendall τ = {tau_full:.4f}  ->  Gaussian ρ = {rho_gauss:.4f}')

    # --- Panel (b): Gaussian Copula sampling ---
    # Use the SAME marginal distributions but impose correlation structure
    # Method: rank-transform actual data to uniform, then apply Gaussian Copula
    from scipy.stats import rankdata

    # Rank-based uniform transform
    u_wind = (rankdata(wind_hist) - 0.5) / n_total
    u_solar = (rankdata(solar_hist) - 0.5) / n_total

    # Transform to standard normal via inverse CDF
    z_wind = norm.ppf(np.clip(u_wind, 1e-6, 1 - 1e-6))
    z_solar = norm.ppf(np.clip(u_solar, 1e-6, 1 - 1e-6))

    # Apply correlation via Cholesky
    cov = np.array([[1.0, rho_gauss], [rho_gauss, 1.0]])
    L = np.linalg.cholesky(cov)
    z_mat = np.column_stack([z_wind, z_solar])
    z_corr = z_mat @ L.T

    # Transform back to uniform, then to original marginal via empirical CDF
    u_wind_cop = norm.cdf(z_corr[:, 0])
    u_solar_cop = norm.cdf(z_corr[:, 1])

    # Inverse empirical CDF: map uniform back to sorted actual values
    wind_sorted = np.sort(wind_hist)
    solar_sorted = np.sort(solar_hist)
    idx_w = np.clip((u_wind_cop * n_total).astype(int), 0, n_total - 1)
    idx_s = np.clip((u_solar_cop * n_total).astype(int), 0, n_total - 1)
    wind_cop = wind_sorted[idx_w]
    solar_cop = solar_sorted[idx_s]

    # --- Panel (c): Independent sampling ---
    # Same marginals, but shuffle independently
    wind_ind = rng.permutation(wind_hist)
    solar_ind = rng.permutation(solar_hist)

    # Downsample for visualization
    idx_vis2 = rng.choice(n_total, size=min(2000, n_total), replace=False)
    w_cop_vis, s_cop_vis = wind_cop[idx_vis2], solar_cop[idx_vis2]
    w_ind_vis, s_ind_vis = wind_ind[idx_vis2], solar_ind[idx_vis2]

    # ================================================================
    # Plotting
    # ================================================================
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.2))
    fig.subplots_adjust(left=0.06, right=0.98, top=0.82, bottom=0.14,
                        wspace=0.30)

    # --- (a) Historical ---
    ax = axes[0]
    ax.scatter(w_vis, s_vis, c='#2C3E50', s=5, alpha=0.35,
               edgecolors='none', rasterized=True, label='Observations')
    add_confidence_ellipse(ax, w_vis, s_vis, '#2C3E50', n_std=2.0)
    ax.set_xlabel('Wind power (p.u.)', fontsize=8.5)
    ax.set_ylabel('Solar power (p.u.)', fontsize=8.5)
    ax.set_title('(a) Historical observations', fontsize=9, fontweight='bold',
                 pad=8)
    ax.text(0.05, 0.95, f'Kendall τ = {tau_full:.3f}',
            transform=ax.transAxes, fontsize=8.5, va='top', ha='left',
            color='#2C3E50', fontweight='bold')
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.tick_params(labelsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.12)

    # --- (b) Copula ---
    ax = axes[1]
    ax.scatter(w_cop_vis, s_cop_vis, c='#2878B5', s=5, alpha=0.35,
               edgecolors='none', rasterized=True, label='Copula samples')
    add_confidence_ellipse(ax, w_cop_vis, s_cop_vis, '#2878B5', n_std=2.0)
    ax.set_xlabel('Wind power (p.u.)', fontsize=8.5)
    ax.set_ylabel('Solar power (p.u.)', fontsize=8.5)
    ax.set_title('(b) Gaussian Copula', fontsize=9, fontweight='bold', pad=8)
    ax.text(0.05, 0.95, f'ρ = {rho_gauss:.3f}\n(preserves τ)',
            transform=ax.transAxes, fontsize=8.5, va='top', ha='left',
            color='#2878B5', fontweight='bold')
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.tick_params(labelsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.12)

    # --- (c) Independent ---
    ax = axes[2]
    ax.scatter(w_ind_vis, s_ind_vis, c='#C0392B', s=5, alpha=0.30,
               edgecolors='none', rasterized=True, label='Independent')
    add_confidence_ellipse(ax, w_ind_vis, s_ind_vis, '#C0392B', n_std=2.0)
    ax.set_xlabel('Wind power (p.u.)', fontsize=8.5)
    ax.set_ylabel('Solar power (p.u.)', fontsize=8.5)
    ax.set_title('(c) Independent sampling', fontsize=9, fontweight='bold',
                 pad=8)
    ax.text(0.05, 0.95, 'ρ = 0.000\n(ignores τ)',
            transform=ax.transAxes, fontsize=8.5, va='top', ha='left',
            color='#C0392B', fontweight='bold')
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.tick_params(labelsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.12)

    # Figure-level title (moved up to avoid overlap)
    fig.suptitle('Wind-solar scenario generation: Copula preserves rank dependence',
                 fontsize=10, fontweight='bold', y=0.97)

    save_both(fig, 'fig_copula_correlation_v2')
    plt.close(fig)
    print('[Copula] Done.')


if __name__ == '__main__':
    main()
