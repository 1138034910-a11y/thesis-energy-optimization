# -*- coding: utf-8 -*-
"""
Nature-standard figure style module for Applied Energy / ECM submission.
Unified rcParams, color palette, and helper functions.

Key standards:
  - Sans-serif font stack (Arial > Helvetica > DejaVu Sans)
  - 8–10 pt hierarchy (tick 8, axis 9, title 10, note 7)
  - Minimal visual noise: no top/right spines, legend frame off
  - White background, 600 dpi raster, vector PDF with editable fonts
  - Output: PDF + PNG only (no SVG/TIFF per project policy)
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ============================================================
# GLOBAL RC PARAMS — Nature standard
# ============================================================
RC = {
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif'],
    'font.size': 8,                  # base
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7.5,
    'figure.titlesize': 10,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'legend.frameon': False,         # Nature: no box around legend
    'legend.borderpad': 0.4,
    'legend.labelspacing': 0.3,
    'figure.dpi': 600,
    'savefig.dpi': 600,
    'pdf.fonttype': 42,              # editable TrueType in PDF
    'svg.fonttype': 'none',
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
    'savefig.edgecolor': 'none',
}

def apply():
    """Apply the Nature-standard rcParams. Call once at script start."""
    plt.rcParams.update(RC)

# ============================================================
# COLOR PALETTE — restrained, print-safe
# ============================================================
C_BESS   = '#2878B5'   # Electric / primary
C_ELC    = '#1E8449'   # Hydrogen-electrolyzer
C_FC     = '#F58518'   # Hydrogen-fuel cell
C_CARBON = '#E45756'   # Carbon / emission
C_CAP    = '#333333'   # Cap line / dark grey
C_REF    = '#888888'   # Reference / threshold
C_EV     = '#C0392B'   # EV line (distinct red)
C_THEORY = '#7B2CBF'   # Theoretical / projection
C_WIND   = '#3498DB'
C_PV     = '#F1C40F'
C_THERM  = '#C0392B'
C_POS    = '#1ABC9C'
C_NEG    = '#9B59B6'

# ============================================================
# HELPERS
# ============================================================
def save_fig(fig, outdir, basename, dpi=600):
    """Save figure as PDF + PNG only. No SVG/TIFF."""
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(f'{outdir}/{basename}.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    fig.savefig(f'{outdir}/{basename}.png', dpi=dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f'  [OK] {basename}.pdf + .png  ->  {outdir}')


def add_note(fig, text, ypos=-0.01, fontsize=7, color='#444444'):
    """Add self-contained figure note below plot area.

    Parameters
    ----------
    ypos : float
        Figure-coordinate y position (default -0.01 just below tight bbox).
    """
    fig.text(0.5, ypos, text, ha='center', va='top', fontsize=fontsize,
             style='italic', color=color, linespacing=1.35,
             transform=fig.transFigure)


def make_legend_line(color, marker, label, linestyle='-', linewidth=1.8,
                     markersize=6, markerfacecolor=None, markeredgecolor=None):
    """Convenience for building legend handles."""
    kwargs = {
        'color': color, 'marker': marker, 'linestyle': linestyle,
        'linewidth': linewidth, 'markersize': markersize,
        'label': label,
    }
    if markerfacecolor is not None:
        kwargs['markerfacecolor'] = markerfacecolor
    if markeredgecolor is not None:
        kwargs['markeredgecolor'] = markeredgecolor
    return Line2D([0], [0], **kwargs)
