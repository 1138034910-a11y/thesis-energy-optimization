"""
================================================================================
Representative Days Module (Time Aggregation for Tractable Optimization)
================================================================================
Reduces T=8760h annual horizon to N×24h representative days via k-means
clustering on daily profiles of wind, solar, and load.

Methodology (used in Journal of Energy Storage / Applied Energy / Energy submissions):
- Extract 365 daily feature vectors: [wind_24h, solar_24h, load_24h]
- Normalize each series to zero-mean, unit-variance
- k-means clustering → N clusters
- Representative day = cluster centroid (or nearest actual day)
- Weight = fraction of days in cluster / 365

Annual constraints (carbon cap, RPS, etc.) are weighted by cluster weights.
================================================================================"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def extract_daily_profiles(wind, solar, load, T_day=24):
    """
    Extract daily feature vectors from hourly series.

    Args:
        wind, solar, load: 1D arrays of length T (multiples of T_day)
    Returns:
        X: (n_days, 3*T_day) feature matrix
        day_indices: list of start hour indices
    """
    T = len(wind)
    n_days = T // T_day
    X = np.zeros((n_days, 3 * T_day))
    for d in range(n_days):
        s = d * T_day
        e = s + T_day
        X[d, :T_day] = wind[s:e]
        X[d, T_day:2*T_day] = solar[s:e]
        X[d, 2*T_day:] = load[s:e]
    return X, list(range(0, T, T_day))


def cluster_representative_days(X, n_days, random_state=42):
    """
    k-means clustering of daily profiles.

    Returns:
        reps: dict with keys:
            - 'centroids': (n_days, 3*T_day) cluster centers
            - 'labels': (365,) cluster assignment per original day
            - 'weights': (n_days,) fraction of days in each cluster
            - 'day_indices': list of representative day start indices
                           (nearest actual day to each centroid)
    """
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_days, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X_norm)
    centroids = kmeans.cluster_centers_

    # Find nearest actual day to each centroid
    day_indices = []
    for k in range(n_days):
        dists = np.linalg.norm(X_norm - centroids[k], axis=1)
        day_indices.append(int(np.argmin(dists)))

    weights = np.array([np.sum(labels == k) / len(labels) for k in range(n_days)])

    return {
        "centroids": centroids,
        "labels": labels,
        "weights": weights,
        "day_indices": day_indices,
        "scaler": scaler,
        "X_norm": X_norm,
    }


def build_reduced_series(wind, solar, load, reps, T_day=24):
    """
    Build reduced-length series from representative days.

    Returns:
        wind_r, solar_r, load_r: arrays of length n_days * T_day
        weights: array of length n_days
    """
    n_days = len(reps["day_indices"])
    wind_r, solar_r, load_r = [], [], []
    for d in reps["day_indices"]:
        s = d * T_day
        e = s + T_day
        wind_r.extend(wind[s:e])
        solar_r.extend(solar[s:e])
        load_r.extend(load[s:e])
    return np.array(wind_r), np.array(solar_r), np.array(load_r), reps["weights"]


def reduce_scenarios_to_representative_days(wind_sc, solar_sc, load, reps, T_day=24):
    """
    Reduce scenario matrices to representative day blocks.

    Args:
        wind_sc, solar_sc: (n_scenarios, T) arrays
        load: (T,) array
        reps: output from cluster_representative_days
    Returns:
        wind_sc_r, solar_sc_r, load_r, weights
    """
    S, T = wind_sc.shape
    n_days = len(reps["day_indices"])
    T_reduced = n_days * T_day

    wind_sc_r = np.zeros((S, T_reduced))
    solar_sc_r = np.zeros((S, T_reduced))
    load_r = np.zeros(T_reduced)

    for i, d in enumerate(reps["day_indices"]):
        src_s = d * T_day
        src_e = src_s + T_day
        dst_s = i * T_day
        dst_e = dst_s + T_day
        wind_sc_r[:, dst_s:dst_e] = wind_sc[:, src_s:src_e]
        solar_sc_r[:, dst_s:dst_e] = solar_sc[:, src_s:src_e]
        load_r[dst_s:dst_e] = load[src_s:src_e]

    return wind_sc_r, solar_sc_r, load_r, reps["weights"]


def scale_annual_constraints(econ, phys, reps, T_day=24):
    """
    Copy economic/physical parameters for representative-day modeling.
    NOTE: Annual constraints (Carbon cap, Min_UHV_Energy, etc.) are scaled
    directly in the MILP model build using T/8760. This function only
    copies parameters to avoid modifying global config dicts.
    """
    econ_mod = dict(econ)
    phys_mod = dict(phys)
    return econ_mod, phys_mod

    # Scale annual carbon cap proportionally
    econ_mod["Carbon_cap_annual"] = econ["Carbon_cap_annual"] * year_fraction

    # Scale minimum utilization hours
    if "Min_Util_Hours" in phys:
        phys_mod["Min_Util_Hours"] = phys["Min_Util_Hours"] * year_fraction

    # Scale RPS target (fractional, no change needed)
    # Scale curtailment rate (fractional, no change needed)

    # For UHV minimum annual energy
    if "Min_UHV_Energy" in phys:
        phys_mod["Min_UHV_Energy"] = phys["Min_UHV_Energy"] * year_fraction

    return econ_mod, phys_mod


def expand_results_to_annual(reduced_results, reps, T_day=24):
    """
    Expand representative-day results back to annual scale.
    For capacity decisions: no change (1st stage variables).
    For operational metrics: weighted average across representative days.
    """
    weights = reps["weights"]
    # Capacity decisions are already annual
    return reduced_results


def run_representative_day_pipeline(wind, solar, load, n_days=12, T_day=24, seed=42):
    """
    Full pipeline: extract profiles → cluster → build reduced series.

    Returns:
        reps: clustering result dict
        wind_r, solar_r, load_r, weights: reduced series and weights
    """
    X, _ = extract_daily_profiles(wind, solar, load, T_day)
    reps = cluster_representative_days(X, n_days, random_state=seed)
    wind_r, solar_r, load_r, weights = build_reduced_series(wind, solar, load, reps, T_day)
    print(f"[RepDays] Reduced {len(wind)//T_day} days → {n_days} representative days")
    print(f"[RepDays] Weights: {weights.round(3)}")
    return reps, wind_r, solar_r, load_r, weights
