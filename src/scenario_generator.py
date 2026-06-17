"""
================================================================================
Scenario Generation & Reduction for Stochastic Optimization
================================================================================
Workflow:
  1. Extract predictive distribution (mu, sigma) from KAN forecaster
  2. Sample N_sample scenarios via Monte Carlo from N(mu, sigma^2)
  3. Reduce to N_scenario representative scenarios using K-means++
  4. Output scenario weights and reduced scenario matrix

NEW (v2.1 - diagnostic report fix):
  - Added Gaussian Copula for joint wind-solar correlation modeling
  - Historical data used to estimate Kendall's tau correlation
  - Fallback to independent sampling if correlation data unavailable

Reference:
- Dupačová et al. (2003). Scenario reduction in stochastic programming.
- Pflug & Pichler (2015). Dynamic Generation of Scenario Trees.
- Martins & Munda (2025). TAKSR: Temporal-Aware K-Means Scenario Reduction.
================================================================================"""

import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import norm, kendalltau


# ==============================================================================
# 1. Independent Monte Carlo (original, kept as fallback)
# ==============================================================================

def monte_carlo_scenarios(mu, sigma, n_samples=200, seed=42):
    """
    Generate scenarios by sampling from independent Gaussian distributions
    at each time step: s_t ~ N(mu_t, sigma_t^2).
    """
    rng = np.random.default_rng(seed)
    T = len(mu)
    scenarios = np.zeros((n_samples, T))
    for t in range(T):
        scenarios[:, t] = rng.normal(loc=mu[t], scale=sigma[t], size=n_samples)
    scenarios = np.clip(scenarios, 0.0, 1.0)
    return scenarios


# ==============================================================================
# 2. Gaussian Copula for Joint Wind-Solar Scenarios (NEW)
# ==============================================================================

def estimate_kendall_tau(wind_actual, solar_actual):
    """
    Estimate Kendall's rank correlation between wind and solar from historical data.
    Returns: tau (scalar, -1 to 1)
    """
    tau, _ = kendalltau(wind_actual, solar_actual)
    if np.isnan(tau):
        tau = 0.0
    return np.clip(tau, -0.99, 0.99)


def pearson_to_gaussian_corr(tau):
    """
    Approximate Gaussian Copula correlation parameter from Kendall's tau.
    For Gaussian Copula: rho = sin(pi/2 * tau)
    """
    return np.sin(np.pi / 2.0 * tau)


def monte_carlo_scenarios_copula(mu_wind, sigma_wind, mu_solar, sigma_solar,
                                  n_samples=200, seed=42,
                                  historical_wind=None, historical_solar=None,
                                  rho_override=None):
    """
    Generate joint wind-solar scenarios using Gaussian Copula.

    Args:
        mu_wind, sigma_wind: (T,) predictive distribution for wind
        mu_solar, sigma_solar: (T,) predictive distribution for solar
        n_samples: number of scenarios to generate
        seed: random seed
        historical_wind, historical_solar: (T,) actual historical series for correlation estimation
        rho_override: if provided, use this Pearson correlation directly instead of estimating

    Returns:
        wind_scenarios: (n_samples, T)
        solar_scenarios: (n_samples, T)
    """
    rng = np.random.default_rng(seed)
    T = len(mu_wind)

    # Step 1: Estimate correlation
    if rho_override is not None:
        rho = rho_override
    elif historical_wind is not None and historical_solar is not None:
        tau = estimate_kendall_tau(historical_wind, historical_solar)
        rho = pearson_to_gaussian_corr(tau)
        print(f"  [Copula] Estimated Kendall's tau = {tau:.4f} -> Gaussian rho = {rho:.4f}")
    else:
        rho = 0.0
        print(f"  [Copula] No historical data provided, using rho = 0.0 (independent)")

    # Step 2: Build correlation matrix
    corr_matrix = np.array([[1.0, rho], [rho, 1.0]])

    # Step 3: Cholesky decomposition for sampling
    try:
        L = np.linalg.cholesky(corr_matrix)
    except np.linalg.LinAlgError:
        # Fallback if correlation matrix is not positive definite
        print("  [Copula] Warning: Correlation matrix not PD, using independent sampling.")
        wind_all = monte_carlo_scenarios(mu_wind, sigma_wind, n_samples=n_samples, seed=seed)
        solar_all = monte_carlo_scenarios(mu_solar, sigma_solar, n_samples=n_samples, seed=seed+1)
        return wind_all, solar_all

    # Step 4: Sample correlated standard normals
    # z: (n_samples, T, 2) -> for each time t, sample a bivariate normal
    wind_scenarios = np.zeros((n_samples, T))
    solar_scenarios = np.zeros((n_samples, T))

    for t in range(T):
        # Standard normal samples for this time step
        z = rng.standard_normal(size=(n_samples, 2))  # (n_samples, 2)
        # Correlate them
        u = z @ L.T  # (n_samples, 2)
        # Transform to uniform via CDF
        u_wind = norm.cdf(u[:, 0])
        u_solar = norm.cdf(u[:, 1])
        # Inverse transform to target Gaussians
        wind_scenarios[:, t] = norm.ppf(u_wind) * sigma_wind[t] + mu_wind[t]
        solar_scenarios[:, t] = norm.ppf(u_solar) * sigma_solar[t] + mu_solar[t]

    # Clip to physical bounds
    wind_scenarios = np.clip(wind_scenarios, 0.0, 1.0)
    solar_scenarios = np.clip(solar_scenarios, 0.0, 1.0)

    return wind_scenarios, solar_scenarios


# ==============================================================================
# 3. Scenario Reduction via K-means++
# ==============================================================================

def scenario_reduction_kmeans(scenarios, n_scenarios, seed=42):
    """
    Reduce N_sample scenarios to N_scenario via K-means++ clustering.
    Cluster centers serve as reduced scenarios; weights = cluster proportions.
    """
    n_samples, T = scenarios.shape
    kmeans = KMeans(n_clusters=n_scenarios, init="k-means++", random_state=seed, n_init=10)
    labels = kmeans.fit_predict(scenarios)
    centers = kmeans.cluster_centers_
    weights = np.array([np.sum(labels == i) / n_samples for i in range(n_scenarios)])
    weights = weights / weights.sum()
    return centers, weights


# ==============================================================================
# 4. Main Entry
# ==============================================================================

def generate_reduced_scenarios(mu_wind, sigma_wind, mu_solar, sigma_solar,
                                n_sample=200, n_scenario=8, seed=42,
                                use_copula=True,
                                historical_wind=None, historical_solar=None,
                                rho_override=None):
    """
    Main entry: generate joint wind-solar scenarios.

    NEW v2.1 parameters:
        use_copula: if True, use Gaussian Copula; otherwise independent sampling
        historical_wind/solar: actual historical series for correlation estimation
        rho_override: manually specified Pearson correlation (-1 to 1)

    Returns:
        wind_scenarios: (n_scenario, T)
        solar_scenarios: (n_scenario, T)
        weights: (n_scenario,)
    """
    print(f"\n{'='*60}")
    print("Scenario Generation & Reduction")
    print(f"{'='*60}")

    if use_copula:
        print(f"  Mode: Gaussian Copula (n_sample={n_sample})")
        wind_all, solar_all = monte_carlo_scenarios_copula(
            mu_wind, sigma_wind, mu_solar, sigma_solar,
            n_samples=n_sample, seed=seed,
            historical_wind=historical_wind,
            historical_solar=historical_solar,
            rho_override=rho_override
        )
    else:
        print(f"  Mode: Independent sampling (n_sample={n_sample})")
        wind_all = monte_carlo_scenarios(mu_wind, sigma_wind, n_samples=n_sample, seed=seed)
        solar_all = monte_carlo_scenarios(mu_solar, sigma_solar, n_samples=n_sample, seed=seed + 1)

    # Combine wind+solar as joint feature vector for clustering
    print(f"  Reducing to {n_scenario} scenarios via K-means++...")
    joint = np.hstack([wind_all, solar_all])  # (n_sample, 2T)
    centers_joint, weights = scenario_reduction_kmeans(joint, n_scenario, seed=seed)

    T = len(mu_wind)
    wind_scenarios = centers_joint[:, :T]
    solar_scenarios = centers_joint[:, T:]

    # Re-clip after clustering
    wind_scenarios = np.clip(wind_scenarios, 0.0, 1.0)
    solar_scenarios = np.clip(solar_scenarios, 0.0, 1.0)

    print(f"  Reduced scenario weights: {np.round(weights, 4)}")
    print(f"  Wind scenario range: [{wind_scenarios.min():.3f}, {wind_scenarios.max():.3f}]")
    print(f"  Solar scenario range: [{solar_scenarios.min():.3f}, {solar_scenarios.max():.3f}]")

    return wind_scenarios, solar_scenarios, weights


def export_scenarios(wind_sc, solar_sc, weights, output_dir):
    """Export scenarios to CSV for inspection."""
    import os
    import pandas as pd
    os.makedirs(output_dir, exist_ok=True)
    T = wind_sc.shape[1]
    for i, w in enumerate(weights):
        df = pd.DataFrame({
            "hour": np.arange(1, T + 1),
            "wind_pu": wind_sc[i],
            "solar_pu": solar_sc[i],
            "weight": w,
        })
        df.to_csv(f"{output_dir}/scenario_{i+1}_w{w:.3f}.csv", index=False)
    print(f"  Scenarios exported to {output_dir}")
