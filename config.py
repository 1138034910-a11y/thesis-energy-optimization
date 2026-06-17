"""
================================================================================
Project: KAN-Enhanced Two-Stage Stochastic Optimization for
         Electric-Hydrogen-Storage-Transportation Systems
Version: 2.0 (Production-Ready for SCI Q1/Q2 Journals)
Target Journals: Applied Energy, Energy, IJEPES, Renewable Energy
================================================================================
All monetary units: 10^4 CNY (万元)
All power units: MW / MWh
All hydrogen units: kg
"""

import numpy as np

# ==============================================================================
# 1. ECONOMIC PARAMETERS
# ==============================================================================
EconParams = {
    # Discount rate & lifetime
    "r_discount": 0.06,
    "N_bess": 15,       # Battery lifetime (years)
    "N_elc": 15,        # Electrolyzer lifetime
    "N_h2_tank": 20,    # H2 tank lifetime
    "N_fc": 15,         # Fuel cell lifetime
    "N_therm": 30,      # Thermal plant lifetime

    # CAPEX (10^4 CNY / unit)
    "C_inv_bess_p": 80.0,      # BESS power
    "C_inv_bess_e": 120.0,     # BESS energy
    "C_inv_elc": 150.0,        # Electrolyzer
    "C_inv_h2_tank": 0.18,     # H2 storage (per kg capacity)
    "C_inv_fc": 220.0,         # Fuel cell

    # Fixed O&M (% of CAPEX annually)
    "rate_om_bess": 0.015,
    "rate_om_elc": 0.030,
    "rate_om_h2_tank": 0.010,
    "rate_om_fc": 0.020,

    # Variable O&M & fuel
    "var_om_bess": 0.005,      # 10^4 CNY / MWh
    "var_om_elc": 0.002,
    "var_om_fc": 0.004,
    "C_coal": 0.030,           # 10^4 CNY / MWh
    "C_start": 5.0,            # Thermal start cost
    "C_stop": 0.5,             # Thermal stop cost
    "C_start_elc": 1.0,
    "C_stop_elc": 0.2,
    "C_pen_curt": 0.015,       # Curtailment penalty (0.15 CNY/kWh = opportunity cost of renewable FIT)

    # Revenue
    "Price_grid": 0.030,       # UHV transmission price (10^4 CNY / MWh)
    "Price_h2": 0.0015,        # H2 sale price (10^4 CNY / kg)
    "Price_h2_green_premium": 0.0003,  # Green H2 premium

    # Carbon cost (NEW for high-impact journals)
    "Carbon_price": 0.008,     # 10^4 CNY / ton CO2
    "EF_coal": 0.85,           # tCO2 / MWh
    "Carbon_cap_annual": 15000000.0,  # Annual carbon cap (15 Mt) — ~45% of 10GW thermal baseline, strict but feasible
    "Carbon_cap_sweep": [5000000, 10000000, 15000000, 20000000, 25000000, 30000000],  # 5-30 Mt sensitivity range
}

# ==============================================================================
# 2. PHYSICAL & TECHNICAL PARAMETERS
# ==============================================================================
PhysParams = {
    # Time horizon
    "T": 8760,
    "dt": 1.0,

    # Installed capacity (fixed)
    "Cap_Wind": 75000.0,
    "Cap_PV": 65000.0,
    "Cap_Therm": 10000.0,  # Reduced from 30 GW to 10 GW for carbon feasibility
    "P_UHV_Max": 8000.0,
    "Load_Base": 13000.0,

    # Planning upper bounds
    "Cap_BESS_P_Max": 30000.0,
    "Cap_BESS_E_Max": 150000.0,
    "Cap_ELC_P_Max": 35000.0,
    "Cap_H2_Tank_Max": 400000.0,    # 400 t — relaxed for better model tractability and Gap convergence
    "Cap_FC_P_Max": 12000.0,

    # Thermal plant
    "Rate_Therm_Min": 0.30,
    "Rate_Ramp_Therm": 0.20,
    "Min_Up_Time": 12,
    "Min_Down_Time": 12,

    # BESS
    "Eta_BESS_Ch": 0.95,
    "Eta_BESS_Dis": 0.95,
    "SOC_Min": 0.10,
    "SOC_Max": 0.90,
    "SOC_Init": 0.50,
    "Self_Discharge": 0.00002,    # ~0.05% per day (was 0.1%/h, unrealistically high)
    "BESS_Min_Duration": 2.0,

    # Electrolyzer (fixed average efficiency based on literature [9])
    "ELC_Min_Load_Ratio": 0.25,
    "Rate_Ramp_ELC": 0.40,
    # Fixed average H2 production efficiency (kg/MWh)
    "ELC_Efficiency": 17.75,

    # H2 storage
    "H2_Tank_Init_Ratio": 0.40,
    "H2_Tank_Min_Ratio": 0.10,
    "H2_Tank_Max_Ratio": 0.95,
    "H2_Tank_Loss": 0.0002,

    # Fuel cell
    "Eta_FC_Power": 0.0165,    # MWh / kg
    "FC_Min_Load_Ratio": 0.15,

    # Downstream H2 demand
    "H2_Demand_Min": 100000.0,
    "H2_Demand_Max": 300000.0,
    "H2_Delivery_Ramp": 30000.0,

    # UHV banded transmission
    "Rate_Ramp_UHV": 0.40,  # Sufficient for band transitions
    "Min_UHV_Energy": 8000.0 * 4000.0,

    # Renewable accommodation constraints
    "Limit_Curt_Rate": 0.12,  # Adjusted for realistic winter operation
    # "Limit_RPS": 0.30,  # REMOVED: redundant with Cap_Therm=10GW (never binding)
    "Min_Util_Hours": 1800,
}

# ==============================================================================
# 3. KAN PROBABILISTIC FORECASTING PARAMETERS
# ==============================================================================
KANParams = {
    "lookback": 168,           # Use past 7 days (168h) for prediction
    "hidden_size": 64,
    "num_layers": 2,           # Number of KAN layers
    "grid_size": 10,           # B-spline grid size
    "spline_order": 3,
    "epochs": 40,
    "lr": 0.001,
    "batch_size": 256,
    "train_ratio": 0.8,
    "device": "cpu",           # Auto-detected in code
}

# ==============================================================================
# 4. SCENARIO GENERATION PARAMETERS
# ==============================================================================
ScenarioParams = {
    "N_sample": 200,           # Monte Carlo samples from predictive distribution
    "N_scenario": 4,           # Reduced from 8 for tractability on local hardware
    "method": "kmeans",        # "kmeans" or "forward_selection"
    "seed": 42,
}

# ==============================================================================
# 4b. REPRESENTATIVE DAYS (Time Aggregation for Tractability)
# ==============================================================================
RepDayParams = {
    "enabled": True,           # Enable representative day reduction
    "n_days": 20,              # Number of representative days (20×24=480h)
    "T_day": 24,               # Hours per day
    "seed": 42,
}

# ==============================================================================
# 5. STOCHASTIC OPTIMIZATION SOLVER SETTINGS
# ==============================================================================
SolverParams = {
    "MIPGap": 0.02,            # Target 1% optimality gap for publication-quality results
    "TimeLimit": 14400,        # 4 hours per solve (TSSP needs more time for tight gap)
    "Threads": 0,              # Use all cores
    "MIPFocus": 1,             # 1=find feasible solutions faster (good upper bound helps Gap)
    "Heuristics": 0.5,         # Moderate heuristic intensity
    "Presolve": 2,             # Aggressive presolve
    "Cuts": 2,                 # Moderate cuts (Cuts=3 may be too heavy for 480h×4scen MILP)
    "Crossover": 0,            # Disable crossover (barrier -> basis is too slow for huge models)
    "ImproveStartGap": 0.10,   # Start solution improvement heuristics when Gap < 10%
}

# ==============================================================================
# 6. PLOTTING & PUBLICATION SETTINGS (Nature-style)
# ==============================================================================
PlotParams = {
    "dpi": 300,
    "fig_width_full": 7.08,    # inches = 180 mm (Nature full column)
    "fig_width_double": 7.48,  # inches = 190 mm (Nature double column)
    "font_family": "sans-serif",
    "font_sans": ["Arial", "DejaVu Sans", "Liberation Sans"],
    "font_size": 8,            # Nature standard
    "font_size_small": 7,
    "font_size_large": 9,
    "line_width": 1.0,
    "marker_size": 3,
    "color_palette": {
        "wind": "#4C78A8",
        "pv": "#F58518",
        "thermal": "#E45756",
        "bess": "#72B7B2",
        "h2": "#54A24B",
        "uhv": "#9D755D",
        "fc": "#B279A2",
        "curt": "#BAB0AC",
        "load": "#2D2D2D",
        "grid": "#A0A0A0",
    },
}

# ==============================================================================
# 7. DATA PATHS
# ==============================================================================
DataPaths = {
    "wind_pred": "data/甘肃_风电_prediction_result.csv",
    "solar_pred": "data/甘肃_光伏_prediction_result.csv",
    "wind_raw": "data/甘肃省风电.xlsx",
    "solar_raw": "data/甘肃省光电.xlsx",
}


# ==============================================================================
# Utility Functions
# ==============================================================================
def get_crf(r, n):
    """Capital Recovery Factor"""
    if n <= 0:
        raise ValueError("n must be positive")
    if abs(r) < 1e-12:
        return 1.0 / n
    return (r * (1 + r) ** n) / ((1 + r) ** n - 1)


def build_uhv_band_profiles(T, p_uhv_max):
    """Construct UHV banded transmission profiles."""
    p_min, p_max = [], []
    for t in range(T):
        hour = t % 24
        if 10 <= hour <= 16:
            min_ratio, max_ratio = 0.55, 1.00
        elif hour >= 20 or hour <= 5:
            min_ratio, max_ratio = 0.45, 0.85
        else:
            min_ratio, max_ratio = 0.25, 0.70
        p_min.append(min_ratio * p_uhv_max)
        p_max.append(max_ratio * p_uhv_max)
    return np.array(p_min), np.array(p_max)


def build_load_profile(T, load_base):
    """Build annual load profile from normalized daily curve."""
    daily = np.array([
        0.65, 0.62, 0.60, 0.58, 0.59, 0.63, 0.75, 0.88,
        0.96, 1.00, 0.98, 0.95, 0.92, 0.90, 0.93, 0.96,
        1.00, 0.99, 0.97, 0.92, 0.85, 0.78, 0.72, 0.68
    ])
    return np.array([daily[t % 24] * load_base for t in range(T)], dtype=float)
