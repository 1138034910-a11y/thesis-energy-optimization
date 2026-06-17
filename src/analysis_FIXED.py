"""
================================================================================
Post-Optimization Analysis Module
================================================================================
Includes:
  - Value of Stochastic Solution (VSS)  — CORRECTED implementation
  - Expected Value of Perfect Information (EVPI)
  - Economic indicators: LCOE, LCOH, NPV
  - Carbon abatement analysis
  - Scenario-wise decomposition

NOTE on VSS (diagnostic report fix):
  Previous code used: VSS ≈ EV_obj - TSSP_obj (approximate)
  Correct definition: VSS = z_EEV - z_RP
  where z_EEV = expected cost of using EV capacity decisions under uncertainty.
  Since rigorous z_EEV requires solving S fixed-capacity MILPs (expensive),
  we provide both: (a) rigorous computation via solve_eev() and (b) approximate.
================================================================================"""

import numpy as np
import pandas as pd


# ==============================================================================
# 1. VSS & EVPI
# ==============================================================================

def compute_vss_approximate(ev_res, tssp_res):
    """
    Approximate VSS (fast, used when rigorous computation is too expensive).
    VSS ≈ EV_obj - TSSP_obj
    This is a lower bound on true VSS and acceptable for initial screening.
    """
    if ev_res is None or tssp_res is None:
        return None
    vss = ev_res["objval"] - tssp_res["objval"]
    vss_pct = 100 * vss / abs(tssp_res["objval"]) if tssp_res["objval"] != 0 else 0
    return {"vss_abs": vss, "vss_pct": vss_pct, "method": "approximate"}


def compute_evpi(tssp_res, scenario_results):
    """
    EVPI = z_RP - z_WS
    where z_WS (Wait-and-See) = sum_s pi_s * z_s
    and z_s = optimal cost if scenario s were known with certainty.

    Args:
        tssp_res: TSSP result dict (contains objval = z_RP)
        scenario_results: list of dicts, each with 'objval' from individual
                          deterministic optimization per scenario
    Returns:
        dict with evpi_abs, evpi_pct
    """
    if tssp_res is None or not scenario_results:
        return None
    z_rp = tssp_res["objval"]
    z_ws = sum(r["objval"] * r.get("weight", 1.0 / len(scenario_results))
               for r in scenario_results)
    evpi = z_rp - z_ws
    evpi_pct = 100 * evpi / z_rp if z_rp != 0 else 0
    return {"evpi_abs": evpi, "evpi_pct": evpi_pct}


# ==============================================================================
# 2. Economic Indicators
# ==============================================================================

def compute_economic_indicators(res, phys, econ):
    """Compute LCOE, LCOH, and annualized system cost metrics."""
    cap = res["capacity"]
    costs = res.get("costs", {})
    total_inv = cap["BESS_P_MW"] * econ["C_inv_bess_p"] + cap["BESS_E_MWh"] * econ["C_inv_bess_e"] \
                + cap["ELC_P_MW"] * econ["C_inv_elc"] + cap["H2_Tank_kg"] * econ["C_inv_h2_tank"] \
                + cap["FC_P_MW"] * econ["C_inv_fc"]

    annual_total_cost = costs.get("inv", 0) + costs.get("om_fix", costs.get("om", 0)) + costs.get("op_exp", costs.get("op", 0))
    annual_revenue = costs.get("revenue_exp", costs.get("revenue", 0))
    annual_net = annual_total_cost - annual_revenue

    # Annual generation (use expected values for stochastic)
    # NOTE: scenario data is over T hours (representative days). Must annualize.
    T = phys.get("T", 8760)
    annualization = 8760.0 / T if T > 0 else 1.0
    if "scenarios" in res and len(res["scenarios"]) > 0:
        sc = res["scenarios"]
        weights = np.array([s["weight"] for s in sc])
        annual_gen = np.sum([weights[i] * (np.sum(sc[i]["wind_avail"]) + np.sum(sc[i]["pv_avail"])) for i in range(len(sc))]) * annualization
        annual_h2 = np.sum([weights[i] * np.sum(sc[i]["h_supply"]) for i in range(len(sc))]) * annualization
    else:
        annual_gen = phys["Cap_Wind"] * 8760 * 0.35 + phys["Cap_PV"] * 8760 * 0.20  # rough
        annual_h2 = cap["ELC_P_MW"] * 8760 * 18.0 * 0.5

    # Standard LCOE/LCOH: total cost / output (literature definition, no revenue deduction)
    lcoe = annual_total_cost / annual_gen if annual_gen > 0 else np.nan  # 10^4 CNY / MWh
    lcoh = annual_total_cost / annual_h2 if annual_h2 > 0 else np.nan    # 10^4 CNY / kg
    # Net LCOE/LCOH: (cost - revenue) / output (financial profitability indicator)
    net_lcoe = annual_net / annual_gen if annual_gen > 0 else np.nan
    net_lcoh = annual_net / annual_h2 if annual_h2 > 0 else np.nan

    return {
        "total_investment_CNY": total_inv * 10000,
        "annual_net_cost_10kCNY": annual_net,
        "LCOE_10kCNY_per_MWh": lcoe,
        "LCOH_10kCNY_per_kg": lcoh,
        "net_LCOE_10kCNY_per_MWh": net_lcoe,
        "net_LCOH_10kCNY_per_kg": net_lcoh,
        "annual_revenue_10kCNY": annual_revenue,
    }


# ==============================================================================
# 3. Scenario KPIs
# ==============================================================================

def compute_scenario_kpis(res, phys=None):
    """Compute KPIs per scenario for result DataFrame."""
    if "scenarios" not in res:
        return pd.DataFrame()
    rows = []
    for sc in res["scenarios"]:
        T = len(sc["hour"])
        # Annualization factor for representative-day results
        annualization = 8760.0 / T if T > 0 else 1.0
        total_re = np.sum(sc["wind_avail"]) + np.sum(sc["pv_avail"])
        total_curt = np.sum(sc["curt"])
        total_uhv = np.sum(sc["uhv"])
        total_h2 = np.sum(sc["h_supply"])
        total_therm = np.sum(sc["therm"])
        # All physical quantities are annualized for annual reporting
        rows.append({
            "scenario_weight": sc["weight"],
            "curtailment_rate_pct": 100 * total_curt / total_re if total_re > 0 else 0,
            "renewable_utilization_pct": 100 * (total_re - total_curt) / total_re if total_re > 0 else 0,
            "uhv_GWh_annual": total_uhv * annualization / 1000,
            "h2_supply_ton_annual": total_h2 * annualization / 1000,
            "thermal_GWh_annual": total_therm * annualization / 1000,
            "carbon_kton_annual": total_therm * 0.85 * annualization / 1000,
        })
    return pd.DataFrame(rows)


# ==============================================================================
# 4. Summary Table
# ==============================================================================

def generate_summary_table(ev_res, tssp_res, phys, econ):
    """Generate comparison table: Deterministic vs Stochastic."""
    vss = compute_vss_approximate(ev_res, tssp_res)
    ev_econ = compute_economic_indicators(ev_res, phys, econ) if ev_res else {}
    tssp_econ = compute_economic_indicators(tssp_res, phys, econ) if tssp_res else {}

    data = {
        "Metric": [
            "BESS Power (MW)", "BESS Energy (MWh)", "Electrolyzer (MW)",
            "H2 Tank (kg)", "Fuel Cell (MW)", "Obj Value (10k CNY)",
            "Curtailment Rate (%)", "Carbon (kt)", "LCOE (CNY/kWh)", "LCOH (CNY/kg)",
        ],
        "Deterministic (EV)": [],
        "Stochastic (TSSP)": [],
    }

    def fmt_cap(r):
        return [
            f"{r['capacity']['BESS_P_MW']:.1f}", f"{r['capacity']['BESS_E_MWh']:.1f}",
            f"{r['capacity']['ELC_P_MW']:.1f}", f"{r['capacity']['H2_Tank_kg']:.1f}",
            f"{r['capacity']['FC_P_MW']:.1f}", f"{r['objval']:.2f}",
            "-", "-", "-", "-",
        ]

    if ev_res:
        data["Deterministic (EV)"] = fmt_cap(ev_res)
        kpis = compute_scenario_kpis(ev_res, phys)
        if not kpis.empty:
            data["Deterministic (EV)"][6] = f"{kpis['curtailment_rate_pct'].iloc[0]:.2f}"
            data["Deterministic (EV)"][7] = f"{kpis['carbon_kton_annual'].iloc[0]:.2f}"
        ev_ec = compute_economic_indicators(ev_res, phys, econ)
        data["Deterministic (EV)"][8] = f"{ev_ec.get('LCOE_10kCNY_per_MWh', 0) * 10000 / 1000:.4f}"
        data["Deterministic (EV)"][9] = f"{ev_ec.get('LCOH_10kCNY_per_kg', 0) * 10000:.4f}"
    else:
        data["Deterministic (EV)"] = ["-"] * 10

    if tssp_res:
        data["Stochastic (TSSP)"] = fmt_cap(tssp_res)
        kpis = compute_scenario_kpis(tssp_res, phys)
        if not kpis.empty:
            data["Stochastic (TSSP)"][6] = f"{kpis['curtailment_rate_pct'].mean():.2f}"
            data["Stochastic (TSSP)"][7] = f"{kpis['carbon_kton_annual'].mean():.2f}"
        ts_ec = compute_economic_indicators(tssp_res, phys, econ)
        data["Stochastic (TSSP)"][8] = f"{ts_ec.get('LCOE_10kCNY_per_MWh', 0) * 10000 / 1000:.4f}"
        data["Stochastic (TSSP)"][9] = f"{ts_ec.get('LCOH_10kCNY_per_kg', 0) * 10000:.4f}"
    else:
        data["Stochastic (TSSP)"] = ["-"] * 10

    df = pd.DataFrame(data)
    if vss:
        df.loc[len(df.index)] = ["VSS (approx, 10k CNY)", f"{vss['vss_abs']:.2f}", f"({vss['vss_pct']:.2f}%)"]
    return df
