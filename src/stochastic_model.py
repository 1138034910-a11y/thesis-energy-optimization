"""
================================================================================
Two-Stage Stochastic Programming (TSSP) for Electric-Hydrogen-Storage-Transportation
================================================================================
First-stage (here-and-now): Capacity planning decisions
Second-stage (wait-and-see): Operational decisions per scenario

Novel features for Q1/Q2 journal impact:
  1. Carbon emission cap & carbon pricing
  2. Fixed average electrolyzer efficiency (17.75 kg/MWh)
  3. Green hydrogen premium revenue
  4. Scenario-based expected cost minimization

Solver: Gurobi (MILP)
================================================================================
"""

import os
import numpy as np
import gurobipy as gp
from gurobipy import GRB


def build_two_stage_model(load_profile,
                          wind_scenarios, solar_scenarios, weights,
                          econ, phys, solver_cfg):
    """
    Build the full two-stage stochastic MILP.

    Parameters:
        load_profile: (T,)
        wind_scenarios: (S, T)
        solar_scenarios: (S, T)
        weights: (S,) probabilities
        econ, phys: dicts from config
        solver_cfg: dict
    """
    T = phys["T"]
    S = len(weights)
    m = gp.Model("TSSP_RE_H2_Storage_UHV")
    m.setParam("OutputFlag", 1)
    m.setParam("MIPGap", solver_cfg.get("MIPGap", 0.05))
    m.setParam("TimeLimit", solver_cfg.get("TimeLimit", 7200))
    m.setParam("Threads", solver_cfg.get("Threads", 0))
    m.setParam("MIPFocus", solver_cfg.get("MIPFocus", 1))
    m.setParam("Heuristics", solver_cfg.get("Heuristics", 0.3))
    m.setParam("Presolve", solver_cfg.get("Presolve", 2))
    m.setParam("Cuts", solver_cfg.get("Cuts", 2))
    m.setParam("Crossover", solver_cfg.get("Crossover", 0))
    if "ImproveStartGap" in solver_cfg:
        m.setParam("ImproveStartGap", solver_cfg["ImproveStartGap"])

    # ==========================================================================
    # FIRST-STAGE VARIABLES (Capacity Planning)
    # ==========================================================================
    x_bess_p = m.addVar(lb=0, ub=phys["Cap_BESS_P_Max"], name="Cap_BESS_P")
    x_bess_e = m.addVar(lb=0, ub=phys["Cap_BESS_E_Max"], name="Cap_BESS_E")
    x_elc_p = m.addVar(lb=0, ub=phys["Cap_ELC_P_Max"], name="Cap_ELC_P")
    x_h2_tank = m.addVar(lb=0, ub=phys["Cap_H2_Tank_Max"], name="Cap_H2_Tank")
    x_fc_p = m.addVar(lb=0, ub=phys["Cap_FC_P_Max"], name="Cap_FC_P")

    # ==========================================================================
    # SECOND-STAGE VARIABLES (Operational, per scenario s)
    # ==========================================================================
    # Helper to create scenario-indexed vars
    def add_svars(name, lb=0, ub=None, vtype=GRB.CONTINUOUS):
        if ub is None:
            return m.addVars(S, T, lb=lb, vtype=vtype, name=name)
        return m.addVars(S, T, lb=lb, ub=ub, vtype=vtype, name=name)

    p_therm = add_svars("P_Therm")
    p_bess_ch = add_svars("P_BESS_Ch")
    p_bess_dis = add_svars("P_BESS_Dis")
    e_bess = add_svars("E_BESS")

    p_elc = add_svars("P_ELC")
    h_prod = add_svars("H2_Prod")
    h_tank = add_svars("H2_Tank")
    h_supply = add_svars("H2_Supply")
    h_fc_use = add_svars("H2_Use_FC")
    p_fc = add_svars("P_FC")

    p_uhv = add_svars("P_UHV", ub=phys["P_UHV_Max"])
    p_curt = add_svars("P_Curt")

    # Binary variables
    u_therm = add_svars("u_Therm", vtype=GRB.BINARY)
    y_start = add_svars("y_Start", vtype=GRB.BINARY)
    z_stop = add_svars("z_Stop", vtype=GRB.BINARY)

    u_elc = add_svars("u_ELC", vtype=GRB.BINARY)
    y_elc_start = add_svars("y_ELC_Start", vtype=GRB.BINARY)
    z_elc_stop = add_svars("z_ELC_Stop", vtype=GRB.BINARY)

    u_bess_ch = add_svars("u_BESS_Ch", vtype=GRB.BINARY)
    u_fc = add_svars("u_FC", vtype=GRB.BINARY)

    # Electrolyzer online capacity linearization auxiliary
    x_elc_on = add_svars("Cap_ELC_On", ub=phys["Cap_ELC_P_Max"])

    m.update()

    # ==========================================================================
    # CONSTRAINTS (applied to ALL scenarios)
    # ==========================================================================
    # --- BESS ---
    for s in range(S):
        m.addConstr(e_bess[s, 0] == phys["SOC_Init"] * x_bess_e
                    + p_bess_ch[s, 0] * phys["Eta_BESS_Ch"]
                    - p_bess_dis[s, 0] / phys["Eta_BESS_Dis"], name=f"BESS_Init_{s}")
        for t in range(1, T):
            m.addConstr(e_bess[s, t] == e_bess[s, t - 1] * (1 - phys["Self_Discharge"])
                        + p_bess_ch[s, t] * phys["Eta_BESS_Ch"]
                        - p_bess_dis[s, t] / phys["Eta_BESS_Dis"], name=f"BESS_Bal_{s}_{t}")
        m.addConstr(e_bess[s, T - 1] == phys["SOC_Init"] * x_bess_e, name=f"BESS_End_{s}")
        for t in range(T):
            m.addConstr(e_bess[s, t] <= phys["SOC_Max"] * x_bess_e, name=f"BESS_SOCMax_{s}_{t}")
            m.addConstr(e_bess[s, t] >= phys["SOC_Min"] * x_bess_e, name=f"BESS_SOCMin_{s}_{t}")
            m.addConstr(p_bess_ch[s, t] <= x_bess_p, name=f"BESS_ChCap_{s}_{t}")
            m.addConstr(p_bess_dis[s, t] <= x_bess_p, name=f"BESS_DisCap_{s}_{t}")
            m.addConstr(p_bess_ch[s, t] <= phys["Cap_BESS_P_Max"] * u_bess_ch[s, t], name=f"BESS_ChStat_{s}_{t}")
            m.addConstr(p_bess_dis[s, t] <= phys["Cap_BESS_P_Max"] * (1 - u_bess_ch[s, t]), name=f"BESS_DisStat_{s}_{t}")
    m.addConstr(x_bess_e >= phys["BESS_Min_Duration"] * x_bess_p, name="BESS_Dur")

    # --- Thermal ---
    p_th_min = phys["Cap_Therm"] * phys["Rate_Therm_Min"]
    p_th_max = phys["Cap_Therm"]
    ramp_th = phys["Cap_Therm"] * phys["Rate_Ramp_Therm"]
    t_on, t_off = phys["Min_Up_Time"], phys["Min_Down_Time"]
    for s in range(S):
        m.addConstr(y_start[s, 0] - z_stop[s, 0] == u_therm[s, 0], name=f"Therm_Logic0_{s}")
        for t in range(1, T):
            m.addConstr(y_start[s, t] - z_stop[s, t] == u_therm[s, t] - u_therm[s, t - 1], name=f"Therm_Logic_{s}_{t}")
        for t in range(T):
            m.addConstr(p_therm[s, t] >= p_th_min * u_therm[s, t], name=f"Therm_Min_{s}_{t}")
            m.addConstr(p_therm[s, t] <= p_th_max * u_therm[s, t], name=f"Therm_Max_{s}_{t}")
        for t in range(1, T):
            m.addConstr(p_therm[s, t] - p_therm[s, t - 1] <= ramp_th * u_therm[s, t - 1] + p_th_min * y_start[s, t], name=f"Therm_RampUp_{s}_{t}")
            m.addConstr(p_therm[s, t - 1] - p_therm[s, t] <= ramp_th * u_therm[s, t] + p_th_min * z_stop[s, t], name=f"Therm_RampDown_{s}_{t}")
        for t in range(T):
            end_t = min(t + t_on, T)
            m.addConstr(gp.quicksum(u_therm[s, k] for k in range(t, end_t)) >= (end_t - t) * y_start[s, t], name=f"Therm_MinUp_{s}_{t}")
            end_t2 = min(t + t_off, T)
            m.addConstr(gp.quicksum(1 - u_therm[s, k] for k in range(t, end_t2)) >= (end_t2 - t) * z_stop[s, t], name=f"Therm_MinDown_{s}_{t}")

    # --- Electrolyzer with Fixed Average Efficiency ---
    # Ramp scales with installed capacity x_elc_p (LinExpr, no auxiliary var needed)
    elc_ramp = phys["Rate_Ramp_ELC"] * x_elc_p
    eta_h2 = phys["ELC_Efficiency"]
    for s in range(S):
        # Start/stop logic
        m.addConstr(y_elc_start[s, 0] - z_elc_stop[s, 0] == u_elc[s, 0], name=f"ELC_Logic0_{s}")
        for t in range(1, T):
            m.addConstr(y_elc_start[s, t] - z_elc_stop[s, t] == u_elc[s, t] - u_elc[s, t - 1], name=f"ELC_Logic_{s}_{t}")

        # Online capacity linearization
        for t in range(T):
            m.addConstr(x_elc_on[s, t] <= x_elc_p, name=f"ELC_On1_{s}_{t}")
            m.addConstr(x_elc_on[s, t] <= phys["Cap_ELC_P_Max"] * u_elc[s, t], name=f"ELC_On2_{s}_{t}")
            m.addConstr(x_elc_on[s, t] >= x_elc_p - phys["Cap_ELC_P_Max"] * (1 - u_elc[s, t]), name=f"ELC_On3_{s}_{t}")
            m.addConstr(p_elc[s, t] <= x_elc_on[s, t], name=f"ELC_Max_{s}_{t}")
            m.addConstr(p_elc[s, t] >= phys["ELC_Min_Load_Ratio"] * x_elc_on[s, t], name=f"ELC_Min_{s}_{t}")

        # Ramp
        for t in range(1, T):
            m.addConstr(p_elc[s, t] - p_elc[s, t - 1] <= elc_ramp, name=f"ELC_RampUp_{s}_{t}")
            m.addConstr(p_elc[s, t - 1] - p_elc[s, t] <= elc_ramp, name=f"ELC_RampDown_{s}_{t}")

        # H2 production via fixed efficiency
        for t in range(T):
            m.addConstr(h_prod[s, t] == eta_h2 * p_elc[s, t], name=f"H2_Prod_{s}_{t}")

    # --- H2 Storage ---
    for s in range(S):
        m.addConstr(h_tank[s, 0] == phys["H2_Tank_Init_Ratio"] * x_h2_tank * (1 - phys["H2_Tank_Loss"])
                    + h_prod[s, 0] - h_supply[s, 0] - h_fc_use[s, 0], name=f"H2_Init_{s}")
        for t in range(1, T):
            m.addConstr(h_tank[s, t] == h_tank[s, t - 1] * (1 - phys["H2_Tank_Loss"])
                        + h_prod[s, t] - h_supply[s, t] - h_fc_use[s, t], name=f"H2_Bal_{s}_{t}")
        m.addConstr(h_tank[s, T - 1] == phys["H2_Tank_Init_Ratio"] * x_h2_tank, name=f"H2_End_{s}")
        for t in range(T):
            m.addConstr(h_tank[s, t] <= phys["H2_Tank_Max_Ratio"] * x_h2_tank, name=f"H2_Max_{s}_{t}")
            m.addConstr(h_tank[s, t] >= phys["H2_Tank_Min_Ratio"] * x_h2_tank, name=f"H2_Min_{s}_{t}")
            m.addConstr(h_supply[s, t] >= phys["H2_Demand_Min"], name=f"H2_SupMin_{s}_{t}")
            m.addConstr(h_supply[s, t] <= phys["H2_Demand_Max"], name=f"H2_SupMax_{s}_{t}")
        for t in range(1, T):
            m.addConstr(h_supply[s, t] - h_supply[s, t - 1] <= phys["H2_Delivery_Ramp"], name=f"H2_RampUp_{s}_{t}")
            m.addConstr(h_supply[s, t - 1] - h_supply[s, t] <= phys["H2_Delivery_Ramp"], name=f"H2_RampDown_{s}_{t}")

    # --- Fuel Cell ---
    for s in range(S):
        for t in range(T):
            m.addConstr(p_fc[s, t] <= x_fc_p, name=f"FC_Max_{s}_{t}")
            # Linearized: p_fc >= FC_Min_Ratio * x_fc_p - M * (1 - u_fc)
            m.addConstr(p_fc[s, t] >= phys["FC_Min_Load_Ratio"] * x_fc_p - phys["FC_Min_Load_Ratio"] * phys["Cap_FC_P_Max"] * (1 - u_fc[s, t]), name=f"FC_Min_{s}_{t}")
            m.addConstr(p_fc[s, t] <= phys["Cap_FC_P_Max"] * u_fc[s, t], name=f"FC_Status_{s}_{t}")
            m.addConstr(p_fc[s, t] == h_fc_use[s, t] * phys["Eta_FC_Power"], name=f"FC_Conv_{s}_{t}")

    # --- UHV Banded Transmission ---
    p_uhv_min_band, p_uhv_max_band = _build_uhv_bands(T, phys["P_UHV_Max"])
    ramp_uhv = phys["P_UHV_Max"] * phys["Rate_Ramp_UHV"]
    for s in range(S):
        for t in range(T):
            m.addConstr(p_uhv[s, t] >= p_uhv_min_band[t], name=f"UHV_Min_{s}_{t}")
            m.addConstr(p_uhv[s, t] <= p_uhv_max_band[t], name=f"UHV_Max_{s}_{t}")
        for t in range(1, T):
            m.addConstr(p_uhv[s, t] - p_uhv[s, t - 1] <= ramp_uhv, name=f"UHV_RampUp_{s}_{t}")
            m.addConstr(p_uhv[s, t - 1] - p_uhv[s, t] <= ramp_uhv, name=f"UHV_RampDown_{s}_{t}")
        # Scale UHV minimum energy by time horizon ratio
        min_uhv_energy = phys["Min_UHV_Energy"] * (T / 8760.0)
        m.addConstr(gp.quicksum(p_uhv[s, t] for t in range(T)) >= min_uhv_energy, name=f"UHV_Annual_{s}")

    # --- Power Balance ---
    for s in range(S):
        for t in range(T):
            p_w = phys["Cap_Wind"] * wind_scenarios[s, t]
            p_pv = phys["Cap_PV"] * solar_scenarios[s, t]
            m.addConstr(
                p_w + p_pv + p_therm[s, t] + p_bess_dis[s, t] + p_fc[s, t]
                == load_profile[t] + p_uhv[s, t] + p_bess_ch[s, t] + p_elc[s, t] + p_curt[s, t],
                name=f"PBal_{s}_{t}"
            )

    # --- Renewable Accommodation ---
    for s in range(S):
        total_re = float(np.sum(phys["Cap_Wind"] * wind_scenarios[s] + phys["Cap_PV"] * solar_scenarios[s]))
        m.addConstr(gp.quicksum(p_curt[s, t] for t in range(T)) <= phys["Limit_Curt_Rate"] * total_re, name=f"CurtRate_{s}")
        # Scale utilization hours by time horizon ratio
        min_util_hours = phys["Min_Util_Hours"] * (T / 8760.0)
        m.addConstr(total_re - gp.quicksum(p_curt[s, t] for t in range(T)) >= min_util_hours * (phys["Cap_Wind"] + phys["Cap_PV"]), name=f"UtilHours_{s}")
        # RPS constraint removed: Cap_Therm=10GW is always below 70% of consumption

    # --- NEW: Carbon Emission Cap (Q1 journal requirement) ---
    carbon_annual = gp.quicksum(
        weights[s] * gp.quicksum(econ["EF_coal"] * p_therm[s, t] for t in range(T))
        for s in range(S)
    )
    # Annual carbon cap scaled proportionally to representative-day horizon
    # This is the SINGLE scaling point for annual constraints — no preprocessing scaling applied
    scaled_carbon_cap = econ["Carbon_cap_annual"] * (T / 8760.0)
    m.addConstr(carbon_annual <= scaled_carbon_cap, name="Carbon_Cap")

    # ==========================================================================
    # OBJECTIVE FUNCTION
    # ==========================================================================
    crf_bess = _crf(econ["r_discount"], econ["N_bess"])
    crf_elc = _crf(econ["r_discount"], econ["N_elc"])
    crf_h2 = _crf(econ["r_discount"], econ["N_h2_tank"])
    crf_fc = _crf(econ["r_discount"], econ["N_fc"])

    cost_inv = (x_bess_p * econ["C_inv_bess_p"] + x_bess_e * econ["C_inv_bess_e"]) * crf_bess \
               + x_elc_p * econ["C_inv_elc"] * crf_elc \
               + x_h2_tank * econ["C_inv_h2_tank"] * crf_h2 \
               + x_fc_p * econ["C_inv_fc"] * crf_fc

    inv_total_bess = x_bess_p * econ["C_inv_bess_p"] + x_bess_e * econ["C_inv_bess_e"]
    inv_total_elc = x_elc_p * econ["C_inv_elc"]
    inv_total_h2 = x_h2_tank * econ["C_inv_h2_tank"]
    inv_total_fc = x_fc_p * econ["C_inv_fc"]

    cost_om_fix = (inv_total_bess * econ["rate_om_bess"] + inv_total_elc * econ["rate_om_elc"]
                   + inv_total_h2 * econ["rate_om_h2_tank"] + inv_total_fc * econ["rate_om_fc"])

    # Expected operating cost across scenarios
    cost_op_exp = gp.quicksum(
        weights[s] * (
            gp.quicksum(p_therm[s, t] * econ["C_coal"] for t in range(T))
            + gp.quicksum(y_start[s, t] * econ["C_start"] + z_stop[s, t] * econ["C_stop"] for t in range(T))
            + gp.quicksum(y_elc_start[s, t] * econ["C_start_elc"] + z_elc_stop[s, t] * econ["C_stop_elc"] for t in range(T))
            + gp.quicksum((p_bess_ch[s, t] + p_bess_dis[s, t]) * econ["var_om_bess"] for t in range(T))
            + gp.quicksum(p_elc[s, t] * econ["var_om_elc"] for t in range(T))
            + gp.quicksum(p_fc[s, t] * econ["var_om_fc"] for t in range(T))
            + gp.quicksum(p_curt[s, t] * econ["C_pen_curt"] for t in range(T))
            + gp.quicksum(p_therm[s, t] * econ["EF_coal"] * econ["Carbon_price"] for t in range(T))  # Carbon cost
        )
        for s in range(S)
    )

    # Expected revenue
    revenue_exp = gp.quicksum(
        weights[s] * (
            gp.quicksum(p_uhv[s, t] * econ["Price_grid"] for t in range(T))
            + gp.quicksum(h_supply[s, t] * (econ["Price_h2"] + econ["Price_h2_green_premium"]) for t in range(T))
        )
        for s in range(S)
    )

    # CRITICAL FIX: Annualize operating costs for representative-day model.
    # Investment cost (via CRF) and fixed O&M are already annualized.
    # Operating cost and revenue are computed over T hours (e.g., 480h for 20 rep days).
    # They must be scaled to a full year (8760h) to be comparable with annualized investment.
    annualization = 8760.0 / T
    m.setObjective(cost_inv + cost_om_fix + (cost_op_exp - revenue_exp) * annualization, GRB.MINIMIZE)

    # Collect variables for extraction
    var_dict = {
        "x_bess_p": x_bess_p, "x_bess_e": x_bess_e,
        "x_elc_p": x_elc_p, "x_h2_tank": x_h2_tank, "x_fc_p": x_fc_p,
        "p_therm": p_therm, "p_bess_ch": p_bess_ch, "p_bess_dis": p_bess_dis,
        "e_bess": e_bess, "p_elc": p_elc, "h_prod": h_prod,
        "h_tank": h_tank, "h_supply": h_supply, "h_fc_use": h_fc_use,
        "p_fc": p_fc, "p_uhv": p_uhv, "p_curt": p_curt,
        "u_therm": u_therm, "u_elc": u_elc,
        "y_start": y_start, "z_stop": z_stop,
        "y_elc_start": y_elc_start, "z_elc_stop": z_elc_stop,
        "u_bess_ch": u_bess_ch, "u_fc": u_fc,
        "x_elc_on": x_elc_on,
        "cost_inv": cost_inv, "cost_om_fix": cost_om_fix,
        "cost_op_exp": cost_op_exp, "revenue_exp": revenue_exp,
        "carbon_annual": carbon_annual,
    }

    return m, var_dict


def solve_and_extract(m, var_dict, load_profile, wind_sc, solar_sc, weights, phys):
    """Solve model and extract results into structured dict."""
    m.optimize()

    status = m.status
    if status in [GRB.INFEASIBLE, GRB.INF_OR_UNBD]:
        print("Model infeasible. Computing IIS...")
        m.computeIIS()
        os.makedirs("results", exist_ok=True)
        m.write("results/conflict_tssp.ilp")
        return None, status

    # Protect against TIME_LIMIT with zero feasible solutions
    if m.SolCount == 0:
        print(f"Model status {status} but no feasible solution found (SolCount=0).")
        return None, status

    S, T = len(weights), phys["T"]
    # CRITICAL: cost_op_exp and revenue_exp expressions are raw T-hour expected values.
    # For consistency with annualized objective, scale them when storing.
    annualization = 8760.0 / T
    res = {
        "status": status,
        "objval": m.ObjVal,
        "mipgap": m.MIPGap,
        "runtime": m.Runtime,
        "capacity": {
            "BESS_P_MW": var_dict["x_bess_p"].X,
            "BESS_E_MWh": var_dict["x_bess_e"].X,
            "ELC_P_MW": var_dict["x_elc_p"].X,
            "H2_Tank_kg": var_dict["x_h2_tank"].X,
            "FC_P_MW": var_dict["x_fc_p"].X,
        },
        "costs": {
            "inv": var_dict["cost_inv"].getValue(),
            "om_fix": var_dict["cost_om_fix"].getValue(),
            "op_exp": var_dict["cost_op_exp"].getValue() * annualization,
            "revenue": var_dict["revenue_exp"].getValue() * annualization,
            # carbon_annual is expected T-hour emission; annualize for consistency
            "carbon_annual": var_dict["carbon_annual"].getValue() * annualization,
        },
        "scenarios": []
    }

    for s in range(S):
        df_s = {
            "hour": np.arange(T),
            "load": load_profile,
            "wind_avail": phys["Cap_Wind"] * wind_sc[s],
            "pv_avail": phys["Cap_PV"] * solar_sc[s],
            "therm": [var_dict["p_therm"][s, t].X for t in range(T)],
            "bess_ch": [var_dict["p_bess_ch"][s, t].X for t in range(T)],
            "bess_dis": [var_dict["p_bess_dis"][s, t].X for t in range(T)],
            "bess_e": [var_dict["e_bess"][s, t].X for t in range(T)],
            "elc": [var_dict["p_elc"][s, t].X for t in range(T)],
            "h_prod": [var_dict["h_prod"][s, t].X for t in range(T)],
            "h_tank": [var_dict["h_tank"][s, t].X for t in range(T)],
            "h_supply": [var_dict["h_supply"][s, t].X for t in range(T)],
            "h_fc_use": [var_dict["h_fc_use"][s, t].X for t in range(T)],
            "fc": [var_dict["p_fc"][s, t].X for t in range(T)],
            "uhv": [var_dict["p_uhv"][s, t].X for t in range(T)],
            "curt": [var_dict["p_curt"][s, t].X for t in range(T)],
            "u_therm": [var_dict["u_therm"][s, t].X for t in range(T)],
            "u_elc": [var_dict["u_elc"][s, t].X for t in range(T)],
            "weight": weights[s],
        }
        res["scenarios"].append(df_s)

    return res, status


# ==============================================================================
# Helpers
# ==============================================================================
def _crf(r, n):
    return (r * (1 + r) ** n) / ((1 + r) ** n - 1) if r > 1e-12 else 1.0 / n


def _build_uhv_bands(T, p_max):
    p_min, p_max_list = [], []
    for t in range(T):
        h = t % 24
        if 10 <= h <= 16:
            mn, mx = 0.55, 1.00
        elif h >= 20 or h <= 5:
            mn, mx = 0.45, 0.85
        else:
            mn, mx = 0.25, 0.70
        p_min.append(mn * p_max)
        p_max_list.append(mx * p_max)
    return np.array(p_min), np.array(p_max_list)

def build_eev_model(load_profile,
                    wind_scenarios, solar_scenarios, weights,
                    fixed_capacity, econ, phys, solver_cfg):
    """
    Build EEV (Expected Eval of EV solution) model by fixing 1st-stage
    capacity decisions to the EV solution, then re-optimizing 2nd-stage
    operational variables under uncertainty.

    This function REUSES build_two_stage_model() to ensure 100% constraint
    consistency between TSSP and EEV. After building, it fixes the capacity
    variables by setting LB=UB=fixed_value, effectively removing them from
    the optimization while preserving all original constraints.

    Parameters:
        load_profile: (T,)
        wind_scenarios: (S, T)
        solar_scenarios: (S, T)
        weights: (S,) probabilities
        fixed_capacity: dict with keys "BESS_P_MW", "BESS_E_MWh",
                        "ELC_P_MW", "H2_Tank_kg", "FC_P_MW"
        econ, phys: dicts from config
        solver_cfg: dict

    Returns:
        m: Gurobi model (capacities fixed, operation free)
        var_dict: variable dictionary (same interface as build_two_stage_model)
    """
    # Step 1: Build the EXACT same model as TSSP
    m, var_dict = build_two_stage_model(
        load_profile, wind_scenarios, solar_scenarios, weights,
        econ, phys, solver_cfg
    )

    # Step 2: Fix 1st-stage capacity variables to EV solution
    # Setting LB=UB removes them from optimization while keeping constraints intact
    var_dict["x_bess_p"].LB = fixed_capacity["BESS_P_MW"]
    var_dict["x_bess_p"].UB = fixed_capacity["BESS_P_MW"]

    var_dict["x_bess_e"].LB = fixed_capacity["BESS_E_MWh"]
    var_dict["x_bess_e"].UB = fixed_capacity["BESS_E_MWh"]

    var_dict["x_elc_p"].LB = fixed_capacity["ELC_P_MW"]
    var_dict["x_elc_p"].UB = fixed_capacity["ELC_P_MW"]

    var_dict["x_h2_tank"].LB = fixed_capacity["H2_Tank_kg"]
    var_dict["x_h2_tank"].UB = fixed_capacity["H2_Tank_kg"]

    var_dict["x_fc_p"].LB = fixed_capacity["FC_P_MW"]
    var_dict["x_fc_p"].UB = fixed_capacity["FC_P_MW"]

    # Rename model for clarity in solver logs
    m.setAttr("ModelName", "EEV_FixedCap_RE_H2_Storage_UHV")
    m.update()

    return m, var_dict
