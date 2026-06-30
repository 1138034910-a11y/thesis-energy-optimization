"""
================================================================================
Deterministic Benchmark Model (Expected-Value Problem)
================================================================================
Uses KAN predicted EXPECTED values (mu) as single-point forecasts.
Serves as benchmark to quantify the Value of Stochastic Solution (VSS).
================================================================================
"""

import numpy as np
import gurobipy as gp
from gurobipy import GRB
# Helpers (duplicated here to avoid relative import issues)
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


def build_deterministic_model(load_profile, mu_wind, mu_solar, econ, phys, solver_cfg):
    """Deterministic MILP using expected renewable profiles."""
    T = phys["T"]
    m = gp.Model("Deterministic_RE_H2_UHV")
    m.setParam("OutputFlag", solver_cfg.get("OutputFlag", 1))
    m.setParam("MIPGap", solver_cfg.get("MIPGap", 0.05))
    m.setParam("TimeLimit", solver_cfg.get("TimeLimit", 3600))
    m.setParam("Threads", solver_cfg.get("Threads", 0))
    m.setParam("MIPFocus", solver_cfg.get("MIPFocus", 1))
    m.setParam("Heuristics", solver_cfg.get("Heuristics", 0.3))
    m.setParam("Presolve", solver_cfg.get("Presolve", 2))
    m.setParam("Cuts", solver_cfg.get("Cuts", 2))
    m.setParam("Crossover", solver_cfg.get("Crossover", 0))
    if "ImproveStartGap" in solver_cfg:
        m.setParam("ImproveStartGap", solver_cfg["ImproveStartGap"])

    # --- First-stage (same as TSSP) ---
    x_bess_p = m.addVar(lb=0, ub=phys["Cap_BESS_P_Max"], name="Cap_BESS_P")
    x_bess_e = m.addVar(lb=0, ub=phys["Cap_BESS_E_Max"], name="Cap_BESS_E")
    x_elc_p = m.addVar(lb=0, ub=phys["Cap_ELC_P_Max"], name="Cap_ELC_P")
    x_h2_tank = m.addVar(lb=0, ub=phys["Cap_H2_Tank_Max"], name="Cap_H2_Tank")
    x_fc_p = m.addVar(lb=0, ub=phys["Cap_FC_P_Max"], name="Cap_FC_P")

    # --- Operational variables (single scenario = expected value) ---
    p_therm = m.addVars(T, lb=0, name="P_Therm")
    p_bess_ch = m.addVars(T, lb=0, name="P_BESS_Ch")
    p_bess_dis = m.addVars(T, lb=0, name="P_BESS_Dis")
    e_bess = m.addVars(T, lb=0, name="E_BESS")
    p_elc = m.addVars(T, lb=0, name="P_ELC")
    h_prod = m.addVars(T, lb=0, name="H2_Prod")
    h_tank = m.addVars(T, lb=0, name="H2_Tank")
    h_supply = m.addVars(T, lb=0, name="H2_Supply")
    h_fc_use = m.addVars(T, lb=0, name="H2_Use_FC")
    p_fc = m.addVars(T, lb=0, name="P_FC")
    p_uhv = m.addVars(T, lb=0, ub=phys["P_UHV_Max"], name="P_UHV")
    p_curt = m.addVars(T, lb=0, name="P_Curt")

    u_therm = m.addVars(T, vtype=GRB.BINARY, name="u_Therm")
    y_start = m.addVars(T, vtype=GRB.BINARY, name="y_Start")
    z_stop = m.addVars(T, vtype=GRB.BINARY, name="z_Stop")
    u_elc = m.addVars(T, vtype=GRB.BINARY, name="u_ELC")
    y_elc_start = m.addVars(T, vtype=GRB.BINARY, name="y_ELC_Start")
    z_elc_stop = m.addVars(T, vtype=GRB.BINARY, name="z_ELC_Stop")
    u_bess_ch = m.addVars(T, vtype=GRB.BINARY, name="u_BESS_Ch")
    u_fc = m.addVars(T, vtype=GRB.BINARY, name="u_FC")

    x_elc_on = m.addVars(T, lb=0, ub=phys["Cap_ELC_P_Max"], name="Cap_ELC_On")

    m.update()

    # --- BESS ---
    m.addConstr(e_bess[0] == phys["SOC_Init"] * x_bess_e + p_bess_ch[0] * phys["Eta_BESS_Ch"] - p_bess_dis[0] / phys["Eta_BESS_Dis"])
    for t in range(1, T):
        m.addConstr(e_bess[t] == e_bess[t - 1] * (1 - phys["Self_Discharge"]) + p_bess_ch[t] * phys["Eta_BESS_Ch"] - p_bess_dis[t] / phys["Eta_BESS_Dis"])
    m.addConstr(e_bess[T - 1] == phys["SOC_Init"] * x_bess_e)
    m.addConstr(x_bess_e >= phys["BESS_Min_Duration"] * x_bess_p)
    for t in range(T):
        m.addConstr(e_bess[t] <= phys["SOC_Max"] * x_bess_e)
        m.addConstr(e_bess[t] >= phys["SOC_Min"] * x_bess_e)
        m.addConstr(p_bess_ch[t] <= x_bess_p)
        m.addConstr(p_bess_dis[t] <= x_bess_p)
        m.addConstr(p_bess_ch[t] <= phys["Cap_BESS_P_Max"] * u_bess_ch[t])
        m.addConstr(p_bess_dis[t] <= phys["Cap_BESS_P_Max"] * (1 - u_bess_ch[t]))

    # --- Thermal ---
    p_th_min = phys["Cap_Therm"] * phys["Rate_Therm_Min"]
    p_th_max = phys["Cap_Therm"]
    ramp_th = phys["Cap_Therm"] * phys["Rate_Ramp_Therm"]
    t_on, t_off = phys["Min_Up_Time"], phys["Min_Down_Time"]
    m.addConstr(y_start[0] - z_stop[0] == u_therm[0])
    for t in range(1, T):
        m.addConstr(y_start[t] - z_stop[t] == u_therm[t] - u_therm[t - 1])
    for t in range(T):
        m.addConstr(p_therm[t] >= p_th_min * u_therm[t])
        m.addConstr(p_therm[t] <= p_th_max * u_therm[t])
    for t in range(1, T):
        m.addConstr(p_therm[t] - p_therm[t - 1] <= ramp_th * u_therm[t - 1] + p_th_min * y_start[t])
        m.addConstr(p_therm[t - 1] - p_therm[t] <= ramp_th * u_therm[t] + p_th_min * z_stop[t])
    for t in range(T):
        end_t = min(t + t_on, T)
        m.addConstr(gp.quicksum(u_therm[k] for k in range(t, end_t)) >= (end_t - t) * y_start[t])
        end_t2 = min(t + t_off, T)
        m.addConstr(gp.quicksum(1 - u_therm[k] for k in range(t, end_t2)) >= (end_t2 - t) * z_stop[t])

    # --- Electrolyzer with Fixed Average Efficiency ---
    # FIX: ramp must scale with installed capacity x_elc_p (consistent with TSSP)
    elc_ramp = phys["Rate_Ramp_ELC"] * x_elc_p
    eta_h2 = phys["ELC_Efficiency"]
    m.addConstr(y_elc_start[0] - z_elc_stop[0] == u_elc[0])
    for t in range(1, T):
        m.addConstr(y_elc_start[t] - z_elc_stop[t] == u_elc[t] - u_elc[t - 1])
    for t in range(T):
        m.addConstr(x_elc_on[t] <= x_elc_p)
        m.addConstr(x_elc_on[t] <= phys["Cap_ELC_P_Max"] * u_elc[t])
        m.addConstr(x_elc_on[t] >= x_elc_p - phys["Cap_ELC_P_Max"] * (1 - u_elc[t]))
        m.addConstr(p_elc[t] <= x_elc_on[t])
        m.addConstr(p_elc[t] >= phys["ELC_Min_Load_Ratio"] * x_elc_on[t])
    for t in range(1, T):
        m.addConstr(p_elc[t] - p_elc[t - 1] <= elc_ramp)
        m.addConstr(p_elc[t - 1] - p_elc[t] <= elc_ramp)
    for t in range(T):
        m.addConstr(h_prod[t] == eta_h2 * p_elc[t])

    # --- H2 Storage ---
    m.addConstr(h_tank[0] == phys["H2_Tank_Init_Ratio"] * x_h2_tank * (1 - phys["H2_Tank_Loss"]) + h_prod[0] - h_supply[0] - h_fc_use[0])
    for t in range(1, T):
        m.addConstr(h_tank[t] == h_tank[t - 1] * (1 - phys["H2_Tank_Loss"]) + h_prod[t] - h_supply[t] - h_fc_use[t])
    m.addConstr(h_tank[T - 1] == phys["H2_Tank_Init_Ratio"] * x_h2_tank)
    for t in range(T):
        m.addConstr(h_tank[t] <= phys["H2_Tank_Max_Ratio"] * x_h2_tank)
        m.addConstr(h_tank[t] >= phys["H2_Tank_Min_Ratio"] * x_h2_tank)
        m.addConstr(h_supply[t] >= phys["H2_Demand_Min"])
        m.addConstr(h_supply[t] <= phys["H2_Demand_Max"])
    for t in range(1, T):
        m.addConstr(h_supply[t] - h_supply[t - 1] <= phys["H2_Delivery_Ramp"])
        m.addConstr(h_supply[t - 1] - h_supply[t] <= phys["H2_Delivery_Ramp"])

    # --- FC ---
    for t in range(T):
        m.addConstr(p_fc[t] <= x_fc_p)
        m.addConstr(p_fc[t] >= phys["FC_Min_Load_Ratio"] * x_fc_p - phys["FC_Min_Load_Ratio"] * phys["Cap_FC_P_Max"] * (1 - u_fc[t]))
        m.addConstr(p_fc[t] <= phys["Cap_FC_P_Max"] * u_fc[t])
        m.addConstr(p_fc[t] == h_fc_use[t] * phys["Eta_FC_Power"])

    # --- UHV ---
    p_uhv_min_band, p_uhv_max_band = _build_uhv_bands(T, phys["P_UHV_Max"])
    ramp_uhv = phys["P_UHV_Max"] * phys["Rate_Ramp_UHV"]
    for t in range(T):
        m.addConstr(p_uhv[t] >= p_uhv_min_band[t])
        m.addConstr(p_uhv[t] <= p_uhv_max_band[t])
    for t in range(1, T):
        m.addConstr(p_uhv[t] - p_uhv[t - 1] <= ramp_uhv)
        m.addConstr(p_uhv[t - 1] - p_uhv[t] <= ramp_uhv)
    min_uhv_energy = phys["Min_UHV_Energy"] * (T / 8760.0)
    m.addConstr(gp.quicksum(p_uhv[t] for t in range(T)) >= min_uhv_energy)

    # --- Power Balance ---
    p_wind_avail = phys["Cap_Wind"] * mu_wind
    p_pv_avail = phys["Cap_PV"] * mu_solar
    for t in range(T):
        m.addConstr(p_wind_avail[t] + p_pv_avail[t] + p_therm[t] + p_bess_dis[t] + p_fc[t]
                    == load_profile[t] + p_uhv[t] + p_bess_ch[t] + p_elc[t] + p_curt[t])

    # --- Renewable constraints ---
    total_re = float(np.sum(p_wind_avail + p_pv_avail))
    m.addConstr(gp.quicksum(p_curt[t] for t in range(T)) <= phys["Limit_Curt_Rate"] * total_re)
    min_util_hours = phys["Min_Util_Hours"] * (T / 8760.0)
    m.addConstr(total_re - gp.quicksum(p_curt[t] for t in range(T)) >= min_util_hours * (phys["Cap_Wind"] + phys["Cap_PV"]))
    # RPS constraint removed: Cap_Therm=10GW is always below 70% of consumption

    # --- Carbon cap ---
    carbon = gp.quicksum(econ["EF_coal"] * p_therm[t] for t in range(T))
    # Annual carbon cap scaled by T/8760 (single scaling point, see stochastic_model.py)
    scaled_carbon_cap = econ["Carbon_cap_annual"] * (T / 8760.0)
    m.addConstr(carbon <= scaled_carbon_cap)

    # --- Objective ---
    crf_bess = _crf(econ["r_discount"], econ["N_bess"])
    crf_elc = _crf(econ["r_discount"], econ["N_elc"])
    crf_h2 = _crf(econ["r_discount"], econ["N_h2_tank"])
    crf_fc = _crf(econ["r_discount"], econ["N_fc"])

    cost_inv = (x_bess_p * econ["C_inv_bess_p"] + x_bess_e * econ["C_inv_bess_e"]) * crf_bess \
               + x_elc_p * econ["C_inv_elc"] * crf_elc + x_h2_tank * econ["C_inv_h2_tank"] * crf_h2 + x_fc_p * econ["C_inv_fc"] * crf_fc
    inv_tot = x_bess_p * econ["C_inv_bess_p"] + x_bess_e * econ["C_inv_bess_e"]
    inv_elc = x_elc_p * econ["C_inv_elc"]
    inv_h2 = x_h2_tank * econ["C_inv_h2_tank"]
    inv_fc = x_fc_p * econ["C_inv_fc"]
    cost_om = inv_tot * econ["rate_om_bess"] + inv_elc * econ["rate_om_elc"] + inv_h2 * econ["rate_om_h2_tank"] + inv_fc * econ["rate_om_fc"]

    cost_op = gp.quicksum(p_therm[t] * econ["C_coal"] for t in range(T)) \
              + gp.quicksum(y_start[t] * econ["C_start"] + z_stop[t] * econ["C_stop"] for t in range(T)) \
              + gp.quicksum(y_elc_start[t] * econ["C_start_elc"] + z_elc_stop[t] * econ["C_stop_elc"] for t in range(T)) \
              + gp.quicksum((p_bess_ch[t] + p_bess_dis[t]) * econ["var_om_bess"] for t in range(T)) \
              + gp.quicksum(p_elc[t] * econ["var_om_elc"] + p_fc[t] * econ["var_om_fc"] + p_curt[t] * econ["C_pen_curt"] for t in range(T)) \
              + gp.quicksum(p_therm[t] * econ["EF_coal"] * econ["Carbon_price"] for t in range(T))

    revenue = gp.quicksum(p_uhv[t] * econ["Price_grid"] + h_supply[t] * (econ["Price_h2"] + econ["Price_h2_green_premium"]) for t in range(T))

    # CRITICAL FIX: Annualize operating costs for representative-day model.
    # Investment cost (via CRF) and fixed O&M are already annualized.
    # Operating cost and revenue are computed over T hours (e.g., 480h for 20 rep days).
    # They must be scaled to a full year (8760h) to be comparable with annualized investment.
    annualization = 8760.0 / T
    m.setObjective(cost_inv + cost_om + (cost_op - revenue) * annualization, GRB.MINIMIZE)

    m.optimize()

    if m.status not in [GRB.OPTIMAL, GRB.TIME_LIMIT] or m.SolCount == 0:
        print(f"  [WARN] Deterministic model: status={m.status}, SolCount={m.SolCount}")
        return None, m.status, None

    # CRITICAL: cost_op and revenue expressions are raw T-hour values.
    # For consistency with annualized objective, scale them when storing.
    annualization = 8760.0 / T
    res = {
        "status": m.status,
        "objval": m.ObjVal,
        "mipgap": m.MIPGap,
        "capacity": {
            "BESS_P_MW": x_bess_p.X, "BESS_E_MWh": x_bess_e.X,
            "ELC_P_MW": x_elc_p.X, "H2_Tank_kg": x_h2_tank.X, "FC_P_MW": x_fc_p.X,
        },
        "costs": {"inv": cost_inv.getValue(), "om": cost_om.getValue(),
                  "op": cost_op.getValue() * annualization,
                  "revenue": revenue.getValue() * annualization,
                  # carbon is T-hour emission; annualize for consistency with annual cap
                  "carbon": carbon.getValue() * annualization},
    }

    # Extract operational variable values for MIPStart in EEV/TSSP models
    op_vars = {
        "p_therm":    np.array([p_therm[t].X    for t in range(T)]),
        "p_bess_ch":  np.array([p_bess_ch[t].X  for t in range(T)]),
        "p_bess_dis": np.array([p_bess_dis[t].X for t in range(T)]),
        "e_bess":     np.array([e_bess[t].X     for t in range(T)]),
        "p_elc":      np.array([p_elc[t].X      for t in range(T)]),
        "h_prod":     np.array([h_prod[t].X     for t in range(T)]),
        "h_tank":     np.array([h_tank[t].X     for t in range(T)]),
        "h_supply":   np.array([h_supply[t].X   for t in range(T)]),
        "h_fc_use":   np.array([h_fc_use[t].X   for t in range(T)]),
        "p_fc":       np.array([p_fc[t].X       for t in range(T)]),
        "p_uhv":      np.array([p_uhv[t].X      for t in range(T)]),
        "p_curt":     np.array([p_curt[t].X     for t in range(T)]),
        "u_therm":    np.array([u_therm[t].X    for t in range(T)]),
        "y_start":    np.array([y_start[t].X    for t in range(T)]),
        "z_stop":     np.array([z_stop[t].X     for t in range(T)]),
        "u_elc":      np.array([u_elc[t].X      for t in range(T)]),
        "y_elc_start":np.array([y_elc_start[t].X for t in range(T)]),
        "z_elc_stop": np.array([z_elc_stop[t].X for t in range(T)]),
        "u_bess_ch":  np.array([u_bess_ch[t].X  for t in range(T)]),
        "u_fc":       np.array([u_fc[t].X       for t in range(T)]),
        "x_elc_on":   np.array([x_elc_on[t].X   for t in range(T)]),
    }

    return res, m.status, op_vars
