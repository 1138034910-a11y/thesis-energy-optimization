"""
================================================================================
Sensitivity Analysis Module
================================================================================
Implements:
  1. Carbon cap sweep: Pareto frontier of cost vs carbon emission
  2. Tornado diagram: one-at-a-time parameter sensitivity
  3. Typical day dispatch analysis: seasonal representative days

Reference (diagnostic report requirements):
- Sec 3.4: Carbon emission constraint sensitivity
- Sec 3.6: Typical day dispatch analysis
- Sec 3.7: Economic parameter sensitivity (tornado diagram)
================================================================================"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ==============================================================================
# 1. Carbon Cap Sensitivity Sweep
# ==============================================================================

def run_carbon_sweep(build_model_fn, load_profile, wind_sc, solar_sc, weights,
                     econ, phys, solver_cfg, carbon_caps,
                     results_dir="results/sensitivity"):
    """
    Run deterministic or TSSP model under varying carbon caps.
    Returns DataFrame with columns: carbon_cap, objval, capacity..., solve_time

    Args:
        build_model_fn: callable that returns (model, result_dict)
        carbon_caps: list of carbon cap values (tons/year)
    """
    os.makedirs(results_dir, exist_ok=True)
    records = []

    for cap in carbon_caps:
        print(f"\n[Carbon Sweep] Cap = {cap:,.0f} ton/year")
        econ_mod = dict(econ)
        econ_mod["Carbon_cap_annual"] = float(cap)

        try:
            # Handle both deterministic (no weights/scenarios) and TSSP signatures
            import inspect
            sig = inspect.signature(build_model_fn)
            params = list(sig.parameters.keys())
            if 'weights' in params:
                res, status = build_model_fn(load_profile, wind_sc, solar_sc, weights,
                                             econ_mod, phys, solver_cfg)
            else:
                # Deterministic model: only uses load_profile, mu_wind, mu_solar
                res, status = build_model_fn(load_profile, wind_sc[0] if len(wind_sc.shape)>1 else wind_sc,
                                             solar_sc[0] if len(solar_sc.shape)>1 else solar_sc,
                                             econ_mod, phys, solver_cfg)
            if res:
                rec = {
                    "carbon_cap_ton": cap,
                    "objval_10kCNY": res["objval"],
                    "bess_p_MW": res["capacity"]["BESS_P_MW"],
                    "bess_e_MWh": res["capacity"]["BESS_E_MWh"],
                    "elc_p_MW": res["capacity"]["ELC_P_MW"],
                    "h2_tank_kg": res["capacity"]["H2_Tank_kg"],
                    "fc_p_MW": res["capacity"]["FC_P_MW"],
                    "mipgap": res.get("mipgap", np.nan),
                    "runtime_s": res.get("runtime", np.nan),
                }
                # Compute actual carbon emission if available
                T = phys.get("T", 8760)
                annualization = 8760.0 / T if T > 0 else 1.0
                if "scenarios" in res and res["scenarios"]:
                    total_carbon = 0
                    for sc in res["scenarios"]:
                        w = sc.get("weight", 1.0 / len(res["scenarios"]))
                        total_carbon += w * np.sum(sc.get("therm", [])) * econ_mod["EF_coal"]
                    rec["actual_carbon_kt"] = total_carbon * annualization / 1000
                elif "costs" in res and "carbon" in res["costs"]:
                    # Deterministic model: carbon already annualized in result extraction
                    rec["actual_carbon_kt"] = res["costs"]["carbon"] / 1000
                else:
                    rec["actual_carbon_kt"] = np.nan
                records.append(rec)
            else:
                records.append({"carbon_cap_ton": cap, "status": "infeasible"})
        except Exception as e:
            print(f"  ERROR at cap={cap}: {e}")
            records.append({"carbon_cap_ton": cap, "status": f"error: {e}"})

    df = pd.DataFrame(records)
    df.to_csv(f"{results_dir}/carbon_sweep.csv", index=False)
    print(f"\n[Carbon Sweep] Results saved to {results_dir}/carbon_sweep.csv")
    return df


def plot_carbon_pareto(df, fig_dir="results/figures archive"):
    """
    Plot Pareto frontier: System Cost vs Carbon Cap / Actual Emission.
    This is a core contribution figure for the paper.
    """
    os.makedirs(fig_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(7.48, 3.0), dpi=300)

    # Plot 1: Cost vs Carbon Cap
    ax = axes[0]
    ax.plot(df["carbon_cap_ton"] / 1e6, df["objval_10kCNY"] / 1e4,
            marker='o', markersize=4, linewidth=1.2, color="#4C78A8")
    ax.set_xlabel("Carbon Cap (MtCO₂/year)", fontsize=8)
    ax.set_ylabel("System Cost (billion CNY/year)", fontsize=8)
    ax.set_title("Cost vs Carbon Cap", fontsize=9, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=7)

    # Plot 2: Capacity mix vs Carbon Cap
    ax = axes[1]
    cap_cols = ["bess_p_MW", "elc_p_MW", "fc_p_MW"]
    cap_labels = ["BESS", "Electrolyzer", "Fuel Cell"]
    colors = ["#72B7B2", "#54A24B", "#B279A2"]
    x = df["carbon_cap_ton"] / 1e6
    bottom = np.zeros(len(df))
    for col, label, color in zip(cap_cols, cap_labels, colors):
        ax.bar(x, df[col], bottom=bottom, width=0.15, label=label, color=color, edgecolor='white', linewidth=0.3)
        bottom += df[col].fillna(0).values
    ax.set_xlabel("Carbon Cap (MtCO₂/year)", fontsize=8)
    ax.set_ylabel("Capacity (MW)", fontsize=8)
    ax.set_title("Optimal Capacity Mix", fontsize=9, fontweight='bold')
    ax.legend(fontsize=7, frameon=False)
    ax.tick_params(labelsize=7)

    plt.tight_layout()
    fig.savefig(f"{fig_dir}/sensitivity_carbon_cap.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Saved {fig_dir}/sensitivity_carbon_cap.png")


# ==============================================================================
# 2. Tornado Diagram (One-at-a-Time Sensitivity)
# ==============================================================================

def run_tornado_sensitivity(base_model_fn, load_profile, wind_sc, solar_sc, weights,
                            econ, phys, solver_cfg,
                            param_specs, results_dir="results/sensitivity"):
    """
    One-at-a-time sensitivity analysis.

    Args:
        param_specs: dict of {param_name: (low_factor, high_factor)}
                     e.g., {"Price_h2": (0.5, 2.0), "C_inv_bess_p": (0.7, 1.3)}
    Returns:
        DataFrame with sensitivity results
    """
    os.makedirs(results_dir, exist_ok=True)
    base_res, _ = base_model_fn(load_profile, wind_sc, solar_sc, weights, econ, phys, solver_cfg)
    base_obj = base_res["objval"] if base_res else np.nan

    records = []
    for param_name, (low_f, high_f) in param_specs.items():
        for factor, label in [(low_f, "low"), (high_f, "high")]:
            econ_mod = dict(econ)
            if param_name in econ_mod:
                econ_mod[param_name] = econ_mod[param_name] * factor
            else:
                continue
            try:
                res, _ = base_model_fn(load_profile, wind_sc, solar_sc, weights,
                                       econ_mod, phys, solver_cfg)
                obj = res["objval"] if res else np.nan
                records.append({
                    "parameter": param_name,
                    "level": label,
                    "factor": factor,
                    "objval": obj,
                    "delta_pct": 100 * (obj - base_obj) / base_obj if base_obj else np.nan,
                })
            except Exception as e:
                print(f"  Tornado error for {param_name}={label}: {e}")
                records.append({"parameter": param_name, "level": label, "factor": factor, "objval": np.nan})

    df = pd.DataFrame(records)
    df.to_csv(f"{results_dir}/tornado.csv", index=False)
    return df


def plot_tornado(df, fig_dir="results/figures archive"):
    """Plot tornado diagram: parameter impact on objective value."""
    os.makedirs(fig_dir, exist_ok=True)

    # Pivot to get low/high delta for each parameter
    pivoted = df.pivot(index="parameter", columns="level", values="delta_pct")
    if "low" not in pivoted.columns or "high" not in pivoted.columns:
        print("[Tornado] Insufficient data for plotting.")
        return

    pivoted = pivoted.sort_values("high", key=abs, ascending=True)
    y_pos = np.arange(len(pivoted))

    fig, ax = plt.subplots(figsize=(4.5, 3.5), dpi=300)
    ax.barh(y_pos, pivoted["high"].fillna(0), height=0.5, color="#E45756", label="High", alpha=0.85)
    ax.barh(y_pos, pivoted["low"].fillna(0), height=0.5, color="#4C78A8", label="Low", alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(pivoted.index, fontsize=7)
    ax.set_xlabel("Change in Objective Value (%)", fontsize=8)
    ax.set_title("Tornado Sensitivity Analysis", fontsize=9, fontweight='bold')
    ax.axvline(0, color='black', linewidth=0.5)
    ax.legend(fontsize=7, frameon=False)
    ax.tick_params(labelsize=7)
    ax.grid(True, axis='x', alpha=0.3)

    plt.tight_layout()
    fig.savefig(f"{fig_dir}/sensitivity_tornado.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Saved {fig_dir}/sensitivity_tornado.png")


# ==============================================================================
# 3. Typical Day Dispatch Analysis
# ==============================================================================

def extract_typical_day(tssp_res, season="spring", season_days=None):
    """
    Extract a representative 24-hour dispatch profile from TSSP results.

    Args:
        tssp_res: TSSP result dict with scenario hourly data
        season: "spring", "summer", "autumn", "winter"
        season_days: dict mapping season to (start_day, end_day) 1-based
    """
    if season_days is None:
        season_days = {
            "spring": (60, 151),   # Mar-Apr-May (approx)
            "summer": (152, 243),  # Jun-Jul-Aug
            "autumn": (244, 334),  # Sep-Oct-Nov
            "winter": (335, 59),   # Dec-Jan-Feb (wrap around)
        }

    start, end = season_days[season]
    # Pick middle day
    if start < end:
        mid_day = (start + end) // 2
    else:
        mid_day = ((start + 365 + end) // 2) % 365
    hour_start = mid_day * 24
    hour_end = hour_start + 24

    # Aggregate across scenarios using weights
    scenarios = tssp_res.get("scenarios", [])
    if not scenarios:
        return None

    weights = np.array([s["weight"] for s in scenarios])
    weights = weights / weights.sum()

    profiles = {}
    for key in ["wind_avail", "pv_avail", "therm", "p_bess_ch", "p_bess_dis",
                "p_elc", "p_uhv", "p_curt", "e_bess", "h_tank"]:
        prof = np.zeros(24)
        for i, s in enumerate(scenarios):
            if key in s and len(s[key]) >= hour_end:
                prof += weights[i] * np.array(s[key][hour_start:hour_end])
        profiles[key] = prof

    profiles["hour"] = np.arange(24)
    profiles["season"] = season
    return profiles


def plot_typical_days(tssp_res, fig_dir="results/figures archive"):
    """Plot 4-season typical day dispatch (core paper figure)."""
    os.makedirs(fig_dir, exist_ok=True)
    seasons = ["spring", "summer", "autumn", "winter"]
    season_titles = ["Spring (Mar-May)", "Summer (Jun-Aug)", "Autumn (Sep-Nov)", "Winter (Dec-Feb)"]

    fig, axes = plt.subplots(2, 2, figsize=(7.48, 5.5), dpi=300)
    axes = axes.flatten()

    colors = {
        "wind_avail": "#4C78A8", "pv_avail": "#F58518", "therm": "#E45756",
        "p_bess_dis": "#72B7B2", "p_bess_ch": "#72B7B2",
        "p_elc": "#54A24B", "p_uhv": "#9D755D", "p_curt": "#BAB0AC",
    }

    for ax, season, title in zip(axes, seasons, season_titles):
        prof = extract_typical_day(tssp_res, season=season)
        if prof is None:
            ax.set_title(f"{title} (No Data)", fontsize=8)
            continue

        h = prof["hour"]
        # Stacked area for generation
        gen_stack = np.vstack([
            prof.get("wind_avail", np.zeros(24)),
            prof.get("pv_avail", np.zeros(24)),
            prof.get("therm", np.zeros(24)),
        ])
        ax.stackplot(h, gen_stack,
                     labels=["Wind", "Solar", "Thermal"],
                     colors=[colors["wind_avail"], colors["pv_avail"], colors["therm"]],
                     alpha=0.85, edgecolor='white', linewidth=0.3)

        # Line for load and UHV
        load = prof.get("p_uhv", np.zeros(24)) + prof.get("p_elc", np.zeros(24)) + prof.get("p_curt", np.zeros(24))
        ax.plot(h, load, color="#2D2D2D", linewidth=1.2, label="Net Load + Export", linestyle='--')

        ax.set_xlim(0, 23)
        ax.set_xlabel("Hour", fontsize=7)
        ax.set_ylabel("Power (MW)", fontsize=7)
        ax.set_title(title, fontsize=8, fontweight='bold')
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=6, loc='upper left', frameon=False)
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    fig.savefig(f"{fig_dir}/typical_days_dispatch.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Saved {fig_dir}/typical_days_dispatch.png")


# ==============================================================================
# 4. Main Orchestrator
# ==============================================================================

def run_all_sensitivity(tssp_res, ev_res, econ, phys, solver_cfg,
                        results_dir="results/sensitivity", fig_dir="results/figures archive"):
    """
    Run all sensitivity analyses and generate plots.
    Note: carbon sweep and tornado require model re-solve and are
    commented out by default due to computational cost.
    """
    os.makedirs(results_dir, exist_ok=True)

    # 1. Typical day dispatch (fast, no re-solve needed)
    if tssp_res:
        plot_typical_days(tssp_res, fig_dir=fig_dir)

    # 2. Carbon sweep and tornado require model re-solve.
    #    These should be called from a dedicated script after main pipeline.
    print(f"\n[Sensitivity] Typical day plots generated.")
    print(f"[Sensitivity] Carbon sweep & tornado require model re-solve.")
    print(f"              Run run_carbon_sweep() and run_tornado_sensitivity()")
    print(f"              from a dedicated script when computational resources allow.")
