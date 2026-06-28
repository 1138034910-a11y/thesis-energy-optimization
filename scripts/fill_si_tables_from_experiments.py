"""
Fill Supplementary Note 12 (scenario-count robustness) and Note 12 (Copula
sensitivity) tables from experimental CSV outputs, then regenerate the merged
manuscript.

Inputs:
    results/tables/scenario_count_comparison.csv
    results/tables/copula_sensitivity_v3.csv
    results/tables/h2_sensitivity_v3_rigorous_vss.csv

Outputs:
    submission_package/supplementary/section_08_supplementary_material.md  (tables updated in place)
    submission_package/manuscript/manuscript_complete_v2_ecm_work.md             (regenerated via merge_manuscript.py)
"""
import os
import re
import sys
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

_project_root = Path(__file__).resolve().parent.parent
os.chdir(_project_root)
sys.path.insert(0, str(_project_root))

SECTION_PATH = _project_root / "submission_package" / "supplementary" / "section_08_supplementary_material.md"
MERGE_SCRIPT = _project_root / "scripts" / "merge_manuscript.py"


def fmt_int(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "*[missing]*"
    return f"{int(round(x)):,}"


def fmt_float(x, decimals=2):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "*[missing]*"
    return f"{x:.{decimals}f}"


def fmt_pct(x, decimals=2):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "*[missing]*"
    return f"{x * 100:.{decimals}f}"


def compute_sse(df_rho, cap_col="H2_Tank_t", bess_col="TSSP_BESS_P_MW"):
    """Arc elasticity of BESS power w.r.t. H2 tank capacity."""
    df_rho = df_rho.sort_values(cap_col).reset_index(drop=True)
    sses = {}
    for i, row in df_rho.iterrows():
        if i == 0:
            sses[row[cap_col]] = np.nan
        else:
            prev = df_rho.iloc[i - 1]
            x0, x1 = prev[cap_col], row[cap_col]
            y0, y1 = prev[bess_col], row[bess_col]
            if x1 != x0 and (y0 + y1) > 0:
                sses[row[cap_col]] = (y1 - y0) / (x1 - x0) * (x0 + x1) / (y0 + y1)
            else:
                sses[row[cap_col]] = np.nan
    return sses


def fill_table_s8(text, csv_path):
    if not csv_path.exists():
        print(f"[WARN] Scenario-count CSV not found: {csv_path}")
        return text

    df = pd.read_csv(csv_path)
    if df.empty:
        print("[WARN] Scenario-count CSV is empty")
        return text

    row = df.iloc[0]

    # Main-protocol 4-scenario reference
    ref_csv = _project_root / "results" / "tables" / "h2_sensitivity_v3_rigorous_vss.csv"
    if ref_csv.exists():
        df_ref = pd.read_csv(ref_csv)
        ref_row = df_ref[df_ref["H2_Tank_t"] == 400].iloc[0]
    else:
        ref_row = None
        print(f"[WARN] Main protocol reference not found: {ref_csv}")

    # Known model-size values for the 4-scenario protocol (from Note 11 text)
    model_4s = {
        "binary": 3_840,
        "continuous": 6_245,
        "constraints": 18_719,
        "nonzeros": 58_410,
    }

    def val(row_or_ref, key, default=None):
        if row_or_ref is None:
            return default
        try:
            v = row_or_ref[key]
            if pd.isna(v):
                return default
            return v
        except KeyError:
            return default

    # Build new table
    lines = [
        "| Item | 4 scenarios (main protocol) | 8 scenarios (robustness test) |",
        "|:---|---:|---:|",
        f"| Binary variables | {fmt_int(model_4s['binary'])} | {fmt_int(val(row, 'n_binary_variables'))} |",
        f"| Continuous variables | {fmt_int(model_4s['continuous'])} | {fmt_int(val(row, 'n_variables', 0) - val(row, 'n_binary_variables', 0))} |",
        f"| Constraints | {fmt_int(model_4s['constraints'])} | {fmt_int(val(row, 'n_constraints'))} |",
        f"| Nonzeros | {fmt_int(model_4s['nonzeros'])} | {fmt_int(val(row, 'n_nonzeros'))} |",
        f"| TSSP objective (10$^4$ CNY) | {fmt_float(val(ref_row, 'TSSP_Obj')) if ref_row is not None else '*[missing]*'} | {fmt_float(val(row, 'final_obj'))} |",
        f"| Final MIPGap (%) | {fmt_pct(val(ref_row, 'TSSP_Gap_pct'), decimals=2) if ref_row is not None else '*[missing]*'} | {fmt_pct(val(row, 'final_gap_pct') / 100 if val(row, 'final_gap_pct') is not None else None, decimals=2)} |",
        f"| Wall-clock time (s) | {fmt_int(val(ref_row, 'TSSP_Time_s')) if ref_row is not None else '*[missing]*'} | {fmt_int(val(row, 'solve_time_s'))} |",
        f"| BESS power (MW) | {fmt_int(val(ref_row, 'TSSP_BESS_P_MW')) if ref_row is not None else '*[missing]*'} | {fmt_int(val(row, 'final_BESS_P_MW'))} |",
        f"| BESS energy (MWh) | {fmt_int(val(ref_row, 'TSSP_BESS_E_MWh')) if ref_row is not None else '*[missing]*'} | {fmt_int(val(row, 'final_BESS_E_MWh'))} |",
        f"| ELC power (MW) | {fmt_int(val(ref_row, 'TSSP_ELC_MW')) if ref_row is not None else '*[missing]*'} | {fmt_int(val(row, 'final_ELC_MW'))} |",
        f"| FC power (MW) | {fmt_int(val(ref_row, 'TSSP_FC_MW')) if ref_row is not None else '*[missing]*'} | {fmt_int(val(row, 'final_FC_MW'))} |",
    ]

    new_table = "\n".join(lines)

    # Replace between the table header and the next paragraph
    pattern = r"(\*\*Table S8\. Four-scenario versus eight-scenario comparison at the 400 t base case\.\*\*\n\n)(.*?)(\n\nTwo observations guide the interpretation)"
    new_text, n = re.subn(pattern, lambda m: m.group(1) + new_table + m.group(3), text, flags=re.DOTALL)
    if n == 0:
        print("[WARN] Could not locate Table S8 in section_08; no replacement made")
    else:
        print("[OK] Table S8 filled")
    return new_text


def fill_table_s9(text, csv_path):
    if not csv_path.exists():
        print(f"[WARN] Copula sensitivity CSV not found: {csv_path}")
        return text

    df = pd.read_csv(csv_path)
    if df.empty:
        print("[WARN] Copula sensitivity CSV is empty")
        return text

    # Main-protocol reference values at 200, 400, 1000 t
    ref_csv = _project_root / "results" / "tables" / "h2_sensitivity_v3_rigorous_vss.csv"
    if ref_csv.exists():
        df_ref = pd.read_csv(ref_csv)
        df_ref = df_ref[df_ref["H2_Tank_t"].isin([200, 400, 1000])][["H2_Tank_t", "TSSP_BESS_P_MW"]].copy()
        df_ref["rho"] = -0.30
    else:
        df_ref = pd.DataFrame(columns=["H2_Tank_t", "TSSP_BESS_P_MW", "rho"])
        print(f"[WARN] Main protocol reference not found: {ref_csv}")

    # Compute SSE for the alternative rhos if not already present
    if "SSE" not in df.columns:
        df["SSE"] = np.nan
    for rho in df["rho"].unique():
        mask = df["rho"] == rho
        sse_map = compute_sse(df[mask])
        for cap, sse_val in sse_map.items():
            df.loc[(df["rho"] == rho) & (df["H2_Tank_t"] == cap), "SSE"] = sse_val

    # Compute SSE for the main-protocol reference as well (over the same 3 points)
    if not df_ref.empty:
        df_ref["SSE"] = np.nan
        sse_map_ref = compute_sse(df_ref)
        for cap, sse_val in sse_map_ref.items():
            df_ref.loc[df_ref["H2_Tank_t"] == cap, "SSE"] = sse_val

    # Combine with main protocol
    df_all = pd.concat([df, df_ref], ignore_index=True)

    def get_vals(rho):
        sub = df_all[df_all["rho"] == rho].sort_values("H2_Tank_t").reset_index(drop=True)
        out = {}
        for _, row in sub.iterrows():
            h2 = int(row["H2_Tank_t"])
            out[h2] = {
                "p": row["TSSP_BESS_P_MW"],
                "sse": row["SSE"] if "SSE" in row and pd.notna(row["SSE"]) else None,
            }
        return out

    v20 = get_vals(-0.20)
    v30 = get_vals(-0.30)
    v41 = get_vals(-0.41)

    h2_points = [200, 400, 1000]
    lines = [
        "| H$_2$ scale (t) | $\\rho = -0.20$ BESS P (MW) | $\\rho = -0.20$ SSE | $\\rho = -0.30$ BESS P (MW) | $\\rho = -0.30$ SSE | $\\rho = -0.41$ BESS P (MW) | $\\rho = -0.41$ SSE |",
        "|:---|---:|---:|---:|---:|---:|---:|",
    ]
    for h2 in h2_points:
        p20 = v20.get(h2, {}).get("p")
        s20 = v20.get(h2, {}).get("sse")
        p30 = v30.get(h2, {}).get("p")
        s30 = v30.get(h2, {}).get("sse")
        p41 = v41.get(h2, {}).get("p")
        s41 = v41.get(h2, {}).get("sse")
        if h2 == 200:
            # Base point: no SSE
            lines.append(
                f"| {h2} | {fmt_int(p20)} | — | {fmt_int(p30)} | — | {fmt_int(p41)} | — |"
            )
        else:
            lines.append(
                f"| {h2} | {fmt_int(p20)} | {fmt_float(s20, decimals=3)} | {fmt_int(p30)} | {fmt_float(s30, decimals=3)} | {fmt_int(p41)} | {fmt_float(s41, decimals=3)} |"
            )

    new_table = "\n".join(lines)

    pattern = r"(\*\*Table S9\. SSE sign pattern under alternative Gaussian Copula correlations\.\*\*\n\n)(.*?)(\n\nNotes\. SSE is computed)"
    new_text, n = re.subn(pattern, lambda m: m.group(1) + new_table + m.group(3), text, flags=re.DOTALL)
    if n == 0:
        print("[WARN] Could not locate Table S9 in section_08; no replacement made")
    else:
        print("[OK] Table S9 filled")
    return new_text


def main():
    if not SECTION_PATH.exists():
        print(f"[FAIL] {SECTION_PATH} not found")
        sys.exit(1)

    text = SECTION_PATH.read_text(encoding="utf-8")

    scenario_csv = _project_root / "results" / "tables" / "scenario_count_comparison.csv"
    copula_csv = _project_root / "results" / "tables" / "copula_sensitivity_v3.csv"

    text = fill_table_s8(text, scenario_csv)
    text = fill_table_s9(text, copula_csv)

    # Backup and write
    backup = SECTION_PATH.with_suffix(".md.bak")
    SECTION_PATH.replace(backup)
    SECTION_PATH.write_text(text, encoding="utf-8")
    print(f"[OK] Updated {SECTION_PATH} (backup: {backup})")

    # Regenerate merged manuscript
    if MERGE_SCRIPT.exists():
        subprocess.run([sys.executable, str(MERGE_SCRIPT)], check=True)
        print("[OK] Regenerated manuscript_complete_v2.md")
    else:
        print(f"[WARN] Merge script not found: {MERGE_SCRIPT}")


if __name__ == "__main__":
    main()
