"""
Cross-validation script for all core data assets.
Run: python scripts/cross_validate_data.py
"""
import json, csv, sys

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_csv(path):
    data = []
    with open(path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data

fev2 = load_json('results/tables/full_experiment_v2.json')
f1000 = load_json('results/tables/full_experiment_1000t.json')
h2_data = load_csv('results/tables/h2_tank_sensitivity_v3.csv')
cp_data = load_csv('results/tables/carbon_price_sensitivity_CONSOLIDATED.csv')
cs1000 = load_csv('results/tables/carbon_sweep_1000t.csv')
seed_data = load_csv('results/tables/seed_robustness_summary.csv')

issues = []
warnings = []
infos = []

def check(cond, msg, level='ISSUE'):
    if not cond:
        if level == 'ISSUE':
            issues.append(msg)
        elif level == 'WARN':
            warnings.append(msg)
    else:
        infos.append('OK: ' + msg)

def print_header(title):
    print('\n' + '=' * 70)
    print(title)
    print('=' * 70)

# ============================================================
# 1. 400t BENCHMARK
# ============================================================
print_header('[1] 400t BENCHMARK: EV vs TSSP Capacity & Cost')
ev_cap = fev2['ev']['capacity']
ts_cap = fev2['tssp']['capacity']

for k in ['BESS_P_MW', 'BESS_E_MWh', 'ELC_P_MW', 'FC_P_MW', 'H2_Tank_kg']:
    infos.append('EV %s=%.0f, TSSP %s=%.0f' % (k, ev_cap[k], k, ts_cap[k]))

ev_dur = ev_cap['BESS_E_MWh'] / ev_cap['BESS_P_MW']
ts_dur = ts_cap['BESS_E_MWh'] / ts_cap['BESS_P_MW']
infos.append('BESS duration: EV=%.2fh, TSSP=%.2fh' % (ev_dur, ts_dur))
check(ts_dur >= 2.0, 'BESS duration %.2fh < 2h minimum' % ts_dur, 'WARN')

# Cost arithmetic
ev_co = fev2['ev']['costs']
ts_co = fev2['tssp']['costs']

def check_cost_arithmetic(costs, label):
    inv = costs.get('inv', 0)
    om = costs.get('om', costs.get('om_fix', 0))
    op = costs.get('op', costs.get('op_exp', 0))
    rev = costs.get('revenue', costs.get('revenue_exp', 0))
    total = inv + om + op - rev
    infos.append('%s: inv=%.1f + om=%.1f + op=%.1f - rev=%.1f = %.1f' % (label, inv, om, op, rev, total))
    return total

ev_total = check_cost_arithmetic(ev_co, 'EV')
ts_total = check_cost_arithmetic(ts_co, 'TSSP')

# Check EV objval consistency
ev_obj = fev2['ev']['objval']
ts_obj = fev2['tssp']['objval']
check(abs(ev_total - ev_obj) < abs(ev_obj)*0.01,
      'EV cost breakdown sum (%.1f) vs objval (%.1f)' % (ev_total, ev_obj), 'ISSUE')
check(abs(ts_total - ts_obj) < abs(ts_obj)*0.01,
      'TSSP cost breakdown sum (%.1f) vs objval (%.1f)' % (ts_total, ts_obj), 'ISSUE')

# Model version consistency check
import os
from datetime import datetime
v2_time = os.path.getmtime('results/tables/full_experiment_v2.json')
v3_time = os.path.getmtime('results/tables/h2_tank_sensitivity_v3.csv')
model_time = os.path.getmtime('src/deterministic_model.py')
infos.append('full_experiment_v2.json: %s' % datetime.fromtimestamp(v2_time))
infos.append('h2_tank_sensitivity_v3.csv: %s' % datetime.fromtimestamp(v3_time))
infos.append('src/deterministic_model.py: %s' % datetime.fromtimestamp(model_time))
if v2_time < model_time and v3_time >= model_time:
    issues.append('CRITICAL: full_experiment_v2.json was generated BEFORE May 7 model update; '
                  'h2_tank_sensitivity_v3.csv was generated AFTER. Different model formulations.')

# VSS
vss_rig = fev2.get('vss_rigorous', {})
if vss_rig and 'vss' in vss_rig:
    vss_val = vss_rig['vss']
    infos.append('Rigorous VSS = %.0f (%.3f%%)' % (vss_val, vss_val/abs(ev_obj)*100))
    check(vss_val > 0, 'VSS=%.0f should be positive' % vss_val, 'ISSUE')
    check(vss_val/abs(ev_obj) < 0.02, 'VSS=%.2f%% > 2%% unusually large' % (vss_val/abs(ev_obj)*100), 'WARN')
else:
    warnings.append('vss_rigorous not found or incomplete')

simple_vss = (-ev_obj) - (-ts_obj)
infos.append('Simple VSS = %.0f (%.3f%%)' % (simple_vss, simple_vss/abs(ev_obj)*100))

# ============================================================
# 2. H2 SENSITIVITY
# ============================================================
print_header('[2] H2 SENSITIVITY: Cross-file consistency')

print('  Raw data from h2_tank_sensitivity_v3.csv:')
h2_map = {}
for row in h2_data:
    tank = float(row['H2_Tank_t'])
    h2_map[int(tank)] = row
    bess_ev = float(row['EV_BESS_P_MW'])
    bess_ts = float(row['TSSP_BESS_P_MW'])
    elc_ev = float(row['EV_ELC_MW'])
    elc_ts = float(row['TSSP_ELC_MW'])
    fc_ev = float(row['EV_FC_MW'])
    fc_ts = float(row['TSSP_FC_MW'])
    print('    %dt: EV BESS=%.0f ELC=%.0f FC=%.0f | TSSP BESS=%.0f ELC=%.0f FC=%.0f' %
          (int(tank), bess_ev, elc_ev, fc_ev, bess_ts, elc_ts, fc_ts))

# Check 400t consistency
if 400 in h2_map:
    row = h2_map[400]
    for col, csv_ev, csv_ts in [
        ('BESS_P_MW', 'EV_BESS_P_MW', 'TSSP_BESS_P_MW'),
        ('ELC_P_MW', 'EV_ELC_MW', 'TSSP_ELC_MW'),
        ('FC_P_MW', 'EV_FC_MW', 'TSSP_FC_MW')
    ]:
        ev_csv = float(row[csv_ev])
        ts_csv = float(row[csv_ts])
        ev_json = ev_cap[col]
        ts_json = ts_cap[col]
        tol = max(abs(ev_json)*0.02, 5)
        check(abs(ev_csv - ev_json) < tol,
              '400t EV %s: CSV=%.0f vs JSON=%.0f' % (col, ev_csv, ev_json))
        check(abs(ts_csv - ts_json) < tol,
              '400t TSSP %s: CSV=%.0f vs JSON=%.0f' % (col, ts_csv, ts_json))

# Check 1000t consistency
if 1000 in h2_map:
    row = h2_map[1000]
    f1000_ev_cap = f1000['ev']['capacity']
    f1000_ts_cap = f1000['tssp']['capacity']
    for col, csv_ev, csv_ts in [
        ('BESS_P_MW', 'EV_BESS_P_MW', 'TSSP_BESS_P_MW'),
        ('ELC_P_MW', 'EV_ELC_MW', 'TSSP_ELC_MW'),
        ('FC_P_MW', 'EV_FC_MW', 'TSSP_FC_MW')
    ]:
        ev_csv = float(row[csv_ev])
        ts_csv = float(row[csv_ts])
        ev_json = f1000_ev_cap[col]
        ts_json = f1000_ts_cap[col]
        tol = max(abs(ev_json)*0.02, 5)
        check(abs(ev_csv - ev_json) < tol,
              '1000t EV %s: CSV=%.0f vs JSON=%.0f' % (col, ev_csv, ev_json))
        check(abs(ts_csv - ts_json) < tol,
              '1000t TSSP %s: CSV=%.0f vs JSON=%.0f' % (col, ts_csv, ts_json))

# Internal logic: trend checks
bess_ts_vals = []
for t in [200, 400, 800, 1000]:
    if t in h2_map:
        bess_ts_vals.append((t, float(h2_map[t]['TSSP_BESS_P_MW'])))

infos.append('H2 BESS trend: ' + ' -> '.join(['%dt:%.0f' % (t,v) for t,v in bess_ts_vals]))

# Check 200->1000 drop claim
bess_200 = float(h2_map[200]['TSSP_BESS_P_MW'])
bess_1000 = float(h2_map[1000]['TSSP_BESS_P_MW'])
pct_drop = (bess_200 - bess_1000) / bess_200 * 100
infos.append('BESS drop 200t->1000t: %.1f%% (claimed 51.6%%)' % pct_drop)
check(abs(pct_drop - 51.6) < 1.0,
      'BESS drop = %.1f%% vs claimed 51.6%%' % pct_drop, 'ISSUE')

# ELC trend
elc_ts_vals = []
for t in [200, 400, 800, 1000]:
    if t in h2_map:
        elc_ts_vals.append((t, float(h2_map[t]['TSSP_ELC_MW'])))
infos.append('H2 ELC trend: ' + ' -> '.join(['%dt:%.0f' % (t,v) for t,v in elc_ts_vals]))

# ============================================================
# 3. CARBON SWEEP
# ============================================================
print_header('[3] CARBON SWEEP')

cs_v2 = fev2.get('carbon_sweep', [])
if isinstance(cs_v2, list):
    print('  From full_experiment_v2.json:')
    for item in cs_v2:
        cap = item.get('carbon_cap_Mt', 'N/A')
        bess = item.get('bess_MW', 0)
        elc = item.get('elc_MW', 0)
        fc = item.get('fc_MW', 0)
        obj = item.get('objval_M_CNY', 0)
        infos.append('CS %sMt: BESS=%.0f ELC=%.0f FC=%.0f obj=%.2f' % (cap, bess, elc, fc, obj))
else:
    infos.append('carbon_sweep in JSON is type %s' % type(cs_v2).__name__)

print('  From carbon_sweep_1000t.csv:')
for row in cs1000:
    cap = row.get('carbon_cap_Mt', 'N/A')
    bess = float(row.get('bess_MW', 0))
    elc = float(row.get('elc_MW', 0))
    fc = float(row.get('fc_MW', 0))
    infos.append('CS1000 %sMt: BESS=%.0f ELC=%.0f FC=%.0f' % (cap, bess, elc, fc))

# ============================================================
# 4. CARBON PRICE SENSITIVITY
# ============================================================
print_header('[4] CARBON PRICE SENSITIVITY')
print('  Rows: %d' % len(cp_data))
for row in cp_data:
    cp = float(row.get('carbon_price_cny_per_ton', 0))
    bess = float(row.get('BESS_P_MW', 0))
    elc = float(row.get('ELC_P_MW', 0))
    fc = float(row.get('FC_P_MW', 0))
    emis = float(row.get('carbon_emission_Mt', 0))
    gap = row.get('mipgap_pct', 'N/A')
    status = row.get('status', 'N/A')
    infos.append('CP=%.0f: BESS=%.0f ELC=%.0f FC=%.0f Emis=%.2f Gap=%s%% Status=%s' %
                 (cp, bess, elc, fc, emis, gap, status))
    if status == 'OPTIMAL':
        try:
            g = float(gap)
            check(g < 3.0, 'CP=%.0f OPTIMAL but Gap=%.2f%% > 3%%' % (cp, g), 'WARN')
        except:
            pass
    if emis > 16:
        warnings.append('CP=%.0f emissions=%.2fMt > 16Mt' % (cp, emis))
    if bess > 10000:
        warnings.append('CP=%.0f BESS=%.0fMW > 10GW' % (cp, bess))

# Cross-check: carbon sweep 15Mt vs carbon price 80
cs_15 = None
for item in cs_v2 if isinstance(cs_v2, list) else []:
    if float(item.get('carbon_cap_Mt', 0)) == 15.0:
        cs_15 = item
        break

cp_80 = None
for row in cp_data:
    if float(row.get('carbon_price_cny_per_ton', 0)) == 80:
        cp_80 = row
        break

if cs_15 and cp_80:
    print('\n  [4b] BASELINE CROSS-CHECK (15Mt cap ~ 80 CNY/t baseline):')
    print('    Carbon sweep 15Mt: BESS=%.0f ELC=%.0f FC=%.0f' % (cs_15['bess_MW'], cs_15['elc_MW'], cs_15['fc_MW']))
    print('    Carbon price 80/t: BESS=%.0f ELC=%.0f FC=%.0f' % (float(cp_80['BESS_P_MW']), float(cp_80['ELC_P_MW']), float(cp_80['FC_P_MW'])))
    check(abs(cs_15['bess_MW'] - float(cp_80['BESS_P_MW'])) < 200,
          'Baseline BESS: sweep=%.0f vs price=%.0f' % (cs_15['bess_MW'], float(cp_80['BESS_P_MW'])))
    check(abs(cs_15['elc_MW'] - float(cp_80['ELC_P_MW'])) < 1000,
          'Baseline ELC: sweep=%.0f vs price=%.0f' % (cs_15['elc_MW'], float(cp_80['ELC_P_MW'])))

# ============================================================
# 5. SEED ROBUSTNESS
# ============================================================
print_header('[5] SEED ROBUSTNESS')
for row in seed_data:
    metric = row.get('metric', 'N/A')
    mean = float(row.get('mean', 0))
    std = float(row.get('std', 0))
    cv = float(row.get('cv_pct', 0))
    infos.append('Seed %s: mean=%.1f std=%.1f CV=%.2f%%' % (metric, mean, std, cv))
    if cv > 30:
        warnings.append('%s CV=%.1f%% very high' % (metric, cv))
    elif cv > 10:
        warnings.append('%s CV=%.1f%% high (disclose in appendix)' % (metric, cv))

# ============================================================
# 6. BESS DURATION
# ============================================================
print_header('[6] BESS DURATION ACROSS EXPERIMENTS')

# 400t baseline
dur_400 = ts_cap['BESS_E_MWh'] / ts_cap['BESS_P_MW']
infos.append('400t baseline: %.2fh' % dur_400)

# 1000t
dur_1000 = f1000['tssp']['capacity']['BESS_E_MWh'] / f1000['tssp']['capacity']['BESS_P_MW']
infos.append('1000t baseline: %.2fh' % dur_1000)

# Carbon price points
cp_durations = []
for row in cp_data:
    cp = float(row.get('carbon_price_cny_per_ton', 0))
    bp = float(row.get('BESS_P_MW', 0))
    be = float(row.get('BESS_E_MWh', 0))
    if bp > 0:
        cp_durations.append((cp, be/bp))

cp_durations.sort()
infos.append('CP duration trend: ' + ' -> '.join(['%.0f:%.2fh' % (cp,d) for cp,d in cp_durations]))

for i in range(len(cp_durations)-1):
    cp1, d1 = cp_durations[i]
    cp2, d2 = cp_durations[i+1]
    if d2 < d1 - 0.5:
        warnings.append('Duration drops: %.0f(%.2fh)->%.0f(%.2fh)' % (cp1, d1, cp2, d2))

# ============================================================
# 7. KAN BASELINE
# ============================================================
print_header('[7] KAN BASELINE COMPARISON')
base_data = load_csv('results/tables/baseline_comparison_full.csv')
for row in base_data:
    model = row.get('model', '')
    target = row.get('target', '')
    crps = float(row.get('crps', 0))
    rmse = float(row.get('rmse', 0))
    time = float(row.get('train_time_sec', 0))
    infos.append('%s/%s: RMSE=%.4f CRPS=%.4f Time=%.1fs' % (model, target, rmse, crps, time))

# Speedup check
kan_times = {}
lstm_times = {}
for row in base_data:
    model = row.get('model', '')
    target = row.get('target', '')
    time = float(row.get('train_time_sec', 0))
    if model == 'KAN':
        kan_times[target] = time
    elif model == 'LSTM':
        lstm_times[target] = time

for target in ['wind', 'solar']:
    if target in kan_times and target in lstm_times:
        ratio = lstm_times[target] / kan_times[target]
        infos.append('%s speedup LSTM/KAN = %.1fx' % (target, ratio))
        check(ratio > 2, '%s LSTM/KAN speedup = %.1fx (< 2x)' % (target, ratio), 'WARN')

# CRPS comparison
kan_crps = {}
lstm_crps = {}
for row in base_data:
    model = row.get('model', '')
    target = row.get('target', '')
    crps = float(row.get('crps', 0))
    if model == 'KAN':
        kan_crps[target] = crps
    elif model == 'LSTM':
        lstm_crps[target] = crps

for target in ['wind', 'solar']:
    if target in kan_crps and target in lstm_crps:
        improvement = (lstm_crps[target] - kan_crps[target]) / lstm_crps[target] * 100
        infos.append('%s KAN CRPS vs LSTM = %+.1f%%' % (target, improvement))

# ============================================================
# 8. SSE ELASTICITY
# ============================================================
print_header('[8] SSE ELASTICITY VERIFICATION')

h2_vals = []
bess_ts_vals = []
bess_ev_vals = []
for t in [200, 400, 800, 1000]:
    if t in h2_map:
        h2_vals.append(t)
        bess_ts_vals.append(float(h2_map[t]['TSSP_BESS_P_MW']))
        bess_ev_vals.append(float(h2_map[t]['EV_BESS_P_MW']))

print('  TSSP Elasticities:')
for i in range(len(h2_vals)-1):
    h1, h2 = h2_vals[i], h2_vals[i+1]
    b1, b2 = bess_ts_vals[i], bess_ts_vals[i+1]
    dh = h2 - h1
    db = b2 - b1
    h_mean = (h1 + h2) / 2
    b_mean = (b1 + b2) / 2
    eps = -(db / dh) * (h_mean / b_mean) if dh != 0 and b_mean != 0 else 0
    print('    %d-%dt: dBESS=%.1f, eps=%.3f' % (h1, h2, db, eps))

print('  EV Elasticities:')
for i in range(len(h2_vals)-1):
    h1, h2 = h2_vals[i], h2_vals[i+1]
    b1, b2 = bess_ev_vals[i], bess_ev_vals[i+1]
    dh = h2 - h1
    db = b2 - b1
    h_mean = (h1 + h2) / 2
    b_mean = (b1 + b2) / 2
    eps = -(db / dh) * (h_mean / b_mean) if dh != 0 and b_mean != 0 else 0
    print('    %d-%dt: dBESS=%.1f, eps=%.3f' % (h1, h2, db, eps))

# ============================================================
# FINAL SUMMARY
# ============================================================
print_header('FINAL SUMMARY')
print('Issues: %d' % len(issues))
for i in issues:
    print('  x %s' % i)
print('Warnings: %d' % len(warnings))
for w in warnings:
    print('  ! %s' % w)
print('Infos: %d (displaying first 40)' % len(infos))
for info in infos[:40]:
    print('  %s' % info)
if len(infos) > 40:
    print('  ... and %d more' % (len(infos)-40))

print('\n' + '=' * 70)
if issues:
    print('RESULT: FAIL — Critical inconsistencies. Do NOT start writing.')
    sys.exit(1)
elif warnings:
    print('RESULT: PASS with WARNINGS — Data consistent. Warnings are discloseable.')
else:
    print('RESULT: PASS — All data validated. Ready for writing.')
print('=' * 70)
