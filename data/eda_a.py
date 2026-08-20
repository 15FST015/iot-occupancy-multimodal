#!/usr/bin/env python3
"""EDA Dataset A — stdlib+numpy, tanpa pandas."""
import csv, datetime, json
import numpy as np
from collections import Counter

PATH = '/mnt/sda3/Documents/Jurnal/Amesanggeng/iot-multimodal-occupancy/data/final_dataset_csv.csv'
OUT = '/mnt/sda3/Documents/Jurnal/Amesanggeng/iot-multimodal-occupancy/data/EDA-A.json'

rows = []
with open(PATH) as f:
    r = csv.DictReader(f)
    cols = r.fieldnames
    for row in r:
        rows.append(row)

n = len(rows)
num_cols = [c for c in cols if c != 'createdAt' and c != 'ground_truth']
stats = {}
for c in num_cols:
    vals = []
    missing = 0
    for row in rows:
        v = row[c].strip()
        if v == '':
            missing += 1
        else:
            vals.append(float(v))
    a = np.array(vals)
    stats[c] = {
        'missing': missing, 'missing_pct': round(100*missing/n, 3),
        'min': round(float(a.min()), 3), 'max': round(float(a.max()), 3),
        'mean': round(float(a.mean()), 3), 'std': round(float(a.std()), 3),
        'q25': round(float(np.percentile(a, 25)), 3), 'median': round(float(np.median(a)), 3),
        'q75': round(float(np.percentile(a, 75)), 3)}

# GT
gt = Counter(r['ground_truth'] for r in rows)
# jam (hour) distribusi okupansi
hour_occ = Counter()
for row in rows:
    if row['ground_truth'] not in ('0', ''):
        try:
            t = datetime.datetime.fromisoformat(row['createdAt'])
            hour_occ[t.hour] += 1
        except Exception:
            pass

# korelasi redundan
def corr(c1, c2):
    x, y = [], []
    for row in rows:
        a, b = row[c1].strip(), row[c2].strip()
        if a != '' and b != '':
            x.append(float(a)); y.append(float(b))
    x, y = np.array(x), np.array(y)
    return round(float(np.corrcoef(x, y)[0, 1]), 3)

out = {
    'rows': n,
    'cols': cols,
    'num_cols_stats': stats,
    'gt_dist': dict(gt),
    'gt_hourly_occupied': dict(sorted(hour_occ.items())),
    'corr_temp1_temp2': corr('temperature_1', 'temperature_2'),
    'corr_lux1_lux2': corr('lux_1', 'lux_2'),
    'corr_co2_humidity': corr('co2', 'humidity'),
}
with open(OUT, 'w') as f:
    json.dump(out, f, indent=1)
print(json.dumps(out, indent=1)[:4000])
