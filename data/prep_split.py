#!/usr/bin/env python3
"""prep_split.py (v2) — Dataset A: day-level stratified split.

Unit split = HARI (group) karena okupansi sporadis (lihat SPLIT-DESIGN.md).
  - Holdout test : 20% hari, dipilih acak-stratified berdasar total okupansi/hari
  - Train+CV     : 80% hari (untuk StratifiedGroupKFold 5-fold saat eksperimen)
  - Hari anomali 2023-12-04 dilaporkan; flag include_anomaly_day untuk mode (a)/(b)

Output: data/train.csv, data/test.csv, data/scaler.json, data/split_summary.json
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEED = 42
TEST_DAY_RATIO = 0.20

SRC = HERE / "final_dataset_csv.csv"
NUM_COLS = [
    "co2", "humidity", "temperature_1", "temperature_2",
    "lux_1", "lux_2", "sound",
    "server_cur_current", "server_cur_power", "server_cur_voltage",
    "pfsense_cur_current", "pfsense_cur_power", "pfsense_cur_voltage",
    "socket_cur_current", "socket_cur_power", "socket_cur_voltage",
]
BINARY_COLS = ["lamp_switch_led", "switch_channel_1", "switch_channel_2", "switch_channel_3"]
ANOMALY_DAY = "2023-12-04"


def main(include_anomaly_day: bool = True):
    df = pd.read_csv(SRC, parse_dates=["createdAt"])
    df = df.sort_values("createdAt").reset_index(drop=True)
    df["occupied"] = (df["ground_truth"].astype(float) > 0).astype(int)
    df["day"] = df["createdAt"].dt.date.astype(str)

    if not include_anomaly_day:
        df = df[df["day"] != ANOMALY_DAY].reset_index(drop=True)

    rng = np.random.default_rng(SEED)
    day_occ = df.groupby("day")["occupied"].sum()
    days = day_occ.index.tolist()

    # stratified pick: sort hari by occupied count, interleave into 5 bins, pick every 5th bin as test pool
    ordered = day_occ.sort_values().index.tolist()
    n_test = max(1, round(len(ordered) * TEST_DAY_RATIO))
    # ambil hari dengan okupansi tinggi + rendah secara proporsional (ACR: sorted lalu sample sistematis)
    idx = np.linspace(0, len(ordered) - 1, n_test).round().astype(int)
    test_days = set(ordered[i] for i in idx)
    train_days = [d for d in days if d not in test_days]

    train = df[df["day"].isin(train_days)].copy()
    test = df[df["day"].isin(test_days)].copy()

    # scaler fit TRAIN only
    scaler = {}
    for c in NUM_COLS:
        m, s = float(train[c].mean()), float(train[c].std())
        if s == 0 or np.isnan(s):
            s = 1.0
        scaler[c] = {"mean": m, "std": s}
        train[c] = (train[c] - m) / s
        test[c] = (test[c] - m) / s

    train.to_csv(HERE / "train.csv", index=False)
    test.to_csv(HERE / "test.csv", index=False)
    (HERE / "scaler.json").write_text(json.dumps(scaler, indent=1))

    summary = {
        "seed": SEED, "include_anomaly_day": include_anomaly_day,
        "n_days": len(days), "train_days": len(train_days), "test_days": len(test_days),
        "test_days_list": sorted(test_days),
        "rows_train": len(train), "rows_test": len(test),
        "occupied_pct_train": round(100 * train["occupied"].mean(), 3),
        "occupied_pct_test": round(100 * test["occupied"].mean(), 3),
        "occupied_rows_train": int(train["occupied"].sum()),
        "occupied_rows_test": int(test["occupied"].sum()),
        "anomaly_day_excluded": ANOMALY_DAY if not include_anomaly_day else None,
    }
    (HERE / "split_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    import sys
    mode_a = not (len(sys.argv) > 1 and sys.argv[1] == "--no-anomaly")
    main(include_anomaly_day=mode_a)
