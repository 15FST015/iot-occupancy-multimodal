#!/usr/bin/env python3
"""baselines.py — Baseline eksperimen Dataset A (v0).

Model   : RandomForest + XGBoost (full-modal) + XGBoost per-modalitas (7 grup)
Fitur   : 16 kanal numerik z-scored + 4 biner + jam (sin/cos) + weekend
Missing : imputasi median (=0 setelah z-score) — baseline konvensional
Evaluasi: per-row + agregasi blok 5 menit (majority vote), metrik lengkap
Output  : experiments/results/baseline_*.json + ringkasan stdout

Catatan: evaluasi per-row mewarisi autokorelasi temporal 10s; agregasi blok
5 menit sebagai cek ketahanan. Day-level CV menyusul di eksperimen lanjutan.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, average_precision_score,
                             matthews_corrcoef, balanced_accuracy_score)
import xgboost as xgb

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)
SEED = 42

NUM_COLS = [
    "co2", "humidity", "temperature_1", "temperature_2",
    "lux_1", "lux_2", "sound",
    "server_cur_current", "server_cur_power", "server_cur_voltage",
    "pfsense_cur_current", "pfsense_cur_power", "pfsense_cur_voltage",
    "socket_cur_current", "socket_cur_power", "socket_cur_voltage",
]
BIN_COLS = ["lamp_switch_led", "switch_channel_1", "switch_channel_2", "switch_channel_3"]
MODALITY_GROUPS = {
    "env_air": ["co2", "humidity", "temperature_1", "temperature_2"],
    "light": ["lux_1", "lux_2"],
    "acoustic": ["sound"],
    "elec_server": ["server_cur_current", "server_cur_power", "server_cur_voltage"],
    "elec_pfsense": ["pfsense_cur_current", "pfsense_cur_power", "pfsense_cur_voltage"],
    "elec_socket": ["socket_cur_current", "socket_cur_power", "socket_cur_voltage"],
    "device_state": ["lamp_switch_led", "switch_channel_1", "switch_channel_2", "switch_channel_3"],
}


def load_split():
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    # pandas 3.x gagal infer tz-aware timestamp campuran (T/spasi) → parse eksplisit
    train["createdAt"] = pd.to_datetime(train["createdAt"], format="ISO8601")
    test["createdAt"] = pd.to_datetime(test["createdAt"], format="ISO8601")
    return train, test


def add_features(df):
    df = df.copy()
    hour = df["createdAt"].dt.hour + df["createdAt"].dt.minute / 60.0
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    df["weekend"] = (df["createdAt"].dt.weekday >= 5).astype(int)
    return df


def metrics(y_true, y_score, y_pred):
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_score)), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_score)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "mcc": round(float(matthews_corrcoef(y_true, y_pred)), 4),
        "balanced_acc": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
    }


def block_metrics(df, y_true_col, y_pred_col, minutes=5):
    """Agregasi majority-vote per blok waktu."""
    d = df.copy()
    d["block"] = d["createdAt"].dt.floor(f"{minutes}min")
    g = d.groupby("block").agg(
        y_true=(y_true_col, lambda s: s.mode().iloc[0]),
        y_pred=(y_pred_col, lambda s: s.mode().iloc[0]),
        n=("block", "size"),
    )
    return metrics(g["y_true"], g["y_pred"], g["y_pred"])  # score=pred utk blok (tidak ada prob agg di sini)


def run_model(name, Xtr, ytr, Xva, yva, Xte, yte, model_kind="xgb"):
    if model_kind == "xgb":
        pos = int(ytr.sum()); neg = int(len(ytr) - pos)
        model = xgb.XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=neg / max(pos, 1),
            eval_metric="aucpr", early_stopping_rounds=30,
            random_state=SEED, n_jobs=-1)
        model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        best_iter = model.best_iteration if hasattr(model, "best_iteration") else None
    else:
        model = RandomForestClassifier(
            n_estimators=200, class_weight="balanced",
            random_state=SEED, n_jobs=-1)
        model.fit(Xtr, ytr)
        best_iter = None

    y_score = model.predict_proba(Xte)[:, 1]
    y_pred = (y_score >= 0.5).astype(int)
    return metrics(yte, y_score, y_pred), best_iter, y_pred


def main():
    train, test = load_split()
    train = add_features(train)
    test = add_features(test)

    feat_cols = NUM_COLS + BIN_COLS + ["hour_sin", "hour_cos", "weekend"]
    # imputasi median (=0, karena z-score) untuk baseline konvensional
    Xtr = train[feat_cols].fillna(0.0)
    Xte = test[feat_cols].fillna(0.0)
    ytr = train["occupied"].values
    yte = test["occupied"].values

    # eval set day-level untuk early stopping XGB (2 hari terakhir dari train)
    days = sorted(train["createdAt"].dt.date.unique())
    eval_days = set(str(d) for d in days[-2:])
    ev = train["createdAt"].dt.date.astype(str).isin(eval_days)
    Xva, yva = Xtr[ev], ytr[ev]
    Xtr2, ytr2 = Xtr[~ev], ytr[~ev]

    results = {}
    # --- full-modal ---
    preds = {}
    for kind, label in (("xgb", "xgb_full"), ("rf", "rf_full")):
        m, bi, yp = run_model(label, Xtr2, ytr2, Xva, yva, Xte, yte, kind)
        results[label] = {"metrics": m, "best_iter": bi, "features": feat_cols}
        preds[label] = yp

    # --- per-modalitas (XGBoost saja) ---
    for gname, cols in MODALITY_GROUPS.items():
        fcols = cols + ["hour_sin", "hour_cos", "weekend"]
        Xt = train[fcols].fillna(0.0)
        Xe = test[fcols].fillna(0.0)
        m, bi, _ = run_model(f"xgb_{gname}", Xt[~ev], ytr2, Xt[ev], yva, Xe, yte, "xgb")
        results[f"xgb_{gname}"] = {"metrics": m, "best_iter": bi, "features": fcols}

    # --- agregasi blok 5 menit (majority vote) untuk full models ---
    for label, yp in preds.items():
        d = test.copy()
        d["y_pred"] = yp
        d["block"] = d["createdAt"].dt.floor("5min")
        g = d.groupby("block").agg(
            y_true=("occupied", lambda s: s.mode().iloc[0]),
            y_pred=("y_pred", lambda s: s.mode().iloc[0]),
            n=("block", "size"))
        results[label]["metrics_block5min"] = metrics(g["y_true"], g["y_pred"], g["y_pred"])

    (OUT / "baseline_summary.json").write_text(json.dumps(results, indent=1))

    print(f"{'model':<14} {'acc':>6} {'prec':>6} {'rec':>6} {'f1':>6} {'roc':>6} {'prauc':>6} {'mF1':>6} {'mcc':>6} {'bacc':>6}")
    for label, r in results.items():
        m = r["metrics"]
        print(f"{label:<14} {m['accuracy']:>6.4f} {m['precision']:>6.4f} {m['recall']:>6.4f} {m['f1']:>6.4f} "
              f"{m['roc_auc']:>6.4f} {m['pr_auc']:>6.4f} {m['macro_f1']:>6.4f} {m['mcc']:>6.4f} {m['balanced_acc']:>6.4f}")
    print("\nHasil tersimpan: ", OUT / "baseline_summary.json")


if __name__ == "__main__":
    main()
