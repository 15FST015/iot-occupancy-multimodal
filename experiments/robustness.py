#!/usr/bin/env python3
"""robustness.py — Eksperimen missing-modality pada baseline XGBoost (Dataset A).

Skenario (saat TEST, model TIDAK dilatih ulang; kanal hilang diisi mean=0 —
mencerminkan pipeline konvensional yang imputasi tanpa sadar-missing):
  1. Dropout per-modalitas (7 grup: light, env_air, acoustic, elec_server,
     elec_pfsense, elec_socket, device_state) — kanal grup di-mask SEMUA baris
  2. Dropout acak per-kanal: rate 10/30/50/70% (16 kanal numerik; biner & fitur
     waktu tidak di-mask — 0 pada biner adalah state valid)
  3. Kombinasi: light + acoustic hilang (multi-modal failure)
Output: experiments/results/robustness_summary.json + tabel degradasi.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
import xgboost as xgb

from baselines import (load_split, add_features, metrics, NUM_COLS, BIN_COLS,
                       MODALITY_GROUPS, SEED)

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)

TIME_COLS = ["hour_sin", "hour_cos", "weekend"]
FEAT_COLS = NUM_COLS + BIN_COLS + TIME_COLS


def train_xgb(Xtr, ytr, Xva, yva):
    pos = int(ytr.sum()); neg = int(len(ytr) - pos)
    model = xgb.XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=neg / max(pos, 1),
        eval_metric="aucpr", early_stopping_rounds=30,
        random_state=SEED, n_jobs=-1)
    model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    return model


def mask_cols(X, cols):
    """Mask kolom -> mean z-score (0)."""
    Xm = X.copy()
    idx = [FEAT_COLS.index(c) for c in cols]
    Xm[:, idx] = 0.0
    return Xm


def mask_random(X, rate, seed=SEED):
    """Mask acak per-nilai pada 16 kanal numerik."""
    rng = np.random.default_rng(seed)
    n_idx = [FEAT_COLS.index(c) for c in NUM_COLS]
    Xm = X.copy()
    m = rng.random((Xm.shape[0], len(n_idx))) < rate
    Xm[:, n_idx] = np.where(m, 0.0, Xm[:, n_idx])
    return Xm


def main():
    train, test = load_split()
    train = add_features(train)
    test = add_features(test)

    Xtr = train[FEAT_COLS].fillna(0.0).values
    Xte = test[FEAT_COLS].fillna(0.0).values
    ytr = train["occupied"].values
    yte = test["occupied"].values

    days = sorted(train["createdAt"].dt.date.unique())
    eval_days = set(str(d) for d in days[-2:])
    ev = train["createdAt"].dt.date.astype(str).isin(eval_days)
    Xva, yva = Xtr[ev], ytr[ev]
    Xtr2, ytr2 = Xtr[~ev], ytr[~ev]

    model = train_xgb(Xtr2, ytr2, Xva, yva)

    def ev(X):
        y_score = model.predict_proba(X)[:, 1]
        y_pred = (y_score >= 0.5).astype(int)
        return metrics(yte, y_score, y_pred)

    results = {}
    results["full"] = ev(Xte)

    # 1) dropout per-modalitas
    for gname, cols in MODALITY_GROUPS.items():
        results[f"missing_{gname}"] = ev(mask_cols(Xte, cols))

    # 2) dropout acak per-kanal
    for rate in (0.1, 0.3, 0.5, 0.7):
        results[f"random_dropout_{int(rate*100)}pct"] = ev(mask_random(Xte, rate))

    # 3) kombinasi multi-modal failure
    results["missing_light_acoustic"] = ev(mask_cols(Xte, MODALITY_GROUPS["light"] + MODALITY_GROUPS["acoustic"]))
    results["missing_light_envair"] = ev(mask_cols(Xte, MODALITY_GROUPS["light"] + MODALITY_GROUPS["env_air"]))

    (OUT / "robustness_summary.json").write_text(json.dumps(results, indent=1))

    ref = results["full"]
    print(f"{'scenario':<26} {'f1':>7} {'prauc':>7} {'mcc':>7} {'bacc':>7} {'dF1':>7} {'dPR':>7}")
    for k, m in results.items():
        dF1 = m["f1"] - ref["f1"]
        dPR = m["pr_auc"] - ref["pr_auc"]
        print(f"{k:<26} {m['f1']:>7.4f} {m['pr_auc']:>7.4f} {m['mcc']:>7.4f} {m['balanced_acc']:>7.4f} {dF1:>+7.4f} {dPR:>+7.4f}")
    print("\nTersimpan:", OUT / "robustness_summary.json")


if __name__ == "__main__":
    main()
