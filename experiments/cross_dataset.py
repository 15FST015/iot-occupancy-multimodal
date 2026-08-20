#!/usr/bin/env python3
"""cross_dataset.py — Generalisasi lintas-dataset A → HPDmobile (zero-shot).

Domain shift: smart office (A, PUCRS) → residensial (HPDmobile, 6 rumah).
Kanal umum (4): co2, humidity, temperature, light (HPD: co2eq_ppm, rh_percent,
temp_c, light_lux). Model dilatih HANYA di A, dievaluasi di HPD TANPA tuning
(threshold tetap 0.5 — zero-shot jujur; PR-AUC threshold-free sebagai utama).

Model: baseline XGB (imputasi) + missing-aware B (dropout augmentation) —
bukti konsep cepat; DL menyusul.

Output: experiments/results/cross_dataset_summary.json
"""
import json
import numpy as np
import pandas as pd
import glob
from pathlib import Path
import xgboost as xgb

from baselines import (load_split, add_features, metrics, MODALITY_GROUPS, SEED)

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)
TIME_COLS = ["hour_sin", "hour_cos", "weekend"]

COMMON = {  # kolom umum: A -> HPD
    "co2": "co2eq_ppm", "humidity": "rh_percent",
    "temperature": "temp_c", "light": "light_lux",
}
A_COLS = ["co2", "humidity", "temperature_1", "lux_1"]  # dipakai utk scaler
HPD_HUBS = {"H1": "H1_RS2_ENV", "H2": "H2_RS1_ENV"}


def load_hpd(home, hub_dir):
    env_dir = DATA / "hpdmobile" / f"{home}_ENVIRONMENTAL" / hub_dir
    dfs = []
    for f in sorted(glob.glob(str(env_dir / "*.csv"))):
        df = pd.read_csv(f, index_col=0)
        dfs.append(df)
    env = pd.concat(dfs)
    env.index = pd.to_datetime(env.index)
    env = env[~env.index.duplicated(keep="first")].sort_index()

    gt_files = sorted(glob.glob(str(DATA / "hpdmobile" / f"{home}_GROUNDTRUTH" / "*.csv")))
    gts = [pd.read_csv(f) for f in gt_files]
    gt = pd.concat(gts)
    gt["timestamp"] = pd.to_datetime(gt["timestamp"])
    gt = gt.set_index("timestamp").sort_index()
    gt = gt[~gt.index.duplicated(keep="first")]

    df = env.join(gt, how="inner")
    df = df.rename(columns={v: k for k, v in COMMON.items()})
    df = df[["co2", "humidity", "temperature", "light", "occupied"]]
    df["occupied"] = df["occupied"].astype(float)
    return df.dropna(subset=["occupied"])


def zscore_hpd(df, scaler):
    d = df.copy()
    for a, h in (("co2", "co2"), ("humidity", "humidity"),
                 ("temperature", "temperature_1"), ("light", "lux_1")):
        m, s = scaler[h]["mean"], scaler[h]["std"]
        d[a] = (d[a] - m) / s
    return d


def fit_xgb(X, y, Xva, yva, with_aug=False, seed=SEED):
    if with_aug:
        rng = np.random.default_rng(seed)
        X = X.copy()
        aug = rng.random(len(X)) < 0.30
        for i in np.where(aug)[0]:
            if i % 2 == 0:      # grup env_air: co2, humidity, temperature
                X.iloc[i, 0:3] = 0.0
            else:               # grup light
                X.iloc[i, 3] = 0.0
    pos = int(y.sum()); neg = int(len(y) - pos)
    m = xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6,
                          subsample=0.8, colsample_bytree=0.8,
                          scale_pos_weight=neg / max(pos, 1),
                          eval_metric="aucpr", early_stopping_rounds=30,
                          random_state=seed, n_jobs=-1)
    m.fit(X, y, eval_set=[(Xva, yva)], verbose=False)
    return m


def main():
    train, _ = load_split()
    train = add_features(train)
    ytr = train["occupied"].values
    scaler = json.load(open(DATA / "scaler.json"))

    days = sorted(train["createdAt"].dt.date.unique())
    evd = set(str(d) for d in days[-2:])
    ev = train["createdAt"].dt.date.astype(str).isin(evd).values

    # fitur A (4 kanal umum z-scored + waktu)
    cols4 = A_COLS + TIME_COLS
    Xa = train[cols4].fillna(0.0)
    base = fit_xgb(Xa[~ev], ytr[~ev], Xa[ev], ytr[ev])
    mb = fit_xgb(Xa[~ev], ytr[~ev], Xa[ev], ytr[ev], with_aug=True)

    # precompute A-train utk fine-tune
    Xa_ft_all = Xa[~ev]
    ytr_ft_all = ytr[~ev]

    results = {}
    for home, hub in HPD_HUBS.items():
        hpd = load_hpd(home, hub)
        hpd = zscore_hpd(hpd, scaler)
        hpd["createdAt"] = hpd.index
        hpd = add_features(hpd)
        Xh_raw = hpd[["co2", "humidity", "temperature", "light"] + TIME_COLS].fillna(0.0)
        Xh = Xh_raw.rename(columns={"temperature": "temperature_1", "light": "lux_1"})
        yh = hpd["occupied"].values

        s_base = base.predict_proba(Xh)[:, 1]
        s_mb = mb.predict_proba(Xh)[:, 1]
        # blending: rho = fraksi kanal NaN asli per baris (natural missing)
        nat = hpd[["co2", "humidity", "temperature", "light"]].isna().mean(axis=1).values
        s_blend = (1 - nat) * s_base + nat * s_mb

        # --- fine-tune: gabung A-train + HARI PERTAMA HPD, eval sisa hari ---
        days = sorted(hpd.index.normalize().unique())
        adapt_mask = hpd.index.normalize() == days[0]
        eval_mask = ~adapt_mask
        X_ft = pd.concat([Xa_ft_all, Xh[adapt_mask]], axis=0)
        y_ft = np.concatenate([ytr_ft_all, yh[adapt_mask]])
        ft = fit_xgb(X_ft, y_ft, Xa[ev], ytr[ev])
        s_ft = ft.predict_proba(Xh[eval_mask])[:, 1]

        res = {}
        for tag, s in (("base", s_base), ("missaware", s_mb), ("blend", s_blend)):
            p = (s >= 0.5).astype(int)
            res[tag] = metrics(yh, s, p)
        p_ft = (s_ft >= 0.5).astype(int)
        res["finetuned_1day"] = metrics(yh[eval_mask], s_ft, p_ft)
        res["majority_baseline"] = metrics(yh, np.zeros_like(yh), np.zeros_like(yh, dtype=int))
        results[home] = {"rows": len(yh), "eval_rows": int(eval_mask.sum()),
                         "occupied_pct": round(100 * yh.mean(), 2),
                         "natural_missing_frac": round(float(nat.mean()), 4), **res}
        print(f"=== {home}: {len(yh)} baris (fine-tune eval {eval_mask.sum()}), okupansi {results[home]['occupied_pct']}% ===")
        for tag in ("majority_baseline", "base", "missaware", "blend", "finetuned_1day"):
            m = res[tag]
            print(f"  {tag:<18} F1 {m['f1']:.4f} PR-AUC {m['pr_auc']:.4f} MCC {m['mcc']:.4f} ROC {m['roc_auc']:.4f}")

    (OUT / "cross_dataset_summary.json").write_text(json.dumps(results, indent=1))
    print("\nTersimpan:", OUT / "cross_dataset_summary.json")


if __name__ == "__main__":
    main()
