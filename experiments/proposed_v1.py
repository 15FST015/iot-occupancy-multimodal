#!/usr/bin/env python3
"""proposed_v1.py — Proposed framework v1 (tree-based, missing-aware) — REVISI.

Dua varian:
  A) stacking experts: 7 XGBoost per-modalitas → score; meta-model atas
     [score×7, reliability×7 (per-row), agregat kanal per grup (mean/std), time]
  B) missing-aware full: XGBoost full-kanal + 7 kolom missing-indicator
     (dilatih dgn indikator → belajar kompensasi saat dropout)

Keduanya dievaluasi pada skenario robustness yang sama dengan baseline
(robustness.py). Reliability per-row (fraksi kanal termask per baris).

Output: experiments/results/proposed_v1_summary.json
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
GROUP_COLS = list(MODALITY_GROUPS.keys())


def load_prepared():
    train, test = load_split()
    train = add_features(train)
    test = add_features(test)
    return train, test


def fit_expert(gname, df, ev_mask):
    cols = MODALITY_GROUPS[gname] + TIME_COLS
    X = df[cols].fillna(0.0)
    y = df["occupied"].values
    pos = int(y[~ev_mask].sum()); neg = int((~ev_mask).sum() - pos)
    m = xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=5,
                          subsample=0.8, colsample_bytree=0.8,
                          scale_pos_weight=neg / max(pos, 1),
                          eval_metric="aucpr", early_stopping_rounds=30,
                          random_state=SEED, n_jobs=-1)
    m.fit(X[~ev_mask], y[~ev_mask], eval_set=[(X[ev_mask], y[ev_mask])], verbose=False)
    return m


def apply_scenario(df, scenario, seed=SEED):
    """Terapkan skenario masking; kembalikan (df_termodifikasi, rel per-grup per-row).
    Semua kanal (numerik & biner) di-mask → 0 + indikator 1 (0 pd biner = state valid,
    tapi indikator membedakan 'mati' vs 'hilang')."""
    d = df.copy()
    rel = pd.DataFrame(0.0, index=d.index, columns=GROUP_COLS)
    if scenario == "full":
        return d, rel
    if scenario.startswith("missing_"):
        found = [g for g in GROUP_COLS if g in scenario]
        for g in found:
            for c in MODALITY_GROUPS[g]:
                d[c] = 0.0
            rel[g] = 1.0
        return d, rel
    if scenario.startswith("random_dropout_"):
        rate = int(scenario.split("_")[2].replace("pct", "")) / 100.0
        rng = np.random.default_rng(seed)
        for g in GROUP_COLS:
            cols = MODALITY_GROUPS[g]
            # hanya kanal numerik yg dimask acak (biner 0 = state valid, tidak acak)
            numc = [c for c in cols if c in NUM_COLS]
            if not numc:
                continue
            m = rng.random((len(d), len(numc))) < rate
            d[numc] = d[numc].where(~m, 0.0)
            rel[g] = m.mean(axis=1)
        return d, rel
    raise ValueError(scenario)


def group_aggr(df):
    A = pd.DataFrame(index=df.index)
    for g in GROUP_COLS:
        cols = [c for c in MODALITY_GROUPS[g] if c in NUM_COLS]
        if cols:
            A[f"m_{g}"] = df[cols].mean(axis=1)
            A[f"s_{g}"] = df[cols].std(axis=1)
    return A


def fit_xgb(X, y, Xva, yva, depth=6, n_est=300, seed=SEED):
    pos = int(y.sum()); neg = int(len(y) - pos)
    m = xgb.XGBClassifier(n_estimators=n_est, learning_rate=0.05, max_depth=depth,
                          subsample=0.8, colsample_bytree=0.8,
                          scale_pos_weight=neg / max(pos, 1),
                          eval_metric="aucpr", early_stopping_rounds=30,
                          random_state=seed, n_jobs=-1)
    m.fit(X, y, eval_set=[(Xva, yva)], verbose=False)
    return m


def tune_threshold(model, Xva, yva):
    """Threshold optimal (max F1) pada eval set."""
    s = model.predict_proba(Xva)[:, 1]
    thr = np.linspace(0.05, 0.95, 91)
    best_t, best_f1 = 0.5, -1
    from sklearn.metrics import f1_score
    for t in thr:
        f = f1_score(yva, (s >= t).astype(int), zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t
    return float(best_t)


def ev(model, Xte, yte, thr=0.5):
    s = model.predict_proba(Xte)[:, 1]
    p = (s >= thr).astype(int)
    return metrics(yte, s, p)


def main():
    train, test = load_prepared()
    ytr = train["occupied"].values
    yte = test["occupied"].values

    days = sorted(train["createdAt"].dt.date.unique())
    eval_days = set(str(d) for d in days[-2:])
    ev_mask = train["createdAt"].dt.date.astype(str).isin(eval_days).values

    scenarios = (["full"]
                 + [f"missing_{g}" for g in GROUP_COLS]
                 + [f"random_dropout_{int(r*100)}pct" for r in (0.1, 0.3, 0.5, 0.7)]
                 + ["missing_light_and_acoustic", "missing_light_and_env_air"])

    # ---------- VARIAN A: stacking experts + meta ----------
    experts = {g: fit_expert(g, train, ev_mask) for g in GROUP_COLS}

    Str = pd.DataFrame({g: experts[g].predict_proba(
        train[MODALITY_GROUPS[g] + TIME_COLS].fillna(0.0))[:, 1] for g in GROUP_COLS},
        index=train.index)
    Agr = group_aggr(train)
    Mtr = pd.concat([Str, pd.DataFrame(0.0, index=train.index, columns=[f"r_{g}" for g in GROUP_COLS]), Agr, train[TIME_COLS]], axis=1)
    meta = fit_xgb(Mtr[~ev_mask], ytr[~ev_mask], Mtr[ev_mask], ytr[ev_mask], depth=4, n_est=200)

    # ---------- VARIAN B: missing-aware full (dropout augmentation) ----------
    ind_cols = [f"ind_{g}" for g in GROUP_COLS]

    def build_ind(df, rel):
        I = pd.DataFrame(0.0, index=df.index, columns=ind_cols)
        for g in GROUP_COLS:
            I[f"ind_{g}"] = rel[g].values
        return I

    # Training: augmentasi dropout — 30% baris, 1 grup random di-mask (kanal=0, ind=1)
    rng = np.random.default_rng(SEED)
    aug = rng.random(len(train)) < 0.30
    aug_groups = rng.choice(GROUP_COLS, size=int(aug.sum()))
    Tr = train.copy()
    rel_tr = pd.DataFrame(0.0, index=Tr.index, columns=GROUP_COLS)
    it = 0
    for i in Tr.index[aug]:
        g = aug_groups[it]; it += 1
        for c in MODALITY_GROUPS[g]:  # semua kanal (numerik & biner) → 0
            Tr.loc[i, c] = 0.0
        rel_tr.loc[i, g] = 1.0
    Btr = pd.concat([Tr[NUM_COLS + BIN_COLS + TIME_COLS].fillna(0.0), build_ind(Tr, rel_tr)], axis=1)
    maf = fit_xgb(Btr[~ev_mask], ytr[~ev_mask], Btr[ev_mask], ytr[ev_mask], depth=6, n_est=300)

    # ---------- baseline full (threshold-tuned, fair comparison) ----------
    Bbase = pd.concat([train[NUM_COLS + BIN_COLS + TIME_COLS].fillna(0.0)], axis=1)
    base = fit_xgb(Bbase[~ev_mask], ytr[~ev_mask], Bbase[ev_mask], ytr[ev_mask], depth=6, n_est=300)

    results = {"variantA": {}, "variantB": {}, "baseline_tuned": {}}
    # threshold masing-masing model di eval set
    thrA = tune_threshold(meta, Mtr[ev_mask], ytr[ev_mask])
    thrB = tune_threshold(maf, Btr[ev_mask], ytr[ev_mask])
    thrBase = tune_threshold(base, Bbase[ev_mask], ytr[ev_mask])
    print(f"threshold: A={thrA:.2f} B={thrB:.2f} base={thrBase:.2f}")

    for sc in scenarios:
        d, rel = apply_scenario(test, sc)
        # A
        S = pd.DataFrame({g: experts[g].predict_proba(d[MODALITY_GROUPS[g] + TIME_COLS].fillna(0.0))[:, 1]
                          for g in GROUP_COLS}, index=d.index)
        Mte = pd.concat([S, pd.DataFrame({f"r_{g}": rel[g].values for g in GROUP_COLS}, index=d.index),
                         group_aggr(d), d[TIME_COLS]], axis=1)
        results["variantA"][sc] = ev(meta, Mte, yte, thrA)
        # B
        Bte = pd.concat([d[NUM_COLS + BIN_COLS + TIME_COLS].fillna(0.0), build_ind(d, rel)], axis=1)
        results["variantB"][sc] = ev(maf, Bte, yte, thrB)
        # baseline
        BteB = d[NUM_COLS + BIN_COLS + TIME_COLS].fillna(0.0)
        results["baseline_tuned"][sc] = ev(base, BteB, yte, thrBase)

    (OUT / "proposed_v1_summary.json").write_text(json.dumps(results, indent=1))

    print(f"{'scenario':<26} | {'A:f1':>7} {'A:prauc':>7} | {'B:f1':>7} {'B:prauc':>7} | {'base:f1':>7} {'base:prauc':>7}")
    for sc in scenarios:
        a, b, bt = results["variantA"][sc], results["variantB"][sc], results["baseline_tuned"][sc]
        print(f"{sc:<26} | {a['f1']:>7.4f} {a['pr_auc']:>7.4f} | {b['f1']:>7.4f} {b['pr_auc']:>7.4f} | {bt['f1']:>7.4f} {bt['pr_auc']:>7.4f}")
    print("\nTersimpan:", OUT / "proposed_v1_summary.json")


if __name__ == "__main__":
    main()
