#!/usr/bin/env python3
"""adaptive_ensemble.py — Ensemble adaptif berbasis reliability (Dataset A).

Motivasi (hasil proposed_v1): baseline menang saat semua sensor sehat, model
missing-aware menang saat dropout — tidak ada model tunggal yang menang semua.

Ide: meta-model (XGBoost kecil) dilatih DENGAN dropout augmentation sehingga
melihat pasangan (s_base, s_missaware, reliability per-modalitas, waktu) dan
belajar kapan mempercayai model mana. Pada test, sistem MENGETAHUI sensor mana
yang mati (status perangkat) → reliability nyata → meta-model merouting.

Evaluasi: threshold masing-masing ditune di eval set; metrik per-row.
Output: experiments/results/adaptive_ensemble_summary.json
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
import xgboost as xgb

from baselines import (load_split, add_features, metrics, NUM_COLS, BIN_COLS,
                       MODALITY_GROUPS, SEED)
from proposed_v1 import (apply_scenario, fit_xgb, tune_threshold, ev,
                         GROUP_COLS, TIME_COLS)

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)
FEAT = NUM_COLS + BIN_COLS + TIME_COLS
IND = [f"ind_{g}" for g in GROUP_COLS]


def main():
    train, test = load_split()
    train = add_features(train)
    test = add_features(test)
    ytr = train["occupied"].values
    yte = test["occupied"].values

    days = sorted(train["createdAt"].dt.date.unique())
    evd = set(str(d) for d in days[-2:])
    ev_mask = train["createdAt"].dt.date.astype(str).isin(evd).values

    # ---- dropout augmentation (sama dgn proposed_v1 varian B) ----
    rng = np.random.default_rng(SEED)
    aug = rng.random(len(train)) < 0.30
    aug_groups = rng.choice(GROUP_COLS, size=int(aug.sum()))
    Tr = train.copy()
    rel_tr = pd.DataFrame(0.0, index=Tr.index, columns=GROUP_COLS)
    it = 0
    for i in Tr.index[aug]:
        g = aug_groups[it]; it += 1
        for c in MODALITY_GROUPS[g]:
            Tr.loc[i, c] = 0.0
        rel_tr.loc[i, g] = 1.0

    Xfull_orig = train[FEAT].fillna(0.0)   # data ASLI utk baseline
    Xfull_tr = Tr[FEAT].fillna(0.0)        # versi augmented utk pelatihan mb & meta
    Xind_tr = pd.concat([Xfull_tr, pd.DataFrame(rel_tr.values, index=Tr.index, columns=IND)], axis=1)

    # ---- dua model anggota: base dilatih data ASLI, mb pada augmented ----
    base = fit_xgb(Xfull_orig[~ev_mask], ytr[~ev_mask], Xfull_orig[ev_mask], ytr[ev_mask])
    mb = fit_xgb(Xind_tr[~ev_mask], ytr[~ev_mask], Xind_tr[ev_mask], ytr[ev_mask])

    # ---- fitur meta: skor kedua model + reliability + waktu ----
    def meta_feats(Xfull, Xind, rel_df):
        M = pd.DataFrame(index=Xfull.index)
        M["s_base"] = base.predict_proba(Xfull)[:, 1]
        M["s_missaware"] = mb.predict_proba(Xind)[:, 1]
        for g in GROUP_COLS:
            M[f"r_{g}"] = rel_df[g].values
        for c in TIME_COLS:
            M[c] = Xfull[c].values
        return M

    Mtrain = meta_feats(Xfull_tr, Xind_tr, rel_tr)
    meta = fit_xgb(Mtrain[~ev_mask], ytr[~ev_mask], Mtrain[ev_mask], ytr[ev_mask],
                   depth=3, n_est=150)

    # ---- threshold masing-masing model (eval set, data bersih utk base/mb) ----
    thr_base = tune_threshold(base, Xfull_orig[ev_mask], ytr[ev_mask])
    thr_mb = tune_threshold(mb, Xind_tr[ev_mask], ytr[ev_mask])
    thr_meta = tune_threshold(meta, Mtrain[ev_mask], ytr[ev_mask])
    print(f"threshold: base={thr_base:.2f} missaware={thr_mb:.2f} ensemble={thr_meta:.2f}")

    scenarios = (["full"]
                 + [f"missing_{g}" for g in GROUP_COLS]
                 + [f"random_dropout_{int(r*100)}pct" for r in (0.1, 0.3, 0.5, 0.7)]
                 + ["missing_light_and_acoustic", "missing_light_and_env_air"])

    results = {"base": {}, "missaware": {}, "ensemble": {}}
    for sc in scenarios:
        d, rel = apply_scenario(test, sc)
        Xf = d[FEAT].fillna(0.0)
        Xi = pd.concat([Xf, pd.DataFrame(rel.values, index=d.index, columns=IND)], axis=1)
        Mte = meta_feats(Xf, Xi, rel)
        results["base"][sc] = ev(base, Xf, yte, thr_base)
        results["missaware"][sc] = ev(mb, Xi, yte, thr_mb)
        results["ensemble"][sc] = ev(meta, Mte, yte, thr_meta)

    (OUT / "adaptive_ensemble_summary.json").write_text(json.dumps(results, indent=1))

    print(f"\n{'scenario':<26} | {'base:f1':>7} {'base:pr':>7} | {'miss:f1':>7} {'miss:pr':>7} | {'ens:f1':>7} {'ens:pr':>7}")
    for sc in scenarios:
        b, m, e = results["base"][sc], results["missaware"][sc], results["ensemble"][sc]
        print(f"{sc:<26} | {b['f1']:>7.4f} {b['pr_auc']:>7.4f} | {m['f1']:>7.4f} {m['pr_auc']:>7.4f} | {e['f1']:>7.4f} {e['pr_auc']:>7.4f}")
    print("\nTersimpan:", OUT / "adaptive_ensemble_summary.json")


if __name__ == "__main__":
    main()
