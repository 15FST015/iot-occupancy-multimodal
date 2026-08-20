#!/usr/bin/env python3
"""blend_ensemble.py — Reliability-weighted blending (Dataset A).

score_final = (1-ρ)·s_base + ρ·s_missaware
  ρ = fraksi kanal NUMERIK termask per baris (0..1); 0 = semua sensor sehat.

Tanpa meta-model (tidak overfit, interpretable, deployable: sistem IoT tahu
status sensor). Threshold tunggal ditune di eval set.

Output: experiments/results/blend_ensemble_summary.json
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

from baselines import (load_split, add_features, metrics, NUM_COLS, BIN_COLS,
                       MODALITY_GROUPS, SEED)
from proposed_v1 import (apply_scenario, fit_xgb, tune_threshold,
                         GROUP_COLS, TIME_COLS)

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)
FEAT = NUM_COLS + BIN_COLS + TIME_COLS
IND = [f"ind_{g}" for g in GROUP_COLS]
N_NUM = len(NUM_COLS)


def main():
    train, test = load_split()
    train = add_features(train)
    test = add_features(test)
    ytr = train["occupied"].values
    yte = test["occupied"].values

    days = sorted(train["createdAt"].dt.date.unique())
    evd = set(str(d) for d in days[-2:])
    ev_mask = train["createdAt"].dt.date.astype(str).isin(evd).values

    # augmentasi (sama dgn proposed_v1 varian B)
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

    Xfull_orig = train[FEAT].fillna(0.0)
    Xfull_tr = Tr[FEAT].fillna(0.0)
    Xind_tr = pd.concat([Xfull_tr, pd.DataFrame(rel_tr.values, index=Tr.index, columns=IND)], axis=1)

    base = fit_xgb(Xfull_orig[~ev_mask], ytr[~ev_mask], Xfull_orig[ev_mask], ytr[ev_mask])
    mb = fit_xgb(Xind_tr[~ev_mask], ytr[~ev_mask], Xind_tr[ev_mask], ytr[ev_mask])

    # rho per baris: fraksi kanal numerik termask
    def rho_of(rel_df):
        # rel_df: group -> fraksi; bobotkan dgn jumlah kanal numerik per grup
        w = {g: len([c for c in MODALITY_GROUPS[g] if c in NUM_COLS]) for g in GROUP_COLS}
        rho = sum(rel_df[g].values * w[g] for g in GROUP_COLS) / N_NUM
        return np.clip(rho, 0.0, 1.0)

    def blend_score(Xf, Xi, rel_df):
        sb = base.predict_proba(Xf)[:, 1]
        sm = mb.predict_proba(Xi)[:, 1]
        rho = rho_of(rel_df)
        return (1 - rho) * sb + rho * sm

    # threshold di eval (data bersih; rho=0)
    s_eval = blend_score(Xfull_orig[ev_mask], Xind_tr[ev_mask], rel_tr[ev_mask])
    # tune threshold manual
    from sklearn.metrics import f1_score as f1s
    cand = np.linspace(0.05, 0.95, 91)
    best_t, best_f = 0.5, -1
    for t in cand:
        f = f1s(ytr[ev_mask], (s_eval >= t).astype(int), zero_division=0)
        if f > best_f:
            best_t, best_f = t, f
    thr = float(best_t)
    print(f"threshold blend={thr:.2f}")

    scenarios = (["full"]
                 + [f"missing_{g}" for g in GROUP_COLS]
                 + [f"random_dropout_{int(r*100)}pct" for r in (0.1, 0.3, 0.5, 0.7)]
                 + ["missing_light_and_acoustic", "missing_light_and_env_air"])

    results = {}
    for sc in scenarios:
        d, rel = apply_scenario(test, sc)
        Xf = d[FEAT].fillna(0.0)
        Xi = pd.concat([Xf, pd.DataFrame(rel.values, index=d.index, columns=IND)], axis=1)
        s = blend_score(Xf, Xi, rel)
        p = (s >= thr).astype(int)
        results[sc] = metrics(yte, s, p)

    (OUT / "blend_ensemble_summary.json").write_text(json.dumps(results, indent=1))

    ref = json.load(open(OUT / "adaptive_ensemble_summary.json"))
    print(f"\n{'scenario':<26} | {'blend:f1':>8} {'blend:pr':>8} | {'base:f1':>7} {'base:pr':>7} | {'miss:f1':>7} {'miss:pr':>7}")
    for sc in scenarios:
        b, m = ref["base"][sc], ref["missaware"][sc]
        e = results[sc]
        print(f"{sc:<26} | {e['f1']:>8.4f} {e['pr_auc']:>8.4f} | {b['f1']:>7.4f} {b['pr_auc']:>7.4f} | {m['f1']:>7.4f} {m['pr_auc']:>7.4f}")
    print("\nTersimpan:", OUT / "blend_ensemble_summary.json")


if __name__ == "__main__":
    main()
