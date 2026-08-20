#!/usr/bin/env python3
"""ci_bootstrap.py — CI 95% (bootstrap) + kalibrasi untuk model DL (Dataset A).

Butuh file model_<variant>.pt (dihasilkan proposed_dl.py versi terbaru).
Metrik: F1, PR-AUC, MCC, ROC-AUC (threshold ditune di eval — disimpan di
proposed_dl_<variant>_summary.json).
Kalibrasi: ECE (10 bin) + reliability plot ringkas (bin → frekuensi aktual).

Output: experiments/results/ci_bootstrap_<variant>.json
"""
import json
import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.metrics import (f1_score, average_precision_score, roc_auc_score,
                             matthews_corrcoef)

from baselines import load_split, add_features, NUM_COLS, BIN_COLS, MODALITY_GROUPS
from proposed_dl import (ProposedDL, MLPEarly, build_tensors, predict, GROUPS,
                         ALL_CH, OUT)

torch.set_num_threads(4)


def main():
    variant = sys.argv[1] if len(sys.argv) > 1 else "full"
    n_boot = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    seed = 42

    _, test = load_split()
    test = add_features(test)
    yte = test["occupied"].values
    X, M = build_tensors(test)

    ckpt = torch.load(OUT / f"model_{variant}.pt", weights_only=False)
    if ckpt["arch"] == "mlp":
        model = MLPEarly()
    else:
        model = ProposedDL(ckpt["use_attention"])
    model.load_state_dict(ckpt["state"])
    s = predict(model, X, M)

    thr = json.load(open(OUT / f"proposed_dl_{variant}_summary.json"))["threshold"]
    p = (s >= thr).astype(int)

    def met(sb, pb):
        return {
            "f1": f1_score(yte[sb], pb, zero_division=0),
            "pr_auc": average_precision_score(yte[sb], s[sb]),
            "roc_auc": roc_auc_score(yte[sb], s[sb]),
            "mcc": matthews_corrcoef(yte[sb], pb),
        }

    rng = np.random.default_rng(seed)
    idx = np.arange(len(yte))
    boots = []
    for _ in range(n_boot):
        b = rng.choice(idx, size=len(idx), replace=True)
        boots.append(met(b, p[b]))
    B = pd.DataFrame(boots)

    point = met(idx, p)
    ci = {}
    for k in point:
        lo, hi = np.percentile(B[k], [2.5, 97.5])
        ci[k] = {"point": round(point[k], 4), "ci95": [round(float(lo), 4), round(float(hi), 4)]}

    # kalibrasi: ECE 10-bin
    bins = np.linspace(0, 1, 11)
    bin_id = np.clip(np.digitize(s, bins) - 1, 0, 9)
    ece, rows = 0.0, []
    for b in range(10):
        m = bin_id == b
        if m.sum() == 0:
            continue
        conf = s[m].mean()
        freq = yte[m].mean()
        ece += (m.sum() / len(yte)) * abs(conf - freq)
        rows.append({"bin": f"{bins[b]:.1f}-{bins[b+1]:.1f}", "n": int(m.sum()),
                     "conf": round(float(conf), 3), "freq": round(float(freq), 3)})

    out = {"variant": variant, "threshold": thr, "n_test": len(yte),
           "metrics_ci": ci, "ece": round(float(ece), 4), "reliability": rows}
    (OUT / f"ci_bootstrap_{variant}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
