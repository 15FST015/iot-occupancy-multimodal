#!/usr/bin/env python3
"""ci_cluster_bootstrap.py — CI 95% cluster bootstrap PER-HARI (Dataset A).

Koreksi atas ci_bootstrap.py (per-baris, kena autokorelasi temporal).
Unit resampling = HARI (31 hari test? tidak — 6 hari test). Test hanya 6 hari
→ bootstrap per-hari dari 6 hari: sampel 6 hari dgn pengembalian, gabung barisnya.

Skenario: full + 4 skenario failure utama (light, env_air, random70, light+env_air).
Skor dihitung SEKALI per skenario; bootstrap hanya resample indeks hari.

Output: experiments/results/ci_cluster_full.json
"""
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.metrics import f1_score, average_precision_score, roc_auc_score, matthews_corrcoef

from baselines import load_split, add_features
from proposed_dl import ProposedDL, build_tensors, predict, make_scenario_tensors, GROUPS, OUT

torch.set_num_threads(4)
N_BOOT = 1000
SEED = 42
SCENARIOS = ["full", "missing_light", "missing_env_air", "random_dropout_70pct",
             "missing_light_and_env_air"]


def main():
    _, test = load_split()
    test = add_features(test)
    yte = test["occupied"].values
    # unit resampling: blok 1 JAM (kompromi — 6 hari test terlalu sedikit utk level-hari:
    # 3 hari tanpa okupansi → CI [0, 0.97] tidak informatif; blok 1 jam ≈ 1030 unit)
    blocks = test["createdAt"].dt.floor("1h").astype(str).values
    block_list = sorted(set(blocks))
    block_idx = {b: np.where(blocks == b)[0] for b in block_list}
    print(f"blok 1-jam test ({len(block_list)}): {block_list[0]} .. {block_list[-1]}", flush=True)

    ckpt = torch.load(OUT / "model_full.pt", weights_only=False)
    model = ProposedDL(True)
    model.load_state_dict(ckpt["state"])
    thr = json.load(open(OUT / "proposed_dl_full_summary.json"))["threshold"]
    print(f"threshold: {thr}", flush=True)

    rng = np.random.default_rng(SEED)
    results = {}
    for sc in SCENARIOS:
        X, M = make_scenario_tensors(test, sc)
        s = predict(model, X, M)
        p = (s >= thr).astype(int)

        def met(idx):
            y, sp, pp = yte[idx], s[idx], p[idx]
            return {
                "f1": f1_score(y, pp, zero_division=0),
                "pr_auc": average_precision_score(y, sp),
                "roc_auc": roc_auc_score(y, sp),
                "mcc": matthews_corrcoef(y, pp),
            }

        boots = []
        for _ in range(N_BOOT):
            sel = [block_list[i] for i in rng.integers(0, len(block_list), len(block_list))]
            idx = np.concatenate([block_idx[b] for b in sel])
            boots.append(met(idx))
        B = pd.DataFrame(boots)
        all_idx = np.arange(len(yte))
        point = met(all_idx)
        ci = {}
        for k in point:
            lo, hi = np.percentile(B[k], [2.5, 97.5])
            ci[k] = {"point": round(float(point[k]), 4),
                     "ci95": [round(float(lo), 4), round(float(hi), 4)]}
        results[sc] = ci
        print(f"{sc:<26} F1 {ci['f1']['point']} CI{ci['f1']['ci95']} | "
              f"PR {ci['pr_auc']['point']} CI{ci['pr_auc']['ci95']} | "
              f"MCC {ci['mcc']['point']} CI{ci['mcc']['ci95']}", flush=True)

    (OUT / "ci_cluster_full.json").write_text(json.dumps({"threshold": thr, **results}, indent=1))
    print("\nTersimpan:", OUT / "ci_cluster_full.json")


if __name__ == "__main__":
    main()
