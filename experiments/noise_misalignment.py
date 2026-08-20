#!/usr/bin/env python3
"""noise_misalignment.py — Eksperimen noise, drift & temporal misalignment (Dataset A).

Model: ProposedDL full (model_full.pt) — threshold 0.75 (dari CI run).
Skenario tambahan (protokol RESEARCH-GAP-MATRIX Bab 25):
  - Gaussian noise: σ = 0.1/0.3/0.5 (data z-scored → σ relatif)
  - Impulse noise : 1%/5% nilai diganti spike ±(2-5)σ
  - Sensor drift  : ramp linier +0.5σ/+1σ selama periode test
  - Misalignment  : shift kanal numerik +1/+5/+10/+30 baris (10s-300s), nilai lama
    diisi 0 (sensor "hidup" tapi basi); M tetap 1
Plus: agregasi blok 5-menit untuk skenario full (koreksi autokorelasi).

Output: experiments/results/noise_misalignment_summary.json
"""
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path

from baselines import (load_split, add_features, metrics, NUM_COLS,
                       MODALITY_GROUPS)
from proposed_dl import (ProposedDL, build_tensors, predict, GROUPS, OUT)

torch.set_num_threads(4)
THR = 0.75


def main():
    _, test = load_split()
    test = add_features(test)
    yte = test["occupied"].values
    X, M = build_tensors(test)

    ckpt = torch.load(OUT / "model_full.pt", weights_only=False)
    model = ProposedDL(True)
    model.load_state_dict(ckpt["state"])

    results = {}

    def ev(tag, Xs):
        s = predict(model, Xs, M)
        p = (s >= THR).astype(int)
        results[tag] = metrics(yte, s, p)
        print(f"{tag:<24} F1 {results[tag]['f1']:.4f} PR-AUC {results[tag]['pr_auc']:.4f} "
              f"MCC {results[tag]['mcc']:.4f} ROC {results[tag]['roc_auc']:.4f}", flush=True)

    ev("full_clean", X)

    # ---- Gaussian noise ----
    rng = np.random.default_rng(42)
    for pct in (0.1, 0.3, 0.5):
        Xs = {g: X[g].clone() for g in GROUPS}
        for g in GROUPS:
            numc = [c for c in MODALITY_GROUPS[g] if c in NUM_COLS]
            idx = [MODALITY_GROUPS[g].index(c) for c in numc]
            if idx:
                Xs[g][:, idx] = Xs[g][:, idx] + torch.from_numpy(
                    rng.normal(0, pct, Xs[g][:, idx].shape).astype(np.float32))
        ev(f"gaussian_{int(pct*100)}pct", Xs)

    # ---- Impulse noise ----
    for rate in (0.01, 0.05):
        Xs = {g: X[g].clone() for g in GROUPS}
        for g in GROUPS:
            numc = [c for c in MODALITY_GROUPS[g] if c in NUM_COLS]
            idx = [MODALITY_GROUPS[g].index(c) for c in numc]
            if not idx:
                continue
            sub = Xs[g][:, idx]
            m = rng.random(sub.shape) < rate
            spikes = rng.choice([-1, 1], sub.shape) * rng.uniform(2, 5, sub.shape)
            Xs[g][:, idx] = torch.where(torch.from_numpy(m), torch.from_numpy(spikes).float(), sub)
        ev(f"impulse_{int(rate*100)}pct", Xs)

    # ---- Sensor drift (ramp linier) ----
    n = len(yte)
    for pct in (0.5, 1.0):
        Xs = {g: X[g].clone() for g in GROUPS}
        ramp = np.linspace(0, pct, n).astype(np.float32)
        for g in GROUPS:
            numc = [c for c in MODALITY_GROUPS[g] if c in NUM_COLS]
            idx = [MODALITY_GROUPS[g].index(c) for c in numc]
            if idx:
                Xs[g][:, idx] = Xs[g][:, idx] + torch.from_numpy(ramp[:, None])
        ev(f"drift_{pct}sigma", Xs)

    # ---- Temporal misalignment (shift baris) ----
    for k in (1, 5, 10, 30):
        Xs = {g: X[g].clone() for g in GROUPS}
        for g in GROUPS:
            numc = [c for c in MODALITY_GROUPS[g] if c in NUM_COLS]
            idx = [MODALITY_GROUPS[g].index(c) for c in numc]
            if idx:
                shifted = torch.empty_like(Xs[g][:, idx])
                shifted[:k] = 0.0
                shifted[k:] = Xs[g][:-k, idx]
                Xs[g][:, idx] = shifted
        ev(f"misalign_{k*10}s", Xs)

    # ---- Blok 5 menit utk full_clean ----
    d = test.copy()
    s = predict(model, X, M)
    d["y_pred"] = (s >= THR).astype(int)
    d["block"] = d["createdAt"].dt.floor("5min")
    g = d.groupby("block").agg(
        y_true=("occupied", lambda s_: s_.mode().iloc[0]),
        y_pred=("y_pred", lambda s_: s_.mode().iloc[0]))
    results["full_block5min"] = metrics(g["y_true"], g["y_pred"], g["y_pred"])
    print(f"{'full_block5min':<24} F1 {results['full_block5min']['f1']:.4f} "
          f"MCC {results['full_block5min']['mcc']:.4f} bacc {results['full_block5min']['balanced_acc']:.4f}")

    (OUT / "noise_misalignment_summary.json").write_text(json.dumps(results, indent=1))
    print("\nTersimpan:", OUT / "noise_misalignment_summary.json")


if __name__ == "__main__":
    main()
