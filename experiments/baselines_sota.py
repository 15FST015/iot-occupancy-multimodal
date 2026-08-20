#!/usr/bin/env python3
"""baselines_sota.py — Baseline SOTA missing-modality (fair, Dataset A).

Menjawab MAJOR ④ Jurnal-Evaluator: bandingkan proposed DL dengan metode
missing-modality standar literatur, protokol IDENTIK (split sama, eval 5 hari
stratified, lr 3e-4, early stop patience 5, threshold ditune di eval):

  B1 mean_impute   — imputasi rata-rata (train means) + MLP early-fusion
                     [pendekatan imputation-based standar]
  B2 indicator_mlp — imputasi 0 + flag presence per kanal + MLP
                     [pendekatan indicator-based (Chevalier et al., GMD-style)]
  B3 gated_fusion  — encoder per-modalitas + gate reliabilitas belajar (sigmoid)
                     + weighted sum termask + MLP head
                     [pendekatan gated/adaptive fusion (Gated Fusion, MMIN, dll)]

Evaluasi: 14 skenario robustness, metrik F1/PR-AUC/MCC/ROC.
Output: results/sota_baselines_summary.json
"""
import json
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, average_precision_score

from baselines import load_split, add_features, metrics, NUM_COLS, BIN_COLS, MODALITY_GROUPS
from proposed_dl import (build_tensors, make_scenario_tensors, predict, GROUPS,
                         CH_DIM, OUT)

torch.manual_seed(42)
np.random.seed(42)
torch.set_num_threads(4)

ALL_CH = NUM_COLS + BIN_COLS
D = 64
BATCH = 512


class MeanImputeMLP(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, 128), nn.ReLU(),
                                 nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


class IndicatorMLP(nn.Module):
    def __init__(self, n_in, n_flag):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in + n_flag, 128), nn.ReLU(),
                                 nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x, f):
        return self.net(torch.cat([x, f], dim=-1)).squeeze(-1)


class GatedFusion(nn.Module):
    """Encoder per-modalitas + gate belajar; tanpa attention/contrastive/augmentasi."""
    def __init__(self):
        super().__init__()
        self.encoders = nn.ModuleDict({
            g: nn.Sequential(nn.Linear(2 * CH_DIM[g], D), nn.ReLU(), nn.Linear(D, D))
            for g in GROUPS})
        self.gates = nn.ModuleDict({g: nn.Linear(D, 1) for g in GROUPS})
        self.head = nn.Sequential(nn.Linear(D, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, X, M):
        toks, w = [], []
        for g in GROUPS:
            e = self.encoders[g](torch.cat([X[g], M[g]], dim=-1))
            toks.append(e)
            w.append(torch.sigmoid(self.gates[g](e)))          # reliabilitas belajar
        T = torch.stack(toks, dim=1)                            # B,7,D
        W = torch.stack(w, dim=1)                               # B,7,1
        mask = torch.stack([(M[g].sum(1) > 0).float() for g in GROUPS], dim=1).unsqueeze(-1)
        W = W * mask
        rep = (T * W).sum(1) / W.sum(1).clamp(min=1e-6)
        return self.head(rep).squeeze(-1)


def train_eval(name, model, Xtr, Mtr, tr_idx, va_idx, ytr, Xva, Mva, yva,
               Xte_mats, yte, epochs=15):
    pos = int(ytr[tr_idx].sum()); neg = int(len(tr_idx) - pos)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / max(pos, 1)]))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    ds = SimpleDS({g: Xtr[g][tr_idx] for g in GROUPS},
                  {g: Mtr[g][tr_idx] for g in GROUPS}, ytr[tr_idx])
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=0)

    best_pr, best_state, patience = -1, None, 0
    curve = []
    for ep in range(epochs):
        model.train()
        tot = 0.0
        for Xb, Mb, yb in dl:
            opt.zero_grad()
            if name == "mean_impute":
                logit = model(torch.cat([Xb[g] for g in GROUPS], dim=-1))
            elif name == "indicator_mlp":
                flags = torch.cat([Mb[g] for g in GROUPS], dim=-1)
                logit = model(torch.cat([Xb[g] for g in GROUPS], dim=-1), flags)
            else:
                logit = model(Xb, Mb)
            loss = crit(logit, yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(yb)
        sva = score(model, name, Xva, Mva)
        pr = float(average_precision_score(yva, sva))
        curve.append([ep + 1, round(pr, 4)])
        if pr > best_pr:
            best_pr = pr
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        print(f"  [{name}] ep {ep+1}/{epochs} loss {tot/len(tr_idx):.4f} evalPR {pr:.4f} "
              f"(best {best_pr:.4f})", flush=True)
        if patience >= 5 and ep >= 5:
            print(f"  [{name}] early stop ep {ep+1}", flush=True)
            break
    model.load_state_dict(best_state)

    sva = score(model, name, Xva, Mva)
    best_t, best_f = 0.5, -1
    for t in np.linspace(0.05, 0.95, 91):
        f = f1_score(yva, (sva >= t).astype(int), zero_division=0)
        if f > best_f:
            best_f, best_t = f, t
    thr = round(float(best_t), 2)
    print(f"  [{name}] thr={thr} best_evalF1={best_f:.4f}", flush=True)

    results = {}
    for sc, (Xs, Ms) in Xte_mats.items():
        s = score(model, name, Xs, Ms)
        p = (s >= thr).astype(int)
        results[sc] = metrics(yte, s, p)
    return {"name": name, "threshold": thr, "best_eval_prauc": best_pr,
            "best_epoch": int(curve[-1][0]), "eval_curve": curve, "scenarios": results}


@torch.no_grad()
def score(model, name, X, M, batch=2048):
    model.eval()
    out = []
    for i in range(0, X[GROUPS[0]].shape[0], batch):
        Xb = {g: X[g][i:i + batch] for g in GROUPS}
        Mb = {g: M[g][i:i + batch] for g in GROUPS}
        if name == "mean_impute":
            z = torch.cat([Xb[g] for g in GROUPS], dim=-1)
            out.append(torch.sigmoid(model(z)))
        elif name == "indicator_mlp":
            z = torch.cat([Xb[g] for g in GROUPS], dim=-1)
            f = torch.cat([Mb[g] for g in GROUPS], dim=-1)
            out.append(torch.sigmoid(model(z, f)))
        else:
            out.append(torch.sigmoid(model(Xb, Mb)))
    return torch.cat(out).numpy()


class SimpleDS(Dataset):
    def __init__(self, X, M, y):
        self.X, self.M, self.y = X, M, torch.from_numpy(y.astype(np.float32))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return {g: self.X[g][i] for g in GROUPS}, {g: self.M[g][i] for g in GROUPS}, self.y[i]


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else "all"  # all | mean_impute | indicator_mlp | gated_fusion
    train, test = load_split()
    train = add_features(train)
    test = add_features(test)
    ytr = train["occupied"].values
    yte = test["occupied"].values

    # imputasi mean (TRAIN SAJA) utk B1: ganti NaN numerik dgn mean train
    means = train[NUM_COLS].mean()
    Xtr, Mtr = build_tensors(train)
    Xtr_mean = {g: Xtr[g].clone() for g in GROUPS}
    for g in GROUPS:
        for j, c in enumerate(MODALITY_GROUPS[g]):
            if c in NUM_COLS:
                m = Xtr_mean[g][:, j]
                Xtr_mean[g][:, j] = torch.where(torch.isnan(m),
                                                torch.tensor(float(means[c])), m)
    Xte, Mte = build_tensors(test)
    Xte_mean = {g: Xte[g].clone() for g in GROUPS}
    for g in GROUPS:
        for j, c in enumerate(MODALITY_GROUPS[g]):
            if c in NUM_COLS:
                m = Xte_mean[g][:, j]
                Xte_mean[g][:, j] = torch.where(torch.isnan(m),
                                                torch.tensor(float(means[c])), m)

    # eval 5 hari stratified (identik MAJOR ①)
    days = sorted(train["createdAt"].dt.date.unique())
    occ = train.groupby(train["createdAt"].dt.date)["occupied"].sum()
    ordered = occ.sort_values().index.tolist()
    idx = np.linspace(0, len(ordered) - 1, 5).round().astype(int)
    eval_days = set(str(ordered[i]) for i in idx)
    ev_mask = train["createdAt"].dt.date.astype(str).isin(eval_days).values
    tr_idx = np.where(~ev_mask)[0]
    va_idx = np.where(ev_mask)[0]
    yva = ytr[va_idx]
    Xva, Mva = {g: Xtr[g][va_idx] for g in GROUPS}, {g: Mtr[g][va_idx] for g in GROUPS}
    print(f"eval days: {sorted(eval_days)}", flush=True)

    scenarios = (["full"] + [f"missing_{g}" for g in GROUPS]
                 + [f"random_dropout_{int(r*100)}pct" for r in (0.1, 0.3, 0.5, 0.7)]
                 + ["missing_light_and_acoustic", "missing_light_and_env_air"])
    Xte_mats = {sc: make_scenario_tensors(test, sc) for sc in scenarios}

    models = {
        "mean_impute": lambda: MeanImputeMLP(len(ALL_CH)),
        "indicator_mlp": lambda: IndicatorMLP(len(ALL_CH), len(ALL_CH)),
        "gated_fusion": GatedFusion,
    }
    names = [only] if only != "all" else list(models)
    out = []
    for name in names:
        print(f"=== {name} ===", flush=True)
        Xtr_use = Xtr_mean if name == "mean_impute" else Xtr
        if name == "mean_impute":
            Xva_use = Xva.clone() if hasattr(Xva, 'clone') else {g: Xva[g].clone() for g in GROUPS}
            for g in GROUPS:
                for j, c in enumerate(MODALITY_GROUPS[g]):
                    if c in NUM_COLS:
                        m = Xva_use[g][:, j]
                        Xva_use[g][:, j] = torch.where(torch.isnan(m),
                                                       torch.tensor(float(means[c])), m)
        else:
            Xva_use = Xva
        if name == "mean_impute":
            # skenario: nilai yang di-mask (M asli 1 → M baru 0) diimputasi MEAN, bukan 0
            Xte_mats_use = {}
            for sc, (Xs, Ms) in Xte_mats.items():
                Xm = {g: Xs[g].clone() for g in GROUPS}
                for g in GROUPS:
                    for j, c in enumerate(MODALITY_GROUPS[g]):
                        if c in NUM_COLS:
                            masked = (Mte[g][:, j] > 0.5) & (Ms[g][:, j] < 0.5)
                            Xm[g][:, j] = torch.where(masked, torch.tensor(float(means[c])),
                                                      Xm[g][:, j])
                Xte_mats_use[sc] = (Xm, Ms)
        else:
            Xte_mats_use = Xte_mats
        res = train_eval(name, models[name](), Xtr_use, Mtr, tr_idx, va_idx, ytr,
                         Xva_use, Mva, yva, Xte_mats_use, yte)
        out.append(res)
        (OUT / f"sota_{name}.json").write_text(json.dumps(res, indent=1))

    (OUT / "sota_baselines_summary.json").write_text(json.dumps(out, indent=1))
    print("\nSELESAI → sota_baselines_summary.json")


if __name__ == "__main__":
    main()
