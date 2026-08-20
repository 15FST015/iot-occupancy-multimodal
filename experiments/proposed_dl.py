#!/usr/bin/env python3
"""proposed_dl.py — Proposed framework DL v2 + ablation & baseline DL (Dataset A).

Arsitektur (full):
  1. Encoder per-modalitas (MLP; input = nilai kanal z-scored + flag presence)
  2. Cross-modal attention: TransformerEncoder atas 7 token modalitas + [CLS],
     key_padding_mask (modalitas hilang tidak ikut attention)
  3. Contrastive InfoNCE (clean ↔ corrupted view) — robustness
  4. Head biner okupansi (BCE + pos_weight); loss = BCE + λ·InfoNCE
  5. Dropout augmentation saat training (30% baris, 1 grup hilang)

Varian (CLI: python proposed_dl.py <epochs> <variant>):
  full           — semua komponen
  no_contrastive — tanpa InfoNCE (augmentasi tetap)
  no_attention   — tanpa Transformer (mean pooling token termask)
  no_aug         — tanpa augmentasi training (contrastive tetap, corrupted view
                   dibentuk dari batch bersih)
  mlp_early      — baseline DL konvensional: concat semua kanal → MLP
                   (tanpa attention, tanpa contrastive, tanpa augmentasi)

Evaluasi: 14 skenario robustness; threshold ditune di eval; metrik per-row.
Output: experiments/results/proposed_dl_<variant>_summary.json
"""
import json
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, average_precision_score

from baselines import (load_split, add_features, metrics, NUM_COLS, BIN_COLS,
                       MODALITY_GROUPS, SEED)

torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_num_threads(4)

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)
GROUPS = list(MODALITY_GROUPS.keys())
CH_DIM = {g: len(MODALITY_GROUPS[g]) for g in GROUPS}
D_MODEL, NHEAD, D_FF, NLAYERS = 64, 4, 128, 2
TEMP, LAM = 0.1, 0.1
BATCH = 512
AUG_PROB = 0.30
ALL_CH = NUM_COLS + BIN_COLS  # 20 kanal utk MLP early fusion


def build_tensors(df):
    X, M = {}, {}
    for g in GROUPS:
        arr = df[MODALITY_GROUPS[g]].values.astype(np.float32)
        X[g] = torch.from_numpy(np.nan_to_num(arr, nan=0.0))
        M[g] = torch.from_numpy((~np.isnan(arr)).astype(np.float32))
    return X, M


class OccDataset(Dataset):
    def __init__(self, X, M, y):
        self.X, self.M, self.y = X, M, torch.from_numpy(y.astype(np.float32))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return {g: self.X[g][i] for g in GROUPS}, {g: self.M[g][i] for g in GROUPS}, self.y[i]


def collate(batch):
    X = {g: torch.stack([b[0][g] for b in batch]) for g in GROUPS}
    M = {g: torch.stack([b[1][g] for b in batch]) for g in GROUPS}
    y = torch.stack([b[2] for b in batch])
    return X, M, y


class ProposedDL(nn.Module):
    def __init__(self, use_attention=True):
        super().__init__()
        self.use_attention = use_attention
        self.encoders = nn.ModuleDict({
            g: nn.Sequential(nn.Linear(2 * CH_DIM[g], D_MODEL), nn.ReLU(),
                             nn.Linear(D_MODEL, D_MODEL))
            for g in GROUPS})
        self.cls_token = nn.Parameter(torch.randn(1, 1, D_MODEL) * 0.02)
        self.type_emb = nn.Parameter(torch.randn(1, len(GROUPS) + 1, D_MODEL) * 0.02)
        if use_attention:
            layer = nn.TransformerEncoderLayer(D_MODEL, NHEAD, D_FF, batch_first=True, dropout=0.1)
            self.transformer = nn.TransformerEncoder(layer, NLAYERS)
        self.head = nn.Linear(D_MODEL, 1)
        self.proj = nn.Sequential(nn.Linear(D_MODEL, 32), nn.ReLU(), nn.Linear(32, 32))

    def encode_tokens(self, X, M):
        B = X[GROUPS[0]].shape[0]
        toks, pad = [], []
        for g in GROUPS:
            inp = torch.cat([X[g], M[g]], dim=-1)
            e = self.encoders[g](inp)
            toks.append(e.unsqueeze(1))
            pad.append((M[g].sum(dim=1) < 0.5).bool())
        if self.use_attention:
            cls = self.cls_token.expand(B, -1, -1)
            toks = torch.cat([cls] + toks, dim=1)
            pad = torch.stack(pad, dim=1)
            pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool), pad], dim=1)
            toks = toks + self.type_emb
            out = self.transformer(toks, src_key_padding_mask=pad)
            return out[:, 0], pad[:, 1:]
        # mean pooling atas token modalitas yang ADA (masked mean)
        toks = torch.cat(toks, dim=1)
        pad = torch.stack(pad, dim=1)
        mask = (~pad).unsqueeze(-1).float()
        rep = (toks * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return rep, pad

    def forward(self, X, M):
        rep, _ = self.encode_tokens(X, M)
        return self.head(rep).squeeze(-1)

    def contrastive(self, z_clean, z_corr):
        zc = F.normalize(self.proj(z_clean), dim=-1)
        zk = F.normalize(self.proj(z_corr), dim=-1)
        logits = zc @ zk.T / TEMP
        labels = torch.arange(len(zc), device=zc.device)
        return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


class MLPEarly(nn.Module):
    """Baseline DL konvensional: concat 20 kanal (imputasi 0) → MLP."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(len(ALL_CH), 128), nn.ReLU(),
                                 nn.Linear(128, 64), nn.ReLU(),
                                 nn.Linear(64, 1))

    def forward(self, X, M):
        return self.net(X).squeeze(-1)


def corrupt_batch(X, M, prob=AUG_PROB, rng=None):
    B = X[GROUPS[0]].shape[0]
    Xc = {g: X[g].clone() for g in GROUPS}
    Mc = {g: M[g].clone() for g in GROUPS}
    if rng is None:
        rng = np.random.default_rng(SEED)
    do = rng.random(B) < prob
    gs = rng.choice(GROUPS, size=int(do.sum()))
    it = 0
    for i in np.where(do)[0]:
        g = gs[it]; it += 1
        Xc[g][i] = 0.0
        Mc[g][i] = 0.0
    return Xc, Mc


def make_scenario_tensors(test, scenario, seed=SEED):
    d = test.copy()
    X, M = build_tensors(d)
    rng = np.random.default_rng(seed)
    if scenario == "full":
        return X, M
    if scenario.startswith("missing_"):
        for g in GROUPS:
            if g in scenario:
                for j, c in enumerate(MODALITY_GROUPS[g]):
                    X[g][:, j] = 0.0
                    M[g][:, j] = 0.0
        return X, M
    if scenario.startswith("random_dropout_"):
        rate = int(scenario.split("_")[2].replace("pct", "")) / 100.0
        for g in GROUPS:
            numc = [c for c in MODALITY_GROUPS[g] if c in NUM_COLS]
            idx = [MODALITY_GROUPS[g].index(c) for c in numc]
            if not idx:
                continue
            m = rng.random((len(d), len(idx))) < rate
            X[g][:, idx] = torch.where(torch.from_numpy(m), torch.zeros_like(X[g][:, idx]), X[g][:, idx])
            M[g][:, idx] = torch.where(torch.from_numpy(m), torch.zeros_like(M[g][:, idx]), M[g][:, idx])
        return X, M
    raise ValueError(scenario)


@torch.no_grad()
def predict(model, X, M, batch=2048):
    model.eval()
    scores = []
    for i in range(0, X[GROUPS[0]].shape[0], batch):
        Xb = {g: X[g][i:i + batch] for g in GROUPS}
        Mb = {g: M[g][i:i + batch] for g in GROUPS}
        if isinstance(model, MLPEarly):
            xcat = torch.cat([Xb[g] for g in GROUPS], dim=-1)
            scores.append(torch.sigmoid(model(xcat, Mb)))
        else:
            scores.append(torch.sigmoid(model(Xb, Mb)))
    return torch.cat(scores).numpy()


def eval_scores(y, s, thr):
    p = (s >= thr).astype(int)
    return metrics(y, s, p)


def main():
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    variant = sys.argv[2] if len(sys.argv) > 2 else "full"
    lr = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-3
    wd = float(sys.argv[4]) if len(sys.argv) > 4 else 1e-5
    n_eval_days = int(sys.argv[5]) if len(sys.argv) > 5 else 2  # 2 = perilaku lama (2 hari terakhir); >=3 = stratified
    use_attention = variant not in ("no_attention", "mlp_early")
    use_contrastive = variant not in ("no_contrastive", "mlp_early")
    aug_prob = 0.0 if variant in ("no_aug", "mlp_early") else AUG_PROB
    print(f"[{variant}] epochs={epochs} lr={lr} wd={wd} eval_days={n_eval_days} "
          f"attention={use_attention} contrastive={use_contrastive} aug={aug_prob}", flush=True)

    train, test = load_split()
    train = add_features(train)
    test = add_features(test)
    ytr = train["occupied"].values
    yte = test["occupied"].values

    # --- eval set day-level: (a) lama: 2 hari TERAKHIR; (b) baru: N hari stratified okupansi ---
    days = sorted(train["createdAt"].dt.date.unique())
    if n_eval_days >= 3:
        occ = train.groupby(train["createdAt"].dt.date)["occupied"].sum()
        ordered = occ.sort_values().index.tolist()
        idx = np.linspace(0, len(ordered) - 1, n_eval_days).round().astype(int)
        eval_days = set(str(ordered[i]) for i in idx)
    else:
        eval_days = set(str(d) for d in days[-n_eval_days:])
    ev_mask = train["createdAt"].dt.date.astype(str).isin(eval_days).values
    print(f"  eval days ({len(eval_days)}): {sorted(eval_days)}", flush=True)

    Xtr, Mtr = build_tensors(train)
    if aug_prob > 0:
        Xtr, Mtr = corrupt_batch(Xtr, Mtr, prob=aug_prob)

    tr_idx = np.where(~ev_mask)[0]
    va_idx = np.where(ev_mask)[0]
    ds = OccDataset({g: Xtr[g][tr_idx] for g in GROUPS},
                    {g: Mtr[g][tr_idx] for g in GROUPS}, ytr[tr_idx])
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True, collate_fn=collate, num_workers=0)
    Xva = {g: Xtr[g][va_idx] for g in GROUPS}
    Mva = {g: Mtr[g][va_idx] for g in GROUPS}
    yva = ytr[va_idx]

    model = MLPEarly() if variant == "mlp_early" else ProposedDL(use_attention)
    if variant == "mlp_early":
        xcat_tr = torch.cat([Xtr[g] for g in GROUPS], dim=-1)
    pos = int(ytr[tr_idx].sum()); neg = int(len(tr_idx) - pos)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / max(pos, 1)]))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    rng = np.random.default_rng(SEED + 1)

    best_pr, best_state, best_epoch, patience = -1, None, 0, 0
    eval_curve = []
    for ep in range(epochs):
        model.train()
        tot = 0.0
        for Xb, Mb, yb in dl:
            opt.zero_grad()
            if variant == "mlp_early":
                xcat = torch.cat([Xb[g] for g in GROUPS], dim=-1)
                logit = model(xcat, Mb)
                loss = crit(logit, yb)
            else:
                rep_c, _ = model.encode_tokens(Xb, Mb)
                logit = model.head(rep_c).squeeze(-1)
                loss = crit(logit, yb)
                if use_contrastive:
                    Xc, Mc = corrupt_batch(Xb, Mb, prob=0.5, rng=rng)
                    rep_k, _ = model.encode_tokens(Xc, Mc)
                    loss = loss + LAM * model.contrastive(rep_c, rep_k)
            loss.backward()
            opt.step()
            tot += loss.item() * len(yb)
        sva = predict(model, Xva, Mva)
        pr = float(average_precision_score(yva, sva))
        eval_curve.append([ep + 1, round(pr, 4)])
        if pr > best_pr:
            best_pr = pr
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = ep + 1
            patience = 0
        else:
            patience += 1
        print(f"  epoch {ep+1}/{epochs} loss {tot/len(tr_idx):.4f} eval PR-AUC {pr:.4f} "
              f"(best {best_pr:.4f} @ ep {best_epoch})", flush=True)
        if patience >= 5 and ep >= 5:
            print(f"  early stop di epoch {ep+1} (patience 5)", flush=True)
            break

    model.load_state_dict(best_state)
    torch.save({"state": best_state, "variant": variant, "use_attention": use_attention,
                "arch": "mlp" if variant == "mlp_early" else "proposed"},
               OUT / f"model_{variant}.pt")
    sva = predict(model, Xva, Mva)
    cand = np.linspace(0.05, 0.95, 91)
    best_t, best_f = 0.5, -1
    for t in cand:
        f = f1_score(yva, (sva >= t).astype(int), zero_division=0)
        if f > best_f:
            best_f, best_t = f, t
    thr = round(float(best_t), 2)
    print(f"  threshold={thr} (best eval F1 {best_f:.4f})", flush=True)

    scenarios = (["full"]
                 + [f"missing_{g}" for g in GROUPS]
                 + [f"random_dropout_{int(r*100)}pct" for r in (0.1, 0.3, 0.5, 0.7)]
                 + ["missing_light_and_acoustic", "missing_light_and_env_air"])

    results = {}
    for sc in scenarios:
        Xs, Ms = make_scenario_tensors(test, sc)
        s = predict(model, Xs, Ms)
        results[sc] = eval_scores(yte, s, thr)

    out = {"variant": variant, "epochs": epochs, "best_epoch": best_epoch, "lr": lr,
           "weight_decay": wd, "n_eval_days": n_eval_days, "eval_days": sorted(eval_days),
           "eval_curve": eval_curve, "threshold": thr,
           "best_eval_prauc": best_pr, "scenarios": results}
    (OUT / f"proposed_dl_{variant}_summary.json").write_text(json.dumps(out, indent=1))
    print(f"[{variant}] selesai → proposed_dl_{variant}_summary.json")


if __name__ == "__main__":
    main()
