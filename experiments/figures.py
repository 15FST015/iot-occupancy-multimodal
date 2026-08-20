#!/usr/bin/env python3
"""figures.py — Generate figure PNG untuk naskah (dari JSON hasil terverifikasi).

Fig 1: arsitektur proposed (encoder per-modalitas → masked cross-modal attention
       → CLS → head; InfoNCE; augmentation) — diagram blok
Fig 2: kurva training (eval PR-AUC per epoch, proposed_dl_full_summary.json)
Fig 3: reliability diagram (ci_bootstrap_full.json — conf vs freq per bin)
Fig 4: PR-AUC per skenario — baseline vs SOTA (mean_impute/indicator/gated) vs PROPOSED

Output: figures/fig1_arsitektur.png, fig2_kurva.png, fig3_reliability.png, fig4_pr_auc.png
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)
R = HERE / "results"
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "figure.dpi": 150})

# ---------- Fig 1: arsitektur ----------
fig, ax = plt.subplots(figsize=(9, 5.2))
ax.axis("off")
groups = ["env_air\n(CO2,RH,Temp)", "light\n(Lux1,Lux2)", "acoustic\n(Sound)",
          "device_state\n(Lamp,Switch)", "elec_socket", "elec_server", "elec_pfsense"]
y0, y1 = 0.72, 0.30
for i, g in enumerate(groups):
    y = y0 - i * (y0 - y1) / 6
    ax.add_patch(plt.Rectangle((0.02, y - 0.055), 0.16, 0.11, fc="#e8f0fe", ec="#1a73e8", lw=1.2))
    ax.text(0.10, y, g, ha="center", va="center", fontsize=6.5)
    ax.annotate("", xy=(0.24, y), xytext=(0.18, y), arrowprops=dict(arrowstyle="-", color="#888"))
ax.text(0.24, 0.52, "Encoder MLP\n(z-score + presence flag)\nD=64 per modalitas", ha="center", va="center", fontsize=7,
        bbox=dict(boxstyle="round,pad=0.3", fc="#fef7e0", ec="#f9ab00"))
ax.add_patch(plt.Rectangle((0.40, 0.30), 0.22, 0.44, fc="#fce8e6", ec="#d93025", lw=1.4))
ax.text(0.51, 0.58, "Masked Cross-Modal\nAttention\n(Transformer 2 layer,\nkey-padding mask)", ha="center", va="center", fontsize=7.5)
ax.text(0.51, 0.335, "[CLS] token", ha="center", va="center", fontsize=6.5, style="italic")
for x in (0.24,):
    ax.annotate("", xy=(0.40, 0.52), xytext=(x, 0.52), arrowprops=dict(arrowstyle="-|>", color="#333"))
ax.add_patch(plt.Rectangle((0.68, 0.40), 0.13, 0.12, fc="#e6f4ea", ec="#188038", lw=1.2))
ax.text(0.745, 0.46, "Head\n(MLP→1)", ha="center", va="center", fontsize=7.5)
ax.annotate("", xy=(0.68, 0.46), xytext=(0.62, 0.46), arrowprops=dict(arrowstyle="-|>", color="#333"))
ax.text(0.745, 0.26, "BCE(pos_weight)\n+ λ·InfoNCE\n(clean↔corrupted)", ha="center", va="center", fontsize=6.8)
ax.annotate("", xy=(0.745, 0.40), xytext=(0.745, 0.32), arrowprops=dict(arrowstyle="-|>", color="#333"))
ax.add_patch(plt.Rectangle((0.02, 0.08), 0.79, 0.09, fc="#f1f3f4", ec="#5f6368", lw=1))
ax.text(0.415, 0.125, "Missing-modality dropout augmentation (30% baris, 1 grup di-mask) → training", ha="center", va="center", fontsize=7)
ax.set_title("Proposed: encoder per-modalitas + masked cross-modal attention + contrastive", fontsize=10)
fig.tight_layout()
fig.savefig(OUT / "fig1_arsitektur.png", bbox_inches="tight")
plt.close(fig)
print("fig1 ok")

# ---------- Fig 2: kurva training ----------
d = json.load(open(R / "proposed_dl_full_summary.json"))
curve = np.array(d["eval_curve"])
fig, ax = plt.subplots(figsize=(5.5, 3.4))
ax.plot(curve[:, 0], curve[:, 1], "-o", ms=4, color="#1a73e8")
best_ep = d["best_epoch"]
ax.axvline(best_ep, ls="--", color="#d93025", lw=1)
ax.text(best_ep + 0.2, 0.5, f"best = epoch {best_ep}\n(eval PR-AUC {d['best_eval_prauc']:.4f})", fontsize=7.5, color="#d93025")
ax.set_xlabel("Epoch")
ax.set_ylabel("Eval PR-AUC (5 hari stratified)")
ax.set_title("Seleksi model: early stopping (patience 5)")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "fig2_kurva.png", bbox_inches="tight")
plt.close(fig)
print("fig2 ok")

# ---------- Fig 3: reliability diagram ----------
c = json.load(open(R / "ci_bootstrap_full.json"))
rows = c["reliability"]
conf = [r["conf"] for r in rows]
freq = [r["freq"] for r in rows]
fig, ax = plt.subplots(figsize=(5.5, 4.2))
ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
ax.plot(conf, freq, "-o", ms=5, color="#188038", label="Model (ECE = %.3f)" % c["ece"])
ax.set_xlabel("Confidence (predicted)")
ax.set_ylabel("Frequency (actual)")
ax.set_title("Reliability diagram — 10 bins")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "fig3_reliability.png", bbox_inches="tight")
plt.close(fig)
print("fig3 ok")

# ---------- Fig 4: PR-AUC per skenario ----------
sota = json.load(open(R / "sota_baselines_summary.json"))
dl = json.load(open(R / "proposed_dl_full_summary.json"))["scenarios"]
blend = json.load(open(R / "blend_ensemble_summary.json"))
scens = ["full", "missing_light", "missing_env_air", "random_dropout_30pct",
         "random_dropout_50pct", "random_dropout_70pct", "missing_light_and_env_air"]
labels = ["full", "miss\nlight", "miss\nenv_air", "drop\n30%", "drop\n50%", "drop\n70%", "light\n+env_air"]
x = np.arange(len(scens))
w = 0.16
fig, ax = plt.subplots(figsize=(7.5, 3.8))
colors = {"mean_impute": "#9aa0a6", "indicator_mlp": "#f9ab00", "gated_fusion": "#c5221f"}
for i, m in enumerate(sota):
    vals = [m["scenarios"][s]["pr_auc"] for s in scens]
    ax.bar(x + (i - 1.5) * w, vals, w, label=m["name"], color=colors[m["name"]], alpha=0.85)
vals_p = [dl[s]["pr_auc"] for s in scens]
ax.bar(x + 1.5 * w, vals_p, w, label="PROPOSED", color="#1a73e8")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=7.5)
ax.set_ylabel("PR-AUC (test)")
ax.set_ylim(0, 1.0)
ax.legend(fontsize=7, ncol=4, loc="upper left")
ax.set_title("Robustness: PR-AUC per skenario (3 baseline SOTA vs PROPOSED)")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "fig4_pr_auc.png", bbox_inches="tight")
plt.close(fig)
print("fig4 ok")
print("Selesai →", OUT)
