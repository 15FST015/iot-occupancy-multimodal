#!/usr/bin/env python
"""Generate figures for manuscript v1 from verified result JSONs."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = "/mnt/sda3/Documents/Jurnal/Amesanggeng/iot-multimodal-occupancy/experiments/results"
OUT = "/mnt/sda3/Documents/Jurnal/Amesanggeng/iot-multimodal-occupancy/drafts/figures"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

# ---------- Fig: training/eval curve (PR-AUC per epoch) ----------
d = json.load(open(f"{BASE}/proposed_dl_full_summary.json"))
curve = np.array(d["eval_curve"])
fig, ax = plt.subplots(figsize=(5.2, 3.2))
ax.plot(curve[:, 0], curve[:, 1], "-o", ms=4, color="#1f77b4")
best = int(d["best_epoch"])
ax.axvline(best, ls="--", color="#d62728", lw=1)
ax.annotate(f"best epoch {best}\n(eval PR-AUC {d['best_eval_prauc']:.3f})",
            xy=(best, d["best_eval_prauc"]), xytext=(best + 1.2, d["best_eval_prauc"] - 0.012),
            fontsize=8, color="#d62728")
ax.set_xlabel("Epoch")
ax.set_ylabel("Eval PR-AUC (5 stratified days)")
ax.set_ylim(0.40, 0.52)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/fig_training_curve.png", dpi=200)
plt.close(fig)

# ---------- Fig: reliability diagram ----------
d = json.load(open(f"{BASE}/ci_bootstrap_full.json"))
rel = d["reliability"]
conf = [b["conf"] for b in rel]
freq = [b["freq"] for b in rel]
n = [b["n"] for b in rel]
fig, ax = plt.subplots(figsize=(4.6, 4.2))
ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
ax.plot(conf, freq, "-o", ms=4, color="#2ca02c", label="Model (10 bins)")
# size-proportional markers for bin population
ax.scatter(conf, freq, s=[max(8, np.sqrt(x) * 3) for x in n], color="#2ca02c", alpha=0.35, zorder=0)
ax.set_xlabel("Mean predicted confidence")
ax.set_ylabel("Observed frequency")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.text(0.98, 0.04, f"ECE = {d['ece']:.3f}", ha="right", fontsize=9)
ax.legend(fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig(f"{OUT}/fig_reliability.png", dpi=200)
plt.close(fig)

# ---------- Fig: PR-AUC grouped bars across 14 scenarios (key models) ----------
def load_scen_pr(fname, key=None):
    d = json.load(open(f"{BASE}/{fname}"))
    if key is not None:
        d = d[key]
    return {k: v["pr_auc"] for k, v in d.items()}

scen_order = ["full", "missing_env_air", "missing_light", "missing_acoustic",
              "missing_elec_server", "missing_elec_pfsense", "missing_elec_socket",
              "missing_device_state", "random_dropout_10pct", "random_dropout_30pct",
              "random_dropout_50pct", "random_dropout_70pct",
              "missing_light_and_acoustic", "missing_light_and_env_air"]
labels = ["full", "env-air", "light", "acoustic", "server", "pfsense", "socket",
          "device", "r10%", "r30%", "r50%", "r70%", "light+ac", "light+env"]

proposed = load_scen_pr("proposed_dl_full_summary.json", "scenarios")
blend = load_scen_pr("blend_ensemble_summary.json")
base_tuned = load_scen_pr("proposed_v1_summary.json", "baseline_tuned")
sota = json.load(open(f"{BASE}/sota_baselines_summary.json"))
mi = {k: v["pr_auc"] for k, v in sota[0]["scenarios"].items()}
ind = {k: v["pr_auc"] for k, v in sota[1]["scenarios"].items()}
gat = {k: v["pr_auc"] for k, v in sota[2]["scenarios"].items()}

x = np.arange(len(scen_order)); w = 0.12
fig, ax = plt.subplots(figsize=(9.5, 3.6))
colors = {"XGB (blind imputation)": "#7f7f7f", "Tree blend": "#bcbd22",
          "Mean-impute MLP": "#17becf", "Indicator MLP": "#9467bd",
          "Gated fusion": "#ff7f0e", "Proposed": "#d62728"}
series = [("XGB (blind imputation)", base_tuned), ("Tree blend", blend),
          ("Mean-impute MLP", mi), ("Indicator MLP", ind), ("Gated fusion", gat),
          ("Proposed", proposed)]
for i, (name, s) in enumerate(series):
    vals = [s[sc] for sc in scen_order]
    ax.bar(x + (i - 2.5) * w, vals, w, label=name, color=colors[name], edgecolor="white", lw=0.4)
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("PR-AUC")
ax.set_ylim(0, 1.0)
ax.legend(fontsize=7.5, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.42))
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/fig_prauc_scenarios.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# ---------- Fig: PR-AUC vs missing rate (30/50/70%) ----------
rates = [0.1, 0.3, 0.5, 0.7]
def pr_at_rates(s):
    return [s[f"random_dropout_{int(r*100)}pct"] for r in rates]
fig, ax = plt.subplots(figsize=(4.8, 3.4))
for name, s in series:
    ax.plot([f"{int(r*100)}%" for r in rates], pr_at_rates(s), "-o", ms=4,
            label=name, color=colors[name], lw=1.4)
ax.set_xlabel("Random channel dropout rate")
ax.set_ylabel("PR-AUC")
ax.set_ylim(0.3, 1.0)
ax.grid(alpha=0.3)
ax.legend(fontsize=7, loc="lower left")
fig.tight_layout()
fig.savefig(f"{OUT}/fig_prauc_missing_rate.png", dpi=200)
plt.close(fig)

# ---------- Fig: dataset GT distribution + natural missing ----------
eda = json.load(open("/mnt/sda3/Documents/Jurnal/Amesanggeng/iot-multimodal-occupancy/data/EDA-A.json"))
gt = eda["gt_dist"]
fig, ax = plt.subplots(figsize=(4.4, 3.2))
cats = ["0", "1", "2", "3", "4"]
vals = [gt[c] for c in cats]
bars = ax.bar(cats, vals, color=["#7f7f7f", "#1f77b4", "#ff7f0e", "#d62728", "#d62728"])
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 3000, f"{v:,}\n({100*v/230976:.2f}%)",
            ha="center", fontsize=7.5)
ax.set_yscale("log")
ax.set_xlabel("Ground-truth occupants (classes)")
ax.set_ylabel("Number of 10-s rows (log)")
ax.set_title("Occupancy ground-truth distribution", fontsize=9)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/fig_gt_distribution.png", dpi=200)
plt.close(fig)

print("Figures written to", OUT)
print(sorted(os.listdir(OUT)))
