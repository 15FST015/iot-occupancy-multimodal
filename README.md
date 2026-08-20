# Robust Multimodal IoT Occupancy Detection Under Sensor Failure

Cross-modal attention + contrastive learning + missing-modality augmentation
for occupancy detection robust to sensor failure.

**Paper (draf):** Robust Multimodal IoT Intelligence Under Sensor Failure:
Cross-Modal Attention and Contrastive Learning for Occupancy Detection
*(target: Scopus Q1 — Energy and Buildings / Building and Environment)*

## Ringkasan

- **Masalah:** model multimodal IoT runtuh saat sensor gagal (modalitas hilang).
  Baseline XGBoost: F1 0.74→0.16 saat light hilang; 0.74→0.17 saat dropout 70%.
- **Solusi:** encoder per-modalitas + masked cross-modal attention + dropout
  augmentation (missing-modality) + contrastive InfoNCE.
- **Hasil:** F1 unggul di 14/14 skenario failure; PR-AUC (metrik utama) 0.824
  full, 0.887 saat light hilang (vs baseline 0.359), 0.887 random 70% (vs 0.46).

## Reproduksi

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Data: unduh dari Zenodo (CC-BY-4.0)
#    https://doi.org/10.5281/zenodo.20548374 → simpan sebagai data/final_dataset_csv.csv
# 2. Split day-level stratified
python data/prep_split.py
# 3. Baseline & robustness
python experiments/baselines.py
python experiments/robustness.py
# 4. Proposed (tree) & blending
python experiments/proposed_v1.py
python experiments/blend_ensemble.py
# 5. Proposed DL (final: lr 3e-4, eval 5 hari stratified, early stop)
python experiments/proposed_dl.py 30 full 0.0003 0.0001 5
# 6. Ablation & statistik
python experiments/proposed_dl.py 30 no_contrastive 0.0003 0.0001 5   # dst.
python experiments/ci_bootstrap.py full 1000
python experiments/ci_cluster_bootstrap.py
# 7. Noise/drift/misalignment & cross-dataset
python experiments/noise_misalignment.py
python experiments/cross_dataset.py
```

Semua eksperimen: seed 42; hasil → `experiments/results/*.json`.

## Struktur

```
data/            # split, scaler, EDA (dataset A: Zenodo 10.5281/zenodo.20548374)
experiments/     # skrip reproducible (baseline, robustness, proposed, ablation, CI)
results/         # JSON hasil per eksperimen (diverifikasi Jurnal-Evaluator)
```

## Dataset

| Dataset | Status | Lisensi |
|---|---|---|
| A: Smart office PUCRS (utama) | Zenodo 10.5281/zenodo.20548374 | CC-BY-4.0 |
| B: HPDmobile (analisis domain-shift) | Figshare 5364449 | CC0 |
| D: ECO (opsional) | via kontak penulis | — |

## Metrik utama

F1, PR-AUC (utama — threshold-free), ROC-AUC, MCC, balanced accuracy;
CI bootstrap 1000 resamples (per-baris + block 1-jam); ECE kalibrasi.

## Kontak

Amesanggeng Pataropura, M.Kom — amesanggeng@buddhidharma.ac.id
Universitas Buddhi Dharma, Tangerang, Indonesia
