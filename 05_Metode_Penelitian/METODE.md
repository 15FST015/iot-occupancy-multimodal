# Metode Penelitian — Occupancy Multimodal

## Desain
Kuantitatif eksperimental; dataset publik Zenodo 10.5281/zenodo.20548374 (CC-BY-4.0):
230.976 sampel, 42,6 hari, 22 kanal, GT okupansi 0-4.

## Urutan proses (15 tahap — detail di URUTAN-PENELITIAN.md §1b)
1. Verifikasi data → 2. Pembersihan (eksklusi 44 baris kelas 3-4) → 3. Normalisasi z-score
train-only → 4. Fitur waktu → 5. Split day-level stratified → 6. Augmentasi missing-modality
→ 7. Baseline (XGB/RF/single-modality) → 8. Robustness protokol (26 kondisi) → 9. Proposed tree
+ blending → 10. Proposed DL (attention+contrastive) → 11. Threshold tuning eval → 12. Evaluasi
test → 13. Statistik (CI bootstrap ×2, ECE) → 14. Ablation (5 varian) → 15. Cross-dataset

## Protokol robustness
14 skenario failure (7 missing per-modalitas, 4 dropout random, 2 multi-failure)
+ 12 skenario stress (3 gaussian, 2 impulse, 2 drift, 4 misalignment, 1 block)

## Artefak
- experiments/*.py (12 skrip reproducible) · data/prep_split.py · URUTAN-PENELITIAN.md §1b
