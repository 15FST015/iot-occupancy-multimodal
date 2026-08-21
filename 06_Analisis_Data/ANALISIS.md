# Analisis Data — Occupancy Multimodal

## Metrik
F1, PR-AUC (UTAMA), ROC-AUC, MCC, balanced-acc; threshold ditune di eval.

## Hasil kunci (model final: lr 3e-4, best_epoch 8, thr 0.95)
| Skenario | PR-AUC proposed | Baseline SOTA terbaik |
|---|---|---|
| full | 0.824 | 0.791 (gated) |
| missing_light | 0.887 | 0.842 (indicator) |
| random 70% | 0.887 | 0.809 (indicator) |
| light+env_air | 0.823 | 0.696 (indicator) |

- Menang PR-AUC 14/14 vs 3 baseline SOTA (mean_impute/indicator/gated)
- CI bootstrap 1000 (per-baris + block 1-jam); ECE 0.055
- Noise/drift/misalignment: F1 ≥ 0.86 semua kondisi
- Audit Jurnal-Evaluator: 100% angka cocok JSON (0 dikarang)

## Artefak
- experiments/results/*.json (17 file) · RESULTS-SUMMARY.md · REVIEW-MANDIRI.md §3-3h
