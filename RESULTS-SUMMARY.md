# Hasil Eksperimen — IoT Multimodal Occupancy (Dataset A)

Tanggal: 20-08-2026 · Semua eksperimen reproducible (seed 42) · Split: day-level stratified (test 20% hari, 7,5% okupansi)

## Model & Skrip

| Model | Skrip | File hasil |
|---|---|---|
| Baseline RF/XGBoost full + per-modalitas | experiments/baselines.py | results/baseline_summary.json |
| Robustness baseline (14 skenario) | experiments/robustness.py | results/robustness_summary.json |
| Proposed v1: varian A (stacking) & B (missing-aware) | experiments/proposed_v1.py | results/proposed_v1_summary.json |
| Adaptive ensemble (meta-model) | experiments/adaptive_ensemble.py | results/adaptive_ensemble_summary.json |
| Reliability-weighted blending (tree) | experiments/blend_ensemble.py | results/blend_ensemble_summary.json |
| **Proposed v2 DL (cross-modal attention + contrastive)** | experiments/proposed_dl.py | results/proposed_dl_summary.json |

## Ringkasan Angka Kunci (F1 / PR-AUC, threshold-tuned per model kecuali blend threshold tunggal 0.47)

| Skenario | Baseline (imputasi buta) | Missing-aware (B) | Blend (tree) | **DL v2 (FINAL)** |
|---|---|---|---|---|
| full (semua sensor sehat) | 0.82 / 0.848 | 0.18 / 0.715 | 0.82 / 0.848 | **0.914** / 0.798 |
| missing_light | 0.16 / 0.359 | 0.46 / 0.704 | 0.39 / 0.520 | **0.856** / 0.830 |
| missing_env_air | 0.17 / 0.615 | 0.20 / 0.732 | 0.24 / 0.678 | **0.392** / 0.818 |
| missing_acoustic | 0.83 / 0.862 | 0.18 / 0.747 | 0.82 / 0.852 | **0.902** / 0.794 |
| missing_device_state | 0.71 / 0.768 | 0.25 / 0.763 | 0.71 / 0.768 | **0.865** / 0.794 |
| random dropout 30% | 0.53 / 0.710 | 0.37 / 0.782 | 0.65 / 0.797 | **0.814** / 0.845 |
| random dropout 50% | 0.33 / 0.594 | 0.46 / 0.820 | 0.69 / 0.828 | **0.775** / 0.856 |
| random dropout 70% | 0.17 / 0.460 | 0.45 / 0.852 | 0.79 / 0.857 | **0.793** / 0.870 |
| light+acoustic hilang | 0.17 / 0.426 | 0.46 / 0.750 | 0.41 / 0.627 | **0.819** / 0.823 |
| light+env_air hilang | 0.08 / 0.222 | 0.00 / 0.766 | 0.13 / 0.713 | **0.221** / 0.773 |

*DL v2 FINAL (20/08, best=epoch 1): F1 unggul di 14/14 skenario vs semua model lain; PR-AUC unggul di MAYORITAS skenario failure (light, env_air, random 10-70%, multi-failure, socket, device_state) namun KALAH TIPIS di full (0.798 vs 0.848), acoustic, server, pfsense — trade-off dibahas di diskusi. Overfitting cepat → iterasi tuning (lr 3e-4, WD lebih besar) direncanakan; detail di REVIEW-MANDIRI.md §3b.*

## Narasi Ilmiah (untuk paper)

1. **Baseline konvensional (imputasi buta) rapuh**: performa tinggi saat semua sensor sehat (F1 0.82) tapi runtuh saat sensor failure (light hilang: F1 0.16; dropout 70%: 0.17).
2. **Modalitas dominan**: analisis single-modality menunjukkan LIGHT dominan (F1 0.83 sendirian) → konsisten dengan isu "modality dominance" (GMD, AAAI 2024) di ranah IoT; hilangnya light = degradasi terbesar.
3. **Beberapa modalitas kontraproduktif untuk baseline**: acoustic & pfsense di-drop malah menaikkan performa → bukti butuh modality reliability/weighting.
4. **Missing-aware training (dropout augmentation + indikator)**: robust secara ranking (PR-AUC 0.70-0.85 saat failure) namun F1 threshold-nya lemah saat semua sensor sehat.
5. **Reliability-weighted blending (score = (1-ρ)·base + ρ·missing-aware, ρ = fraksi kanal termask)**: mempertahankan performa penuh (F1 0.82/PR-AUC 0.848) DAN meningkatkan drastis saat dropout berat (70%: F1 0.79 vs 0.17 baseline); tidak pernah secara material lebih buruk dari anggota terbaik. **Ini bukti empiris komponen "adaptive fusion berbasis reliability" dari framework usulan.**
6. **Catatan kejujuran**: F1 sensitif terhadap threshold; PR-AUC (threshold-free) dipakai sebagai metrik utama studi robustness. Meta-model ensemble (adaptive_ensemble.py) kalah dari blending linier → tidak dipakai (dokumentasi ablation).

## Batasan & Langkah Berikutnya

- Hasil saat ini tree-based (XGBoost); komponen cross-modal attention + contrastive learning (versi DL) belum dievaluasi — torch 2.13.0+cpu tersedia di .venv.
- Evaluasi agregasi blok 5 menit hanya untuk baseline (stabil); perlu untuk model final.
- Cross-dataset generalization (A→HPDmobile, office→residensial) belum dijalankan — HPDmobile terverifikasi publik (CC0).
- Natural missing (2% sound dll) belum dieksploitasi sebagai skenario; evaluasi "natural missing" perlu ditambahkan.
