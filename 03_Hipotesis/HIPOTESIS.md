# Hipotesis — Occupancy Multimodal (status teruji)

| # | Hipotesis | Status | Bukti |
|---|---|---|---|
| H1 | Baseline runtuh saat modalitas dominan hilang | ✅ TERBUKTI | XGB F1 0.82→0.16 (light hilang); 0.17 @dropout 70% |
| H2 | Dropout augmentation = mekanisme paling kritis | ✅ TERBUKTI | ablation: tanpa augmentasi F1 0.001 vs 0.856 |
| H3 | Masked attention mempertahankan akurasi saat sensor gagal | ✅ TERBUKTI | PR-AUC menang 14/14 vs 3 baseline SOTA |
| H4 | Contrastive memperkuat robustness | ⚠️ SEBAGIAN | efek sekunder/komplementer (ablation jujur) |
| H5 | Generalizable lintas gedung (zero-shot) | ❌ TIDAK TERBUKTI | A→HPDmobile ROC<0.5 → klaim DITURUNKAN dari judul |

## Keputusan metodologis terkait
- H5 gagal → framing "robustness in-domain + analisis domain-shift" (20/08)
- PR-AUC = metrik utama (threshold-free) setelah evaluasi sensitivitas F1
