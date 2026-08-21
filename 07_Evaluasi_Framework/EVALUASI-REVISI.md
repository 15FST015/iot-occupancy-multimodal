# Evaluasi & Revisi Framework — Occupancy Multimodal

## Revisi yang DILAKUKAN (20/08 — framework dinamis)
1. **H5 gagal (cross-dataset)**: klaim "Generalizable" DITURUNKAN → judul direvisi;
   cross-dataset jadi analisis domain-shift (diskusi/limitation) — bukan disembunyikan
2. **Seleksi model tidak defensible** (best=epoch 1, eval 2 hari): revisi → eval 5 hari
   stratified, lr 3e-4, early stop patience 5 → best_epoch 8 (PR-AUC naik 13/14 skenario)
3. **Contrastive netral**: diposisikan "complementary" (bukan klaim utama) — novelty statement
   final 5 kontribusi (REVIEW-MANDIRI §3i)
4. **6 MAJOR evaluator ditutup**: seleksi model, repositioning, cluster bootstrap (blok 1-jam
   — level-hari tidak informatif, 6 hari test/3 kosong), baseline SOTA (3 metode fair),
   repo publik (github.com/15FST015/iot-occupancy-multimodal), koreksi dokumentasi

## Bukti evaluasi independen
- REVIEW-EVALUATOR.md: audit angka LULUS BERSIH; verdict BERSYARAT → setelah 6 MAJOR: LAYAK
- EVALUASI-NASKAH-V1.md: MAJOR 3 | MINOR 10 | COMMENT 8 → v2 memperbaiki semua

## Keputusan akhir
- Judul final tanpa "Generalizable" · PR-AUC metrik utama · E&B (Elsevier Q1) target utama
- Status: T15 submit manual penulis (package lengkap di drafts/ + submissions/)
