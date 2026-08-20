# Dataset A — Desain Preparasi & Split (draf)

**File:** `data/final_dataset_csv.csv` (34.76 MB, 230.976 baris, 22 kolom)
**Sumber:** Zenodo 10.5281/zenodo.20548374 (v3.0.0, CC-BY-4.0)
**Hasil EDA:** `data/EDA-A.json`

## 1. Tugas & Target

- **Tugas utama (paper):** biner okupansi — `ground_truth > 0` (dihuni / kosong). Rasio ~1:36 (6.255 : 224.721) → imbalanced.
- **Tugas sekunder (opsional):** multiclass 0-4 orang — kelas 3-4 sangat jarang (44 baris) → kemungkinan digabung ke kelas 2+ atau di-drop, putuskan saat eksperimen.
- **Metrik:** Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Macro-F1, MCC, Balanced Accuracy (imbalance → fokus PR-AUC/MCC/macro-F1).

## 2. Split — REVISI (day-level stratified, bukan temporal murni)

**Temuan EDA (20/08):** okupansi SANGAT sporadis per hari; hari-hari aktif: 4/12 (68,9% — anomali, kemungkinan kalibrasi), 14/12 (12,6%), 18/12 (4,6%), 21/12 (13,7%), 3-4/1, 8/1, 15/1. Gap data 22-31/12 (libur). Temporal split murni 70/15/15 → test (13-16/1) hanya 0,47% okupansi = tidak informatif.

**Desain final:**
1. **Unit split = HARI** (group), bukan baris — menjaga autokorelasi intra-hari & integritas sequence.
2. **Holdout test:** 20% hari dipilih ACR (acak-stratified berdasarkan total okupansi per hari) → pastikan test mengandung okupansi memadai; sisa 80% hari untuk CV.
3. **Validasi internal:** StratifiedGroupKFold 5-fold (group = hari, stratify = kelas okupansi) — fold utuh per hari.
4. **Hari 2023-12-04 (68,9%)**: tandai sebagai periode anomali; jalankan eksperimen dua mode — (a) disertakan, (b) dikecualikan → laporkan keduanya.
5. Setiap split dilaporkan distribusi okupansinya (kejujuran terhadap reviewer).

| Split | Unit | Proporsi | Catatan |
|---|---|---|---|
| Train | hari (acak stratified) | 64% | sisa setelah test, sebelum CV |
| Validation (per fold) | hari | 16% | StratifiedGroupKFold internal |
| Test (holdout) | hari | 20% | ACR stratified okupansi; HANYA sekali |

*Urutan waktu DALAM hari tetap dipertahankan (sequence 10s). Urutan antar-hari diacak — dilaporkan eksplisit sebagai trade-off karena okupansi sporadis.*

## 3. Normalisasi

- Z-score per-kanal, fit HANYA di train (mean/std train disimpan untuk inference test).
- Kanal biner (lamp_switch_led, switch_channel_1..3) TIDAK dinormalisasi.
- Fitur waktu opsional: hour-of-day (sin/cos encoding), day-of-week — catat sebagai ablation.

## 4. Missing Data Strategy

Dataset punya missing NATURAL (sound 2,0%, pfsense 0,47%, socket 0,21%, dll) — dua mode evaluasi:

1. **Mode natural:** biarkan missing apa adanya → model harus handle. Baseline: imputasi median/ffill; Proposed: missing-aware (masking).
2. **Mode simulasi (sensor failure):** dropout buatan per-kanal/per-grup-kanal dengan rate 10/30/50/70% + sensor failure total per modalitas (mis. semua kanal listrik hilang) + noise Gaussian/impulse + temporal misalignment (+1s/+5s/+10s/+30s).

Grup modalitas untuk eksperimen dropout:

| Modalitas | Kanal |
|---|---|
| Environment (indoor air) | co2, humidity, temperature_1, temperature_2 |
| Light | lux_1, lux_2 |
| Acoustic | sound |
| Electrical server | server_cur_{current,power,voltage} |
| Electrical pfsense | pfsense_cur_{current,power,voltage} |
| Electrical socket | socket_cur_{current,power,voltage} |
| Device state | lamp_switch_led, switch_channel_1..3 |

## 5. Pipeline (skrip di `data/` — venv `.venv/`)

1. `prep_split.py` — load CSV → split temporal → simpan `train.csv`, `val.csv`, `test.csv` + `scaler.json`
2. `features.py` — windowing (sequence length 60×10s = 10 menit, stride 10s) untuk model sekuensial; agregasi statistik per window untuk tree models
3. `baselines.py` — RF, XGBoost (feature-agg), LSTM (sequence) — single-modality + early/late fusion
4. `proposed.py` — missing-modality-aware framework (setelah baseline jalan)

## 6. Reproducibility

- Seed tetap (mis. 42) untuk semua eksperimen.
- Versi Python: 3.14 (system) / venv proyek; numpy 2.5.1, pandas/sklearn/xgboost via venv.
- Semua angka eksperimen dicatat di `results/` (belum dibuat) + git (repo riset lokal).
