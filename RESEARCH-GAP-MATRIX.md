# Research Gap Matrix: IoT Multimodal Occupancy

**Status:** Fase 1 selesai (20-08-2026) — 34 paper terverifikasi via Crossref/arXiv (2022-2026).
**Konteks:** Niche "Missing-Modality-Aware Multimodal IoT Learning" — target Scopus Q1.

> **Legenda:** `Q` = estimasi quartile (ScimagoJR; verifikasi sebelum sitasi). `?` = belum terkonfirmasi dari abstrak, perlu cek PDF. DOI tersedia untuk semua entri jurnal → siap disitasi.

## Matriks Literatur

| # | Paper | Thn | Jurnal | Q | Dataset | Modalities | Model | Fusion | Missing Mod. | Generalisasi | Limitasi | Research Gap |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | A multimodal dataset for environmental occupancy detection | 2026 | Data in Brief | Q2 | **Zenodo v3 (terverifikasi)** | CO2, RH, temp×2, lux×2, sound, listrik (server/pfsense/socket: current/power/voltage), lamp+switch state, GT okupansi 0-4 | — (dataset paper) | — | Ada natural (sound 2%, dll) | ? | Paper data; tidak ada baseline model | **Dataset kunci — Profile A di bawah** |
| 2 | Multimodal sensor fusion framework for residential building occupancy detection | 2022 | Energy and Buildings | Q1 | Residensial (publik?) | Environmental + ? | ? | Multi-level | ? | ? | ? | Asumsi modalitas lengkap |
| 3 | Multimodal Framework for Smart Building Occupancy Detection | 2024 | Sustainability | Q2 | Publik (env. sensing) | Environmental | ? | ? | Tidak | Tidak | Akurasi turun saat okupansi tinggi; kurang dataset publik | Evaluasi skala/okupansi; robustness |
| 4 | Fusion of Environmental Sensors for Occupancy Detection in a Real Construction Site | 2023 | Sensors | Q2 | Lapangan konstruksi | Temp, dust, air quality | NN + late fusion | Late fusion (concat) | Tidak | Tidak | Late fusion saja; satu lokasi | Evaluasi lebih dari satu site |
| 5 | Indoor Occupancy Detection Based on Environmental Data Using CNN-XGBoost | 2022 | Sustainability | Q2 | Residensial (validasi eksperimen) | Environmental | CNN-XGBoost | Early (feature) | Tidak | Tidak | Satu gedung | Validasi lintas gedung |
| 6 | Multimodal feature fusion and ensemble learning for non-intrusive occupancy monitoring using smart meters | 2025 | Building and Environment | Q1 | Smart meter publik | Listrik (smart meter) + ? | Ensemble | ? | Tidak | ? | Fokus single-stream listrik | Fusi antar-modalitas heterogen |
| 7 | DMFF: Deep multimodel feature fusion for building occupancy detection | 2024 | Building and Environment | Q1 | ? | ? | Deep feature fusion | Intermediate | Tidak | ? | ? | Ketahanan saat sensor gagal |
| 8 | Transfer learning for estimating occupancy and recognizing activities in smart buildings | 2022 | Building and Environment | Q1 | ? | ? | Transfer learning | — | Tidak | Ya (transfer antar gedung) | ? | Transfer terbatas antar-domain; missing modality belum disentuh |
| 9 | Deep and transfer learning for building occupancy detection: A review | 2022 | Eng. Appl. of AI | Q1 | Review | Review | Review | Review | Tidak dibahas | Dibahas parsial | Review; gap metodologis belum diidentifikasi utk robustness | **Review: konfirmasi belum ada yg menangani missing modality + cross-dataset** |
| 10 | Real-Time Occupancy Detection Using Low-Resolution Thermopile Array | 2022 | IEEE Access | Q2 | Lokal | Thermal (1 modalitas) | ? | Single-mod | — | Tidak | Single sensor | Fusi multimodal diperlukan |
| 11 | Audio Feature Application for CO2 Sensor-Based Occupancy Detection Enhancement | 2026 | Buildings | Q2 | Lokal (ventilasi alami) | CO2 + audio (MEMS mic) | Random Forest | Feature-level | Tidak | Tidak | 2 modalitas saja; RF | Fusi >2 modalitas; deep fusion |
| 12 | Improving Indoor Occupancy Detection Accuracy of the SLEEPIR Sensor Using LSTM | 2023 | IEEE Sensors J. | Q1 | Lokal | PIR (SLEEPIR) | LSTM | Single-mod | — | Tidak | Single sensor | Fusi dengan modalitas lain |
| 13 | Cascaded DL framework for non-intrusive load and occupancy monitoring | 2025 | J. Building Engineering | Q1 | Smart meter | Listrik | Cascaded DL | — | Tidak | ? | Single sumber data | Fusi multimodal |
| 14 | MissModal: Increasing Robustness to Missing Modality in Multimodal Sentiment Analysis | 2023 | TACL | — (ACL) | MSA (text/audio/vision) | Text, audio, vision | Representation-based | Joint/coordinated | **Ya (acak)** | Tidak | Domain NLP; skala kecil | **Metode belum diuji di IoT/sensor** |
| 15 | Gradient-Guided Modality Decoupling (GMD) for Missing-Modality Robustness | 2024 | AAAI | Q1 (conf) | Vision/Language | Vision, text | GMD + gradien | Decoupling | **Ya** | Tidak | Modalitas dominan; butuh label | Gradien untuk sensor IoT belum dieksplorasi |
| 16 | Unified multimodal computational pathology with missing-modality robustness via Riemannian learning | 2027 | Information Fusion | Q1 | Patologi | WSI + ? | Riemannian | ? | **Ya** | ? | Domain medis | Transfer ke IoT belum ada |
| 17 | TCTR: Text-Guided Contrastive Learning + Token-Level Reconstruction for missing modalities | 2026 | Information Fusion | Q1 | MSA | Text, audio, vision | Contrastive + reconstruction | Token-level | **Ya** | Tidak | Domain MSA | **Kombinasi contrastive + missing-modality = relevan; belum di IoT** |
| 18 | Graph attention contrastive learning with missing modality for multimodal recommendation | 2025 | Knowledge-Based Systems | Q1 | Rekomendasi | User/item multi-mod | GNN + contrastive | Graph | **Ya** | ? | Domain rekomendasi | GNN utk sensor graph IoT? |
| 19 | Multimodal learning with missing modality for chemical process system | 2025 | Computers & Chemical Eng. | Q1 | Proses kimia | Sensor industri | ? | ? | **Ya** | ? | Domain proses kimia | Sensor industri ≠ gedung pintar |
| 20 | FedMIR: Multimodal Federated Learning with Missing Modality Imputation | 2026 | Sensors | Q2 | Federated | ? | FL + imputation | ? | **Ya** | ? | Butuh federasi multi-client | FL angle; single-device belum |
| 21 | Heterogeneous Multimodal FL With Missing Modality via Mask-Restoration | 2026 | IEEE TMM | Q1 | Federated | ? | Mask-restoration | ? | **Ya** | ? | Federasi | — |
| 22 | MCL-MGN: Multi-level contrastive learning for missing modality generation | 2026 | Digital Signal Processing | Q2 | MSA | Text, audio, vision | Contrastive + generation | ? | **Ya** | Tidak | Domain MSA | Generasi modalitas ≠ adaptif fusion |
| 23 | Conformal Fusion Under Missing Modalities | 2026 | arXiv | — | Umum | Umum | Conformal prediction | Any | **Ya (teoretis)** | ? | Teoretis; belum aplikasi IoT | **Kerangka ketidakpastian utk missing modality** |
| 24 | GAUGE: Counterfactual Gating for Incomplete Multimodal Classification | 2026 | arXiv | — | Umum | Umum | Gating + counterfactual | Dynamic | **Ya** | ? | Umum | Belum divalidasi sensor IoT |
| 25 | Co-Learning for Missing Arbitrary Modalities | 2026 | arXiv | — | Umum (CV/AI) | Umum | Co-learning | ? | **Ya** | ? | Umum | — |
| 26 | M3F-UAV: Missing-Modality Multimodal Foundation Model (Wireless Sensing) | 2026 | arXiv | — | Wireless/UAV | RF + ? | Foundation model | ? | **Ya** | ? | Domain UAV | Foundation model utk occupancy? |
| 27 | A Cross-Modal Attention-Driven Multi-Sensor Fusion (Point Cloud Segmentation) | 2025 | Sensors | Q2 | Point cloud | LiDAR + kamera | Cross-attention | Cross-modal | Tidak | ? | Domain CV 3D | **Cross-modal attention terbukti; belum utk environmental sensing** |
| 28 | MixFormer: cross-intra-modal attention for fault diagnosis | 2026 | Meas. Sci. Technol. | Q2 | Mesin industri | Multi-sensor vibrasi | Transformer | Cross-intra | Tidak | ? | Domain fault diagnosis | Arsitektur mirip, aplikasi berbeda |
| 29 | Cross-modal attention fusion network for RGB-D semantic segmentation | 2023 | Neurocomputing | Q1 | RGB-D | RGB + depth | Cross-attention | Cross-modal | Tidak | ? | Domain CV | — |
| 30 | Cross-modal guiding attention for RGBT tracking | 2026 | Information Fusion | Q1 | RGB-T | RGB + thermal | Guiding attention | Cross-modal | Tidak | ? | Domain tracking | — |
| 31 | Learning from the global view: Supervised contrastive learning of multimodal representation | 2023 | Information Fusion | Q1 | Umum | Multi-mod | SupCon global view | Contrastive | Tidak | Ya (global view) | Butuh label | Contrastive utk representasi multimodal IoT |
| 32 | MMBind: Multimodal Learning in IoT | 2024 | arXiv | — | IoT sensing | Distributed heterogeneous | Binding (alignment) | Embedding | Tidak | Ya (heterogen) | Tanpa label besar | **Align antar-sensor IoT; belum missing-modality** |
| 33 | SB-BEVFusion: Robustness against Sensor Malfunction | 2026 | arXiv | — | Autonomous driving | Kamera + LiDAR | BEV fusion | Early/mid | **Ya (malfunction)** | ? | Domain kendaraan | **Robustness sensor failure di 3D; bukan gedung** |
| 34 | When Multi-Sensor Fusion Fails to Generalize (Cattle Posture) | 2026 | arXiv | — | Peternakan | IMU + ? | Fusion | ? | Tidak | **Ya (distribution shift)** | Domain ternak | **Bukti empiris: fusi gagal generalisasi lintas subjek/waktu** |

## Temuan Utama (SOTA Map 2022-2026)

1. **Occupancy detection multimodal SUDAH ramai** (E&B, B&E, JOBE, Sensors, Sustainability) — tapi hampir semua mengasumsikan **semua modalitas tersedia** dan dievaluasi **satu gedung/lokasi**.
2. **Missing-modality robustness DIDOMINASI domain non-IoT**: sentiment analysis (TACL, Information Fusion), medis/patologi (Information Fusion), rekomendasi (KBS), federated learning (TMM, Sensors), proses kimia. **Belum ada yang mentransfer ke environmental sensing gedung pintar.**
3. **Cross-modal attention + contrastive learning terbukti efektif** di CV (RGB-D, RGB-T, point cloud) dan MSA — tapi **belum dikombinasikan dengan missing-modality handling untuk IoT occupancy**.
4. **Cross-dataset generalization jarang diuji**; kalaupun ada (transfer learning B&E 2022), tidak mempertimbangkan sensor failure. Paper arXiv 2606.24986 menunjukkan fusi multimodal **bisa gagal generalisasi** — argumen kuat untuk penelitian ini.
5. **Dataset publik multimodal occupancy baru rilis 2026** (Data in Brief 10.1016/j.dib.2026.112948) — peluang jadi salah satu evaluator pertama.

## Research Gap (Sintesis)

- **G1:** Tidak ada framework multimodal IoT yang menangani **missing modality sistematis** (dropout 10–70%, sensor failure per-modalitas, temporal misalignment) untuk occupancy detection.
- **G2:** Metode missing-modality yang ada (imputation, generation, decoupling) **belum divalidasi pada sensor lingkungan heterogen** (CO2/temp/RH/audio/listrik) dengan karakteristik drift & noise nyata.
- **G3:** **Cross-dataset / cross-building generalization** + ketahanan saat sensor gagal **belum dievaluasi bersama** dalam satu studi.
- **G4:** Kombinasi **cross-modal attention + contrastive learning + modality reliability** belum diuji untuk occupancy IoT; dominasi modalitas (GMD, AAAI 2024) belum dieksplorasi di ranah sensor.

## Novelty Statement (draf)

> "Kami mengusulkan kerangka fusi multimodal yang sadar-missing-modality untuk deteksi okupansi gedung pintar, menggabungkan (i) deteksi reliabilitas modalitas, (ii) cross-modal attention yang toleran dropout, dan (iii) contrastive learning untuk representasi yang stabil saat degradasi sensor — dievaluasi sistematis di bawah sensor failure, noise, dan temporal misalignment, serta divalidasi lintas-dataset publik."

## Kandidat Dataset (terverifikasi via Crossref/Zenodo/OpenAlex/Figshare — 20/08)

| Kode | Dataset | Sumber | Status verifikasi |
|---|---|---|---|
| A | **A multimodal dataset for smart office occupancy estimation** | Zenodo concept DOI 10.5281/zenodo.19184829 · versi 3.0.0: 10.5281/zenodo.20548374 · CC-BY-4.0 | ✅ **PUBLIK — TERUNDUH** (lihat Profil A) |
| B | **HPDmobile** — Human Presence Detection, 6 rumah (H1-H6), 4-8 minggu/rumah: env (temp/RH/light/CO2/TVOC/distance) + audio + image + GT | Figshare collection 5364449 (CC0, ~60 item ZIP); kode GitHub mhsjacoby/HPDmobile (GPL-3.0) + Zenodo 10.5281/zenodo.4655276; descriptor Sci. Data 10.1038/s41597-021-01055-x | ✅ **PUBLIK (CC0)** — kandidat terbaik uji cross-dataset (office→residential domain shift) |
| C | "Environmental sensor data for occupancy detection" (gedung konstruksi 27 lantai, ASHVIN; sound/temp/RH/dust/pressure) — dataset paper Sensors 2023 | Zenodo 10.5281/zenodo.8203278 | ⛔ **RESTRICTED** (access_right=restricted, 0 file, butuh request ke penulis; paper klaim "public" tapi praktiknya tidak) |
| D | **ECO dataset** (ETH Zurich, Kleiminger et al.) — smart meter 6 rumah tangga Swiss, 8 bulan, 1 Hz + GT okupansi — dataset paper B&E 2025 | https://vs.inf.ethz.ch/res/show.html?what=eco-data | ✅ **PUBLIK** (gratis riset, tapi unduhan butuh kontak W. Kleiminger; link lama 404, ETH archive login) |

*Catatan cross-dataset: B (HPDmobile, residensial) + A (smart office) = pasangan ideal untuk uji generalisasi domain office↔residensial. C tidak direkomendasikan (akses dibatasi). D (ECO) relevan untuk eksperimen listrik-only.*

## Profil Dataset A (TERVERIFIKASI — file CSV 34.76 MB di `data/final_dataset_csv.csv`)

- **Asal:** smart office PUCRS, Porto Alegre/Brasil; hardware = perangkat komersial Tuya (TuyAPI, LAN) + node kustom ESP32 (Arduino C++, HTTP); MongoDB → JSON Lines + CSV.
- **Periode:** 2023-12-04 → 2024-01-16 (42,6 hari), agregasi 10 detik → **230.976 baris**.
- **Kanal (22 kolom):**
  - Environmental (7): co2, humidity, temperature_1, temperature_2, lux_1, lux_2, sound
  - Electrical (9): server_cur_{current,power,voltage}, pfsense_cur_{current,power,voltage}, socket_cur_{current,power,voltage}
  - Device state (4): lamp_switch_led, switch_channel_1..3
  - Target: ground_truth (jumlah orang 0-4, label manual dari kamera; YOLO26m hanya pre-annotasi)
- **Distribusi GT:** 0 = 224.721 (97,3%), 1 = 5.865 (2,5%), 2 = 346, 3 = 42, 4 = 2 → **sangat tidak seimbang** (wajib macro-F1/MCC/PR-AUC).
- **Missing values NATURAL:** sound 4.551 (2,0%), pfsense 1.085, socket 484, server 424, co2/humidity/temp_2/lux_2 91, lamp 86, temp_1/lux_1 26, switch ~11 → **bukti empiris missing modality di deployment nyata — validasi premis riset!**
- **Sinyal untuk eksperimen sensor failure:** temp_1 vs temp_2, lux_1 vs lux_2 (redundansi sensor), listrik terpisah per perangkat → simulasi dropout per-kanal mudah.

## Kandidat Model (baseline → proposed)

- **Baseline klasik:** RF, XGBoost, SVM
- **Baseline DL:** MLP, CNN, LSTM, GRU, CNN-LSTM, Transformer
- **Baseline fusion:** early, late, intermediate, gated fusion
- **State-of-the-art missing-modality:** MissModal, GMD (AAAI 2024), TCTR, GAUGE, Conformal Fusion
- **Proposed:** Modality-reliability estimation → cross-modal attention (dropout-tolerant) → contrastive learning → adaptive fusion

## Protokol Eksperimen (ringkas)

1. Single-modality → kontribusi tiap sensor
2. Fusion baseline (early/late/intermediate)
3. Proposed framework
4. Missing modality: dropout 10/30/50/70% + sensor failure per-jenis
5. Noise: Gaussian, impulse; Drift: sensor drift simulasi
6. Temporal misalignment: +1s/+5s/+10s/+30s
7. Cross-dataset: train A → test B/C/D
8. Ablation: tanpa contrastive, tanpa cross-attention, tanpa reliability
9. Metrik: Acc, Prec, Rec, F1, ROC-AUC, PR-AUC, Macro-F1, MCC, Balanced Accuracy
10. Analisis komputasi: latency, params, FLOPs (untuk opsi edge)

## Kandidat Jurnal Target (Q1)

| Jurnal | Kecocokan | Catatan |
|---|---|---|
| Information Fusion | Sangat tinggi (missing modality + fusion) | Perlu kontribusi metodologis kuat |
| Energy and Buildings | Tinggi (occupancy, energi) | Sudah banyak paper occupancy → butuh angle robustness |
| Building and Environment | Tinggi | Sama |
| IEEE Internet of Things Journal | Tinggi (IoT) | Cek scope & APC |
| Engineering Applications of AI | Sedang-tinggi | Review 2022 sudah ada |

---
*Dibuat oleh Jurnal-Kampret (Riset sub-agen) — 20-08-2026. Sumber: Crossref API (mailto: amesanggeng@buddhidharma.ac.id) & arXiv API. Quartile = estimasi, verifikasi via ScimagoJR sebelum sitasi.*
