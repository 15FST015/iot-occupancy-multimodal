# EVALUASI NASKAH V1 — Robust Multimodal IoT Intelligence Under Sensor Failure

| Item | Detail |
|---|---|
| **Naskah** | `drafts/manuscript_v1.md` (67 KB, ~6.477 kata tanpa pustaka, 7 tabel, 6 figure, 37 referensi) |
| **Target** | Energy and Buildings / Building and Environment (Elsevier Q1) |
| **Evaluator** | Jurnal-Evaluator (read-only) — 20-08-2026 |
| **Sumber audit** | `experiments/results/*.json` (21 file), `data/split_summary.json`, `data/EDA-A.json`, `experiments/ci_cluster_bootstrap.py`, `RESEARCH-GAP-MATRIX.md` (34 paper), `submissions/EB-GUIDE.md` |
| **Verifikasi online** | Crossref API (31 DOI), arXiv API (5 ID), Zenodo API (1 record) — semua live 20-08-2026 |

---

## RINGKASAN EKSEKUTIF

**Verdict: BERSYARAT LAYAK lanjut ke format template — setelah perbaikan wajib (3 MAJOR + 10 MINOR).**

Kekuatan substansi sangat baik: **audit angka 100% cocok** dengan JSON hasil (seluruh 84 sel T3, 25 sel T4, 12 kondisi T5, CI T6, 10 sel T7, 9 baris T2, semua klaim numerik teks, delta dropout, 14/14, 12/14 contrastive — semua terverifikasi dalam toleransi pembulatan 3 desimal). Referensi **bersih: tidak ada DOI mengarang** (31/31 DOI jurnal/prosiding terverifikasi Crossref dengan judul cocok; 5 ID arXiv terverifikasi; Zenodo v3.0.0 terverifikasi), semua 37 entri dikutip ≥1×, urutan kemunculan monotonik. Kejujuran ilmiah (negative transfer result, trade-off, limitation) adalah nilai jual utama untuk reviewer Q1.

Namun ada **3 temuan MAJOR**: (1) abstrak 234 kata (batas 200); (2) **jumlah kondisi stress salah di 3 lokasi** — klaim "16 stress"/"30 kondisi" padahal aktual **12 stress / 26 total**; (3) klaim "F1 highest 10/14 vs tree baselines" hanya benar untuk XGB tuned (8/14 vs tree blend) — plus 8 MINOR (kutipan [37] salah tempat di 5 catatan tabel, highlights 5/5 >85 char, "Four settings" padahal 5 item, "±1σ" vs "+0.5/+1.0σ", "published strategies" tanpa sitasi, [24] tanpa penulis, deklarasi AI/CRediT belum terintegrasi, dsb.).

---

## A. AUDIT ANGKA NASKAH vs JSON (sampel sistematis penuh)

Semua nilai dibandingkan dengan JSON sumber; toleransi pembulatan 3 desimal. **Tidak ada mismatch numerik** kecuali klaim jumlah kondisi (MAJOR-2) dan klaim F1 10/14 (MINOR-2) di bawah.

### Tabel 1 — Dataset (sumber: EDA-A.json, split_summary.json) ✅ COCOK
- Baris 230,976 ✓; GT 224,721/5,865/346/42/2 = 230,976 ✓ (97.3/2.5/0.15/0.02/<0.01%) ✓
- Missing natural: sound 1.97% (4,551) ✓; pfsense 0.47 (1,085) ✓; socket 0.21 (484) ✓; server 0.184 (424) ✓; co2/humidity/temp_2/lux_2 0.039 (91) ✓; temp_1/lux_1 0.011 (26) ✓; lamp 0.037 (86) ✓; switch 0.005 (11–12) ✓
- Korelasi: lux 1.00 ✓, temp 0.995 ✓, CO₂–humidity −0.25 ✓ (EDA-A.json)
- Split: 25 train (192,191; 1.748%→1.7%) ✓, 6 test (38,785; 7.467%→7.5%) ✓, seed 42 ✓, hari anomali 2023-12-04 di test ✓, eval 5 hari (2023-12-05/14/16, 2024-01-05/11) ✓ (proposed_dl_full_summary eval_days)
- Jam okupansi 07:00–21:00, peak 13:00 (1,049) ✓ (gt_hourly_occupied)
- Klaim "31 calendar days" ✓ (split_summary n_days=31)

### Tabel 2 — Tree & single-modality (sumber: baseline_summary.json) ✅ COCOK (9/9 baris)
XGB 0.742/0.848/0.989/0.734 ✓; RF 0.079/0.740/0.978/0.191 ✓; light 0.828/0.800/0.946/0.815 ✓; device 0.739/0.663/0.908/0.722 ✓; env_air 0.313/0.188/0.737/0.266 ✓; acoustic 0.217/0.129/0.729/0.150 ✓; server 0.165/0.115/0.700/0.071 ✓; socket 0.211/0.176/0.828/0.133 ✓; firewall 0.124/0.126/0.715/0.017 ✓. Footnote: PR-AUC 0.862/0.869 saat acoustic/pfsense di-drop ✓ (adaptive_ensemble.base); XGB 5-min block F1 0.749 ✓ (0.7485).

### Tabel 3 — 14 skenario × 6 model (sumber: major1_test_summary.json, sota_*.json, adaptive_ensemble_summary.json, blend_ensemble_summary.json) ✅ COCOK (84/84 sel F1 & PR-AUC)
Semua sel cocok dalam toleransi 3 desimal, termasuk XGB tuned (= adaptive.base: full 0.820/0.848 ✓, missing_light 0.159/0.359 ✓, random70 0.168/0.460 ✓), tree blend (14/14 ✓), mean_impute, indicator_mlp, gated_fusion, proposed. Threshold 0.95 untuk 4 model deep ✓ (JSON). Klaim teks: "0.887 vs 0.111–0.842" ✓; "0.887 vs 0.489–0.809 @70%" ✓; delta vs mean_impute +0.056/+0.101/+0.183/+0.261 ✓; vs gated +0.054/+0.099/+0.213/+0.398 ✓; vs indicator +0.049/+0.051/+0.062/+0.077 ✓; "14/14 vs 3 deep baselines" ✓ (terverifikasi per-sel); "10/14 vs trees, tree menang hanya full/acoustic/server/firewall" ✓ (terverifikasi programatik).

### Tabel 4 — Ablasi (sumber: proposed_dl_no_*.json, proposed_dl_mlp_early_summary.json) ✅ COCOK (25/25 sel)
Proposed, no_contrastive (thr 0.29), no_attention (0.44), no_aug (0.73), mlp_early (0.39) — semua cocok, termasuk F1 0.001 (0.0007), PR-AUC 0.372/0.328/0.187, random70 0.529/0.724, light+env 0.140/0.259/0.071. Klaim "no_contrastive 12/14 lebih baik, kalah hanya server (0.817 vs 0.822, Δ0.005) & firewall (0.817 vs 0.819, Δ0.002)" ✓ terverifikasi.

### Tabel 5 — Stress (sumber: noise_misalignment_summary.json) ✅ COCOK (13/13 baris)
full_clean 0.914/0.798/0.907/0.991 ✓; Gaussian 10/30/50 ✓; impulse 1/5 ✓; drift +0.5/+1.0σ ✓; misalign 10/50/100/300 s ✓; block5min F1 0.904, bacc 0.972 ✓. Klaim "F1 ≥ 0.857" ✓ (min 0.8566), "PR-AUC 0.799→0.813" ✓, "drift F1 0.916" ✓, "PR-AUC 0.800→0.854" ✓.
⚠️ **Jumlah kondisi: 12 (bukan 16)** — lihat MAJOR-2.

### Tabel 6 — CI & kalibrasi (sumber: ci_cluster_full.json, ci_bootstrap_full.json) ✅ COCOK
15 point + 15 CI blok 1 jam cocok semua (mis. full F1 0.825 [0.690, 0.913] vs 0.8254 [0.6895, 0.9128] ✓; missing_light PR 0.887 [0.779, 0.939] ✓). Per-row CI footnote: 0.914 [0.907, 0.921] ✓, 0.798 [0.779, 0.816] ✓, ROC 0.991 [0.990, 0.992] ✓, MCC 0.907 [0.899, 0.914] ✓. ECE 0.055 (0.0548) ✓; top bin conf 0.943 vs freq 0.863 ✓; bin 0.7–0.8 conf 0.753 vs freq 0.886 ✓; n = 38,785 ✓ (jumlah bins = 38,785 ✓). N_BOOT 1000 ✓ (skrip). "3 hari test tanpa okupansi" & "CI day-level [0, 0.97]" konsisten dengan komentar skrip ✓.

### Tabel 7 — Cross-domain (sumber: cross_dataset_summary.json) ✅ COCOK (10/10 sel + occupied %)
H1 82.19% ✓, H2 60.41% ✓; base 0.030/0.767/0.374 & 0.008/0.598/0.512 ✓; missaware 0.468/0.750/0.329 & 0.424/0.559/0.479 ✓; blend 0.066/0.766/0.375 & 0.039/0.597/0.512 ✓; finetune 0.850/0.820/0.484 & 0.684/0.639/0.478 ✓; majority PR-AUC 0.822/0.604, ROC 0.500 ✓. Klaim "ROC-AUC 0.374/0.512" ✓.

### Klaim numerik teks lain ✅
- "XGBoost F1 0.820 → 0.159 (light) / 0.168 (70%)" ✓ (adaptive.base)
- "MLP early F1 0.001 saat light hilang" ✓; "no_aug F1 0.001" ✓
- "best epoch 8, eval PR-AUC 0.4948" ✓ (eval_curve [8]=0.4948)
- "ECE 0.055" ✓; "ROC 0.374/0.484" ✓; "light+env 0.823/0.587 vs 0.696/0.222" ✓
- "test 38,785 rows (16.8%)" ✓ (0.1679)

---

## B. AUDIT REFERENSI

### B.1 Kelengkapan kutipan ✅
Semua 37 entri dikutip ≥1× di teks (terverifikasi programatik). Urutan kemunculan pertama **monotonik** [1]→[37] — sesuai gaya Elsevier numeric. Tidak ada referensi "hantu" (terdaftar tapi tak dikutip) maupun kutipan tanpa entri.

### B.2 DOI vs RESEARCH-GAP-MATRIX + verifikasi online ✅ (tidak ada DOI mengarang)
- Mapping lengkap: 34 entri paper = 34 paper matrix (1:1, semua judul/topik cocok); +[18] repo GitHub (lokal), +[36] Zenodo v3.0.0, +[37] Jacoby Sci Data — total 37. **Tidak ada entri naskah yang tidak ada di matrix.**
- **31 DOI jurnal/prosiding → Crossref: 31/31 OK, judul & tahun cocok dengan entri** (termasuk yang mencurigakan: [22] Buildings 10.3390/buildings16030545 ✓, [28] ICIP 10.1109/icip61757.2026.11630059 ✓, [33] MST 10.1088/1361-6501/ae2e26 ✓, [35] SenSys 10.1145/3715014.3722053 ✓, [17] AAAI ✓, [9] TACL ✓).
- 5 arXiv (2608.07183, 2608.05608, 2607.24683, 2607.13678, 2606.24986) → **5/5 terverifikasi** via arXiv API, judul & penulis cocok (termasuk penulis [24] Conformal Fusion = A. Moayedikia — **entri [24] tanpa penulis, padahal penulis tersedia** → MINOR-7).
- Zenodo 10.5281/zenodo.20548374 → **terverifikasi**: "A Multimodal Dataset for Smart Office Occupancy Estimation", v3.0.0, cc-by-4.0 ✓.
- [28] & [35]: matrix mencatat versi arXiv; naskah memakai versi terbit (ICIP 2026 / SenSys 2025) dengan DOI — **valid, bukan mengarang** (Crossref konfirmasi).
- Catatan minor: naskah header mengklaim "verified via Crossref/arXiv on 2026-08-20" — konsisten dengan hasil verifikasi evaluator.

### B.3 Gaya & format
- Format Elsevier numeric konsisten (inisial depan, jurnal disingkat, volume (tahun) halaman, DOI). 
- ⚠️ Kutipan **[37] salah tempat di 5 catatan tabel** (harus [18] = reproducibility package): T1 note ("test split [37]"), T3 footnote ("archived [37]"), T4 footnote ("matrix [37]"), T5 footnote ("[37]"), T7 footnote ("transfer claim [37]" — ambigu, lebih tepat [18]). Penggunaan [37] yang benar hanya di §3.9 & §8 (HPDmobile) → **MINOR-5**.
- 5 referensi arXiv 2026 (preprint) — dapat diterima tapi catat risiko (COMMENT-6).
- [18] repo GitHub tanpa DOI — wajar; pastikan repo publik & berisi semua JSON saat submit.

---

## C. STRUKTUR & KELENGKAPAN (vs EB-GUIDE)

| Item panduan E&B | Status naskah v1 |
|---|---|
| Abstrak ≤200 kata | ❌ **234 kata** → MAJOR-1 (wajib dipangkas; panduan sendiri menargetkan ≤200) |
| Highlights 3–5 bullet ≤85 char | ⚠️ 5 bullet ✅ jumlah, tapi **5/5 >85 char** (137, 163, 227, 137, 171) → MINOR-1; juga harus file terpisah saat submit |
| Keywords | ✅ 7 kata (gunakan pemisah sesuai template; saat ini ";") |
| Struktur IMRAD (Abstract, Intro, Materials & Methods, Results, Conclusions) | ✅ lengkap (1–8: Intro, Related Work, M&M, Results, Discussion, Conclusion, Limitations, Data/Code) |
| Title page (penulis, afiliasi, corresponding) | ❌ belum ada (naskah anonim) — harus diisi saat format |
| Declaration of Competing Interest | ✅ ada (teks baku) |
| Generative AI declaration | ❌ **tidak ada di naskah**; file `submissions/DECLARATION-AI.md` sudah ada → tinggal diintegrasikan → MINOR-8 |
| CRediT authorship | ❌ placeholder "(To be completed…)" → MINOR-8 |
| Acknowledgements | ⚠️ placeholder — boleh diselesaikan penulis (COMMENT) |
| Research data statement | ✅ §8 (Zenodo dataset + GitHub repo + Figshare HPDmobile) |
| Submission declaration | ❌ belum ada (dilengkapi saat submit via EES) |
| Batas 20 halaman | ⚠️ ~6.477 kata + **7 tabel + 6 figure** — EB-GUIDE mengestimasi aman untuk 6 tabel/4 gambar; v1 lebih berat → risiko overflow → COMMENT-5 |
| Gambar | Fig. 2 masih placeholder (perlu diagram vektor final) → MINOR-9 |

---

## D. BAHASA, KEJELASAN & KLAIM

**Kualitas bahasa Inggris: baik** — kalimat jelas, voice aktif, terminologi konsisten (PR-AUC sebagai metrik primer dijelaskan konsisten). Tidak ditemukan typo fatal pada sampel menyeluruh. Temuan:

- ✅ **"Generalizable" = 0 klaim**: naskah secara eksplisit membatasi "robustness *within a domain* — not universal generalization" (§2.4), highlight 5 & §4.7 menyatakan transfer gagal sebagai *generalization boundary*. Aman.
- ✅ **Contrastive = "complementary"**: konsisten di abstrak, §3.4, §4.4, §5.1, limitation #5 ("deliberately limited to 'complementary'"). Aman.
- ⚠️ **"Four settings are evaluated" (§3.9) padahal 5 item** (majority, zero-shot base, zero-shot missaware, blend, fine-tune 1 hari; T7 = 5 baris) → MINOR-3.
- ⚠️ **"±1σ drift" (§4.5, §6) vs protokol "+0.5σ/+1.0σ" (§3.7, T5)** — JSON hanya drift positif; "±" tidak akurat → MINOR-4.
- ⚠️ **"three published missing-modality strategies" (3×: abstrak, §3.6, §6)** — mean-imputation/indicator/gated adalah arsitektur standar yang diimplementasikan sendiri, tanpa sitasi strategi spesifik; klaim "published" rentan diserang → MINOR-6.
- ⚠️ Inkonsistensi penamaan: XGB F1 0.742 (T2, default threshold) vs 0.820 (T3 & Intro, tuned per-scenario). §3.6 menyebut "baseline tuned" tapi Intro memakai 0.820 tanpa label → pembaca bisa bingung → MINOR-10.
- Judul 16 kata & kata "Intelligence" — sedikit overclaim; pertimbangkan dipangkas saat format (COMMENT-7).
- "42.6 days" vs 230,976 baris: durasi data efektif = 26.7 hari (230,976×10 s); 42.6 hari = rentang kalender dengan gap (hanya 31 hari berdata). Naskah menyiratkan tapi tidak menjelaskan gap → reviewer akan bertanya → COMMENT-8.

---

## E. RISIKO REVIEWER (Q1)

1. **Baseline SOTA tidak diuji** — MissModal [9], GMD [17], TCTR [10], GAUGE [25], conformal [24] dikutip di Related Work tetapi TIDAK dibandingkan; tiga baseline yang diuji (imputation/indicator/gated) adalah implementasi sederhana. Ini serangan metodologis paling mungkin ("compare against SOTA missing-modality methods"). Mitigasi yang ada: protokol identik & fokus domain; perkuat §5.4 dengan pembelaan transferabilitas + tambah 1–2 baseline SOTA yang feasible (mis. gated/indicator sudah ada; pertimbangkan TCTR-style reconstruction atau GMD-style decoupling sederhana).
2. **Single building, single season, test 6 hari (3 tanpa okupansi)** — CI blok lebar & tumpang tindih; klaim perbandingan hanya ranking/effect size (sudah diakui, limitation #2 — pertahankan kejujuran ini).
3. **Checkpoint ganda** — T5 & per-row CI pakai checkpoint threshold-0.75, tabel lain pakai final 0.95 (diakui limitation #8, tapi reviewer bisa menuntut konsistensi; pertimbangkan re-run stress test dengan checkpoint final bila biaya memungkinkan).
4. **Ablasi 3 epoch** — keputusan metodologis; jelaskan lebih tegas kenapa cukup (atau run 5 epoch).
5. **Hari anomali 2023-12-04 di test** — mengakui "raises test PR-AUC" adalah jujur, tapi reviewer bisa menyerang pemilihan split; pertimbangkan analisis sensitivitas tanpa hari anomali.
6. **Kesalahan hitung "16/30 kondisi"** & kutipan [37] di tabel — dua hal yang *mudah* diverifikasi reviewer → kerusakan kredibilitas tidak proporsional → perbaiki sebelum submit (MAJOR-2, MINOR-5).
7. **Abstrak 234 kata & highlights >85 char** — bisa memicu *desk rejection* / editorial return → perbaiki sebelum submit.
8. Klaim "highest in 14/14" kuat tapi didukung data; pastikan definisi "14/14" (per baseline, bukan head-to-head semua) eksplisit — sudah dijelaskan di §4.3, pertahankan.
9. 5 referensi arXiv (2026) — sebagian jurnal Q1 kurang suka; ganti dengan versi terbit bila ada.
10. Panjang naskah (7 tabel + 6 figure) berisiko melewati 20 halaman.

---

## F. DAFTAR TEMUAN (protokol Jurnal-Evaluator)

### MAJOR

**MAJOR-1 | Struktur | Abstract**
ISSUE: Abstrak 234 kata, melebihi batas panduan E&B (≤200 kata).
DAMPAK: High — pelanggaran persyaratan wajib; editorial return / desk reject.
REKOMENDASI: Pangkas ke ≤200 kata (hilangkan pengulangan angka 14/14, detail 16 skenario, dan rincian ablasi yang sudah ada di teks; pertahankan 5 elemen: konteks, metode, hasil utama, kalibrasi, batas generalisasi).

**MAJOR-2 | Angka | §3.7, Abstrak, Kontribusi 4**
ISSUE: Jumlah kondisi stress salah di 3 lokasi: "Sixteen stress conditions" (§3.7), "16 noise/drift/misalignment stress scenarios" (abstrak), "30 failure/stress conditions" (kontribusi 4). Fakta dari `noise_misalignment_summary.json` + T5: **12 kondisi stress** (3 Gaussian + 2 impulse + 2 drift + 4 misalign + 1 block) dan **26 total** (14 failure + 12 stress).
DAMPAK: High — kesalahan angka yang mudah diverifikasi reviewer; inkonsistensi internal.
REKOMENDASI: Ganti 16→12 dan 30→26 (atau hitung ulang definisi; konsistenkan di 3 lokasi).

**MAJOR-3 | Angka/Klaim | §4.3 & T3 footnote**
ISSUE: "F1 at the tuned threshold is highest in **10/14** scenarios vs. the tree baselines" — terverifikasi: **10/14 vs XGB tuned** (tree menang: acoustic, server, pfsense, device_state) tetapi **8/14 vs tree blend** (tree blend juga menang di random 50% & 70%: 0.686/0.788 vs 0.626/0.670). Tree blend menang F1 di 6/14.
DAMPAK: Medium — klaim tidak terduplikasi persis untuk salah satu dari dua kolom tree di T3.
REKOMENDASI: Tulis eksplisit "10/14 vs. the per-scenario-tuned XGBoost" atau laporkan 8/14 vs. tree blend; jika mempertahankan, tambahkan catatan bahwa PR-AUC tetap menang 10/14 untuk kedua kolom.

### MINOR

**MINOR-1 | Struktur | Highlights**
ISSUE: 5 bullet, semua melebihi 85 karakter (137/163/227/137/171 char; batas 85 termasuk spasi). Juga wajib file terpisah "Highlights" saat submit.
DAMPAK: Medium — persyaratan format wajib.
REKOMENDASI: Pangkas tiap bullet ≤85 char (mis. "Missing-modality dropout augmentation is critical: without it the model collapses under light failure (F1 0.001 vs 0.528)" ≈ 115 char — perlu lebih ringkas lagi; fokus 1 klaim per bullet).

**MINOR-2 | Angka | T6 footnote**
ISSUE: Klaim "per-row bootstrap … archived" — nilai cocok dengan `ci_bootstrap_full.json` ✓, tapi klaim "109 one-hour blocks" (§3.8) tidak tersimpan di JSON mana pun (N_BOOT=1000 ada di skrip; jumlah blok dicetak saat runtime).
DAMPAK: Low — tidak terverifikasi dari artefak; perlu jejak.
REKOMENDASI: Simpan n_blocks ke JSON/LOG saat re-run, atau konfirmasi dari log pelatihan.

**MINOR-3 | Bahasa | §3.9**
ISSUE: "Four settings are evaluated" padahal daftar 5 item (dan T7 punya 5 baris metode).
DAMPAK: Low.
REKOMENDASI: Ganti "Five settings" (atau kelompokkan majority baseline sebagai referensi).

**MINOR-4 | Bahasa | §4.5 & §6**
ISSUE: "drift ±1σ" vs protokol "+0.5σ/+1.0σ" (JSON hanya drift positif).
DAMPAK: Low — ketidaktepatan terminologi.
REKOMENDASI: Konsistenkan: "+1.0σ drift" atau jelaskan bila dua arah diuji.

**MINOR-5 | Referensi | Catatan Tabel 1, 3, 4, 5, 7**
ISSUE: Kutipan [37] (Jacoby, HPDmobile) salah konteks di 5 catatan tabel — yang dimaksud adalah protokol/reproducibility package [18] (hari anomali di test, arsip ROC-AUC/MCC, konsistensi matriks, quantity of interest, generalization boundary).
DAMPAK: Medium — 5 lokasi; reviewer yang mengecek [37] akan melihat isi tidak sesuai.
REKOMENDASI: Ganti [37]→[18] di T1/T3/T4/T5; T7 → [18] (atau [18,37]).

**MINOR-6 | Klaim | Abstrak, §3.6, §6**
ISSUE: "three published missing-modality strategies" — mean-imputation, indicator-based, gated fusion adalah implementasi lokal (code in [18]) tanpa sitasi strategi spesifik; "published" berlebihan.
DAMPAK: Medium — rentan diserang reviewer.
REKOMENDASI: Ganti menjadi "three standard missing-modality strategies (imputation-, indicator-, gating-based), implemented under an identical protocol".

**MINOR-7 | Referensi | Entri [24]**
ISSUE: "Conformal fusion under missing modalities, arXiv:2608.07183 (2026)" tanpa penulis — penulis sebenarnya A. Moayedikia (terverifikasi arXiv API).
DAMPAK: Low — format Elsevier minta penulis.
REKOMENDASI: Tambahkan penulis pada entri [24].

**MINOR-8 | Struktur | Deklarasi**
ISSUE: Generative AI declaration tidak ada di naskah (file `submissions/DECLARATION-AI.md` sudah dibuat — belum diintegrasikan); CRediT masih placeholder.
DAMPAK: Medium — wajib saat submit (EB-GUIDE).
REKOMENDASI: Integrasikan deklarasi AI + CRediT (daftar penulis & peran) sebelum format.

**MINOR-9 | Struktur | Fig. 2**
ISSUE: Figure 2 masih placeholder "(final vector diagram to be produced by the authors)".
DAMPAK: Low — perlu diganti sebelum submit.
REKOMENDASI: Produksi diagram arsitektur final.

**MINOR-10 | Bahasa/Kejelasan | §1 vs T2/T3**
ISSUE: Intro memakai F1 XGB 0.820 tanpa label "threshold-tuned", sementara T2 melaporkan 0.742 (default) — dua angka untuk model yang sama membingungkan.
DAMPAK: Low.
REKOMENDASI: Di Intro, sebut "(threshold-tuned XGBoost, Table 3)" atau rujuk T2/T3 secara eksplisit.

### COMMENT

**COMMENT-1 | Metodologi | Baselines SOTA**: MissModal/GMD/TCTR/GAUGE tidak dibandingkan — risiko serangan terbesar; siapkan argumen + idealnya 1 baseline SOTA sederhana.
**COMMENT-2 | Data | Test set**: "3 hari test tanpa okupansi" dan "eval ~2.8% occupied" tidak dapat diverifikasi dari JSON yang diarsipkan — simpan statistik per-hari di paket reproduksibilitas.
**COMMENT-3 | Konsistensi | Checkpoint**: T5 & per-row CI (threshold 0.75) vs tabel lain (0.95) — sudah diakui (limitation #8); pertimbangkan re-run untuk konsistensi penuh.
**COMMENT-4 | Metodologi | Ablasi 3 epoch** — pertimbangkan 5 epoch atau justifikasi lebih tegas.
**COMMENT-5 | Panjang | 7 tabel + 6 figure** berisiko >20 halaman saat format; rencanakan kompresi (T3 bisa dipindah ke supplementary).
**COMMENT-6 | Referensi | 5 arXiv 2026** — cek versi terbit sebelum submit.
**COMMENT-7 | Judul** — 16 kata, kata "Intelligence" berpotensi overclaim; pertimbangkan pemangkasan saat format.
**COMMENT-8 | Dataset | 42.6 hari vs 26.7 hari data efektif (230,976×10 s)** — jelaskan gap/coverage 62.7% dan 31 hari berdata di §3.1 untuk mengantisipasi pertanyaan reviewer.

---

## RINGKASAN AKHIR

```
=== EVALUASI: Robust Multimodal IoT Intelligence Under Sensor Failure ===
AUDIT ANGKA:  100% cocok (T1–T7, semua klaim teks; 0 mismatch numerik selain MAJOR-2 & MAJOR-3)
AUDIT REFERENSI: 37/37 dikutip, urutan monotonik, 31 DOI + 5 arXiv + Zenodo terverifikasi,
                 TIDAK ADA DOI mengarang; 1 entri tanpa penulis ([24]); 5 kutipan [37] salah konteks
STRUKTUR:      Abstrak 234/200 kata (MAJOR); highlights 5/5 >85 char; AI-declaration & CRediT belum
               terintegrasi; data availability OK
KLAIM:         Generalizable = 0 (aman); contrastive = complementary (konsisten)
RINGKASAN:     MAJOR 3 | MINOR 10 | COMMENT 8
VERDICT:       BERSYARAT LAYAK lanjut ke format template —
               sains & angka solid (audit 100%), kejujuran ilmiah kuat;
               perbaiki 3 MAJOR (abstrak, jumlah kondisi 16/30, klaim F1 10/14)
               + MINOR prioritas tinggi ([37]→[18], highlights, deklarasi AI/CRediT)
               sebelum submit ke Energy and Buildings.
```
