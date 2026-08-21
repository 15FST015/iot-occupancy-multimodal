# Cover Letter — Energy and Buildings (DRAFT)

*(Salin ke email/EES; isi [placeholder] sebelum kirim)*

---

**Date:** [tanggal submit]

**To:** The Editor-in-Chief
Energy and Buildings (Elsevier)

**Subject:** Submission of Original Research Article — "Robust Multimodal Occupancy Detection Under Sensor Failure: Cross-Modal Attention with Missing-Modality Augmentation"

Dear Editor,

We are pleased to submit our original research article entitled **"Robust Multimodal Occupancy Detection Under Sensor Failure: Cross-Modal Attention with Missing-Modality Augmentation"** for consideration for publication in *Energy and Buildings*.

**Relevance to the journal.** Occupancy detection is a cornerstone of building energy management (demand-controlled ventilation, HVAC scheduling, and lighting control). Yet real deployments consistently suffer from sensor failures — a single failed light sensor can collapse an occupancy model (F1 drops from 0.82 to 0.16 for a standard XGBoost baseline in our study). Our work directly addresses this reliability gap that limits the practical deployment of occupancy-driven energy savings.

**What we contribute:**
1. A missing-modality-aware deep fusion framework (per-modality encoders + masked cross-modal attention + contrastive learning) that maintains accuracy when sensors fail;
2. Empirical evidence that missing-modality dropout augmentation is the critical mechanism for robustness (without it, the model collapses: F1 0.001 vs 0.856 under dominant-modality failure);
3. The most comprehensive robustness evaluation protocol in this niche (26 failure/stress conditions: per-modality dropout, random channel dropout 10–70%, multi-failure, Gaussian/impulse noise, drift, temporal misalignment), with bootstrap CIs and calibration;
4. A fair comparison against three standard missing-modality strategies (imputation-, indicator-, and gating-based): our method wins PR-AUC in 14/14 scenarios, with the margin growing under increasing failure rates;
5. An honest domain-shift analysis (office→residential zero-shot transfer failed; structural causes identified) that delineates the generalization boundary.

**Key results.** On a public smart-office dataset (Zenodo, 230,976 samples, 42.6 days, 22 channels): PR-AUC 0.824 (full health) and 0.887 under dominant-modality failure — versus 0.135–0.842 for the best standard baselines — with calibration error 0.055. All code, scripts, and results are publicly available (https://github.com/15FST015/iot-occupancy-multimodal) for full reproducibility.

**Declarations.** This manuscript is original, has not been published previously, and is not under consideration elsewhere. All authors approve the submission. We declare no competing interests. AI-assisted tools were used in the writing/analysis process and are disclosed in the manuscript per Elsevier's policy. The dataset is publicly available under CC-BY-4.0.

Thank you for your consideration.

Sincerely,

Amesanggeng Pataropura, M.Kom.
Universitas Buddhi Dharma, Jalan Imam Bonjol No. 41, Kota Tangerang, Banten 15115, Indonesia
Email: amesanggeng@buddhidharma.ac.id
ORCID: 0009-0007-4950-0769

---

# Checklist Submission EES — Energy and Buildings (T15)

## File yang disiapkan
- [ ] Manuscript: `drafts/manuscript_E&B_v2.docx` (≤20 halaman; abstrak ≤200 kata)
- [ ] Highlights: 5 bullet ≤85 char (halaman 1 docx — atau file terpisah bila diminta sistem)
- [ ] Figures: 6 PNG (fig1_arsitektur, fig_gt_distribution, fig_training_curve, fig_reliability, fig_prauc_scenarios, fig_prauc_missing_rate)
- [ ] Tables: inline di manuscript (7; T3 = Supplementary Table S1)
- [ ] Supplementary Material: Table S1 (di akhir docx)
- [ ] Cover letter (draf di atas — identitas terisi)
- [x] Title page: drafts/TITLE-PAGE.md (Amesanggeng Pataropura, UBD, ORCID 0009-0007-4950-0769 terverifikasi; afiliasi lengkap + alamat + funding [konfirmasi])
- [ ] CRediT authorship statement (isi nama per peran)
- [ ] Declaration of Competing Interest (PDF teks baku Elsevier — sudah ada di naskah)
- [ ] Generative AI declaration (sudah di naskah §Declaration)
- [ ] Research Data statement (sudah di naskah §Data Availability)

## Sebelum submit (verifikasi akhir)
- [ ] Abstrak ≤200 kata (v2: 192 ✅)
- [ ] Highlights ≤85 char (✅ 66-67)
- [ ] Referensi: 37 entri, semua dikutip, DOI valid (✅ audited)
- [ ] "Generalizable" = 0 kemunculan (✅)
- [ ] Identitas penulis & afiliasi DIISI di title page + sistem
- [ ] Metadata sistem: title, abstract (copy), keywords, authors, ORCID
- [ ] Suggested reviewers: [isi 3-5 nama bidang smart building/occupancy/multimodal — opsional]
- [ ] Statement: tidak sedang dipertimbangkan jurnal lain
- [ ] File PDF/Word final hasil verifikasi halaman ≤ 20

## Setelah submit
- [ ] Catat nomor manuskrip (EES)
- [ ] Arsip konfirmasi + tanggal ke STATUS-JURNAL.md
- [ ] Tunggu editorial decision (umumnya 1-3 bulan E&B)
