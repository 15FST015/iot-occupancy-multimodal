# Model Konseptual — Occupancy Multimodal

## Alur konseptual
Sensor (IV: 22 kanal/7 modalitas) → normalisasi z-score + presence flag
→ encoder per-modalitas (MLP) → masked cross-modal attention (key-padding mask
utk modalitas hilang) → [CLS] → head → prediksi okupansi (DV)

## Komponen pelengkap
- InfoNCE clean↔corrupted (λ=0.1) — penguat representasi
- Dropout augmentation 30% (1 grup di-mask) — mekanisme inti robustness
- Baseline pembanding: imputation-, indicator-, gated-based (fair, protokol identik)

## Diagram
- experiments/figures/fig1_arsitektur.png (di naskah = Fig. 2)

## Artefak
- experiments/proposed_dl.py (implementasi penuh)
