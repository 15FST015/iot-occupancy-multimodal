# Identifikasi Variabel — Occupancy Multimodal

## Variabel Bebas (Independen) — 22 kanal sensor, 7 grup modalitas
| Grup | Kanal | Tipe |
|---|---|---|
| env_air | co2, humidity, temperature_1, temperature_2 | numerik |
| light | lux_1, lux_2 | numerik |
| acoustic | sound | numerik |
| device_state | lamp, switch | biner |
| elec_socket | socket_power | numerik |
| elec_server | server_power | numerik |
| elec_pfsense | pfsense_power | numerik |
+ fitur waktu: hour_sin, hour_cos, weekend

## Variabel Tergantung (Dependen)
- Status okupansi ruangan: GT 0-4 orang → biner (0 kosong / 1 terisi); 97,3% kosong (imbalance ekstrem)

## Variabel Pengendali (Kontrol)
- Split day-level stratified (test 20% hari = 6 hari/38.785 baris; eval 5 hari stratified)
- Seed 42 · z-score train-only (anti-leakage) · NaN → 0 + flag presence · protokol skenario identik antar model

## Artefak
- data/EDA-A.json · data/SPLIT-DESIGN.md · data/scaler.json · data/prep_split.py
