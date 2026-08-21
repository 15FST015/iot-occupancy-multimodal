# Verifikasi Jurnal Kandidat (20-08-2026) — untuk paper Occupancy Multimodal

Metode: curl situs resmi + grep (Scopus sourceid, APC, ISSN); ScimagoJR 403 dari host ini,
Sinta HTTP 000, Wayback/CDX gagal → **quartile tidak dapat diverifikasi dari host ini**;
status per item dicantumkan jujur.

## Ringkasan

| Jurnal | Scopus | APC | Verdict verifikasi |
|---|---|---|---|
| JIKI UI (UI) | ❓ TIDAK terbukti (0 badge/sourceid di situs) | ✅ GRATIS ("no charges for submitted or published articles") | Klaim "Q3 Scopus" TIDAK didukung; Sinta 2 terverifikasi |
| JoICT (ITB) | ✅ TERVERIFIKASI (sourceid 21100268428 di about page) | ❓ belum terbukti (halaman tidak menampilkan; umumnya gratis) | Scopus ✅; quartile belum |
| IJEECS (IAES) | ❓ belum (0 sourceid di halaman dicek) | ✅ **USD 225.00** (about/submissions) | Berbayar terverifikasi; Scopus perlu cek manual |
| IJAIN (UAD) | ✅ TERVERIFIKASI (sourceid 21100890645, CiteScore Tracker 2026 aktif) | ❓ tidak ditemukan biaya di situs (mendukung klaim gratis) | Scopus ✅; quartile belum |

## Detail per jurnal

### 1. Jurnal Ilmu Komputer dan Informasi (JIKI UI) — jiki.cs.ui.ac.id
- HTTP 200 ✅ · ISSN 2088-7051 (print) / 2502-9274 (online) ✅
- Sinta: link resmi "Sinta 2" (sinta.kemdiktisaintek.go.id) di situs ✅
- APC: halaman resmi jiki/apc → "no charges for submitted or published articles in our journal" ✅ GRATIS
- Scopus: **TIDAK ditemukan** badge/sourceid Scopus di halaman index/about → klaim "Q3" TIDAK terbukti
- KESIMPULAN: gratis & Sinta 2 terverifikasi; **kemungkinan besar TIDAK di Scopus** — verifikasi manual
  Scopus preview (cari "Jurnal Ilmu Komputer dan Informasi") sebelum dijadikan target

### 2. Journal of ICT Research and Applications (JoICT ITB) — journals.itb.ac.id/index.php/jictra
- HTTP 200 ✅ · ISSN 2337-5787 (print) / 2338-5499 (online) ✅
- Scopus: link resmi scopus.com/sourceid/21100268428 di halaman about ✅ TERINDEKS SCOPUS
- APC: tidak ditemukan di halaman about (jurnal ITB umumnya gratis — belum diverifikasi)
- KESIMPULAN: Scopus ✅; klaim Q3 masuk akal (tidak diverifikasi di sini); cek quartile di ScimagoJR manual

### 3. Indonesian Journal of Electrical Engineering and Computer Science (IJEECS) — ijeecs.iaescore.com
- HTTP 200 ✅ (domain baru ijeecs.iaescore.com; iaescore.com/journal/index.php 404) · ISSN 2502-4752 / 2502-4760 ✅
- APC: about/submissions → "Publication Fee (APF/APC): 225.00 (USD)" ✅ BERBAYAR USD 225
- Scopus: sourceid tidak ditemukan di halaman yang dicek → perlu cek manual (IJEECS dikenal terindeks)
- KESIMPULAN: berbayar USD 225 terverifikasi; Scopus/quartile perlu konfirmasi manual

### 4. International Journal of Advances in Intelligent Informatics (IJAIN UAD) — ijain.org
- HTTP 200 ✅ · Scopus: link resmi scopus.com/sourceid/21100890645 + "SCOPUS CiteScore Tracker 2026 ACCEPTED" ✅ TERINDEKS SCOPUS
- APC: tidak ditemukan biaya di halaman index/about ✅ (mendukung klaim gratis — konfirmasi manual)
- KESIMPULAN: Scopus ✅ terindeks aktif; klaim Q2 perlu konfirmasi ScimagoJR manual (umumnya benar, CiteScore tracker aktif)

## Rekomendasi untuk paper occupancy

1. **Target utama: TETAP Energy and Buildings (Q1)** — submission sedang diformat (T14)
2. **Cadangan #1: IJAIN (UAD)** — Scopus terindeks ✅, gratis, scope AI/informatika cocok (paper multimodal fusion)
3. **Cadangan #2: JoICT (ITB)** — Scopus terindeks ✅, ICT applied research cocok
4. **Cadangan #3: IJEECS** — berbayar USD 225; hanya jika opsi lain tidak layak
5. **JIKI UI: TURUNKAN** — klaim Q3 tidak terbukti (Scopus tidak ditemukan); hanya cocok jika target Sinta

## Tindak lanjut (butuh browser manusia — ScimagoJR/scopus.com memblokir host ini)
- [ ] Konfirmasi quartile IJAIN (scimagojr.com/journalsearch.php?q=21100890645) — klaim Q2
- [ ] Konfirmasi quartile JoICT (q=21100268428) — klaim Q3
- [ ] Konfirmasi quartile IJEECS + Scopus sourceid
- [ ] Konfirmasi status Scopus JIKI UI (cari di scopus.com/source — mungkin TIDAK terindeks)
