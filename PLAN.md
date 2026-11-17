# LookTwice — Rencana Kerja Final Project

Dasar: `Proposal Deep Learning NEW.pdf` + brief resmi `Form Submit Final Project COMP6826001.docx` (AY 2026/2027).
`Final project.docx` adalah template lama (AY 2025/2026, ada demo video & bobot 40/40/20) — **pakai rubrik Form Submit**, kecuali dosen bilang lain.

---

## 1. Apa yang sebenarnya dinilai

| Kriteria | Bobot | Artinya buat LookTwice |
|---|---|---|
| Data Preparation & Experimental Workflow | 20% | Pipeline ekstraksi frame → face crop → split per aktor yang **reproducible** (script, bukan klik manual) |
| DL Model Construction | 20% | Jelaskan arsitektur, pretraining (ImageNet), strategi fine-tuning, loss, optimizer — **bukan black box** |
| Model Behavior & Performance Analysis | 20% | Kurva train/val loss & metric, overfitting, efek tiap variabel eksperimen, **error analysis** |
| Real-World Suitability & Limitations | 15% | Robustness (kompresi/blur), latency, ukuran model, hardware, keterbatasan |
| Working Application | 15% | Extension + FastAPI yang **benar-benar jalan** pakai model hasil training sendiri |
| AI Usage, Verification & Reflection | 10% | **AI Usage Log** (tool, tujuan, komponen, cara verifikasi) — mulai dicatat dari sekarang |

Prinsip brief: bukan "akurasi setinggi mungkin", tapi bisa jelaskan *kenapa model ini, bagaimana perilakunya, kapan gagal, dan cocok tidak untuk deployment*.

---

## 2. Hal di proposal NEW yang perlu disesuaikan saat implementasi

Proposal NEW belum memuat perbaikan dari `Catatan Revisi Proposal LookTwice.md`. Yang paling berpengaruh ke nilai:

1. **Skenario A/B/C mengubah semua parameter sekaligus.** Brief (bagian 10) minta setiap eksperimen jelas *apa yang diubah, kenapa, efeknya apa*. Solusi: tetap jalankan A/B/C sebagai "training recipe", lalu tambah **ablation satu variabel** di model terbaik (lihat §5).
2. **Cosine `T_max = 20` padahal 25 epoch** → set `T_max` = jumlah epoch.
3. **Normalisasi**: ikuti checkpoint pretrained masing-masing (Xception di `timm` pakai mean/std 0.5, bukan ImageNet).
4. **Evaluasi per video** (rata-rata probabilitas frame), bukan hanya per frame.
5. **Threshold & pita "Uncertain"** (Amber) ditentukan dari validation set.
6. **Badge smoothing**: moving average ~10 frame terakhir supaya badge tidak berkedip.
7. **Donor leakage**: video fake berisi wajah aktor target *dan* donor (lihat §4 step 1).
8. Brief bilang mockup/static interface **bukan** working app — halaman demo boleh, asal extension + backend + model beneran jalan.

---

## 3. Kondisi mesin & keputusan infrastruktur

- Laptop: RTX 3050 6 GB, **disk C tinggal ~11 GB**, PyTorch belum terinstall.
- Dataset DFD kemungkinan besar puluhan GB → **jangan download ke laptop**.

**Rekomendasi:** preprocessing + training di **Kaggle Notebooks** (dataset sudah ada di Kaggle, GPU gratis ~30 jam/minggu). Output preprocessing (≈12–13 ribu face crop JPG, beberapa ratus MB) disimpan sebagai Kaggle Dataset sendiri, lalu dipakai semua notebook training. Laptop hanya untuk: backend FastAPI, ONNX inference, Chrome extension, pengukuran latency.

---

## 4. Tahapan kerja

### Tahap 0 — Setup (hari 1–2)
- [ ] Repo GitHub (private) dengan struktur di §7, semua anggota jadi collaborator
- [ ] Buka dataset di Kaggle, **cek format nama file** (dugaan: original `01__scene.mp4`, fake `01_02__scene__XXXX.mp4` = target 01, donor 02) — ini menentukan split
- [ ] Mulai `docs/ai_usage_log.md`

### Tahap 1 — Data pipeline (minggu 1) → LO-1
1. **Metadata**: parse semua video → tabel `video, label, target_actor, donor_actor, scene`.
2. **Split per aktor** 20/4/4. Hitung dulu berapa video fake tersisa per split kalau aturan "target & donor harus satu split" dipakai — dengan 4 aktor di test, kemungkinan sisa fake-nya sedikit sekali. Kalau terlalu sedikit: split berdasarkan target actor, dan donor leakage dicatat sebagai limitation (atau dilaporkan sebagai subset analysis).
3. **Ekstraksi frame**: real ~18 frame/video, fake ~2 frame/video, diambil merata sepanjang video.
4. **Face crop**: MediaPipe/RetinaFace, bbox +20%, wajah terbesar saja, simpan JPG kualitas tinggi + CSV (path, label, video, actor, split).
5. **EDA untuk laporan**: distribusi kelas per split, contoh crop, crop yang gagal terdeteksi, resolusi/kualitas.

### Tahap 2 — Training & eksperimen (minggu 2) → LO-2, LO-3
Satu `train.py` + config YAML per run, semua log (loss/metric per epoch) ke CSV/W&B. Pakai `timm`: `resnet50`, `xception` (legacy_xception), `efficientnet_b4`. Head 1 logit, BCE + label smoothing 0.05, AMP.

### Tahap 3 — Evaluasi & analisis (minggu 2–3) → LO-3, LO-4
- Test set **dipakai sekali** di akhir untuk model terpilih + pembanding
- Metric: Acc, Precision, Recall, F1, AUC-ROC, confusion matrix — per frame & per video
- Robustness: test ulang dengan JPEG quality 90/70/50/30, downscale, blur
- Error analysis: kumpulkan false positive/negative, cari pola (pose miring, gelap, blur, crop gagal)
- Grad-CAM beberapa contoh (nilai plus untuk explainability)
- Tabel trade-off: params, ukuran file, training time, latency CPU/GPU, ONNX vs PyTorch

### Tahap 4 — Aplikasi (paralel mulai minggu 2) → LO-2, LO-4
- **Backend FastAPI**: `POST /predict` (JPEG frame) → face detect → crop → ONNX Runtime → `{verdict, prob, face_found, latency_ms}`
- **Chrome extension (MV3)**: content script cari `<video>`, capture canvas 2 FPS, kirim ke backend lewat service worker, overlay badge Hijau/Amber/Merah + "no face detected"
- **Demo page** di localhost: video dari aktor test (tidak pernah dilihat saat training), served same-origin supaya canvas tidak ter-*taint* oleh CORS
- Bisa mulai pakai model dummy/ResNet awal, ganti ke model final belakangan

### Tahap 5 — User testing (minggu 3)
- 10 klip (5 real + 5 fake) dari aktor test, informed consent
- Tanpa extension vs dengan extension; idealnya counterbalanced
- Ukur: akurasi peserta sebelum/sesudah, SUS, feedback kelancaran video
- **Hasilnya harus asli** — brief melarang keras feedback/metric karangan

### Tahap 6 — Laporan (minggu 3–4)
Ikuti 10 bagian di brief §18 (Introduction … AI Usage Declaration). Tulis Methodology & Dataset sambil jalan, jangan ditumpuk di akhir.

---

## 5. Desain eksperimen (memenuhi "baseline + ≥2 skenario dengan variabel jelas")

**Tahap A — Perbandingan arsitektur × recipe (9 run, sesuai proposal)**

| | Config A | Config B | Config C |
|---|---|---|---|
| Optimizer | Adam 1e-3 | AdamW 1e-4 | SGD m=0.9, 1e-3 |
| Batch | 16 | 32 | 32 |
| LR schedule | Constant | Cosine (`T_max`=epochs) | ReduceLROnPlateau (0.5, p=3) |
| Weight decay | 1e-4 | 1e-2 | 1e-4 |
| Epoch / early stop | 20 / 5 | 25 / 5 | 30 / 7 |

× ResNet-50 (baseline), Xception, EfficientNet-B4.

**Tahap B — Ablation satu variabel pada model + config terbaik**

| ID | Variabel | Baseline | Eksperimen | Kenapa relevan |
|---|---|---|---|---|
| E1 | Augmentasi | tanpa | JPEG 40–80 + blur + flip + brightness | Video browser terkompresi |
| E2 | Transfer learning | backbone freeze (linear probe) | full fine-tune | Syarat brief §8 soal pretrained |
| E3 | Label smoothing | 0 | 0.05 | Klaim proposal soal overconfidence |
| E4 | Learning rate | 1e-4 | 1e-3 / 1e-5 | Sensitivitas hyperparameter |

Model terpilih = val AUC + F1 tertinggi yang lolos target latency < 200 ms. Ulang 3 seed untuk config final (mean ± SD).

Estimasi compute: ~12k gambar 224×224, ±1–3 menit/epoch di GPU Kaggle → total ±10–15 jam GPU. Masih muat kuota 1–2 minggu.

---

## 6. Pembagian tugas (usulan, 3 orang)

| Peran | Tanggung jawab utama |
|---|---|
| **Data & Eval** | Tahap 1, EDA, robustness test, error analysis, bab Dataset |
| **Model & Eksperimen** | `train.py`, 9 run + ablation, kurva, ONNX export, bab Model & Experiments |
| **App & User Testing** | FastAPI, extension, demo page, user testing, bab System Design & Real-World Eval |

Semua: AI usage log masing-masing, review laporan, Introduction & Conclusion bareng.

---

## 7. Struktur repo

```
look-twice/
├── README.md
├── PLAN.md
├── requirements.txt
├── configs/            # YAML per run (model, optimizer, lr, batch, aug, seed)
├── notebooks/          # Kaggle notebooks: 01_preprocess, 02_train, 03_eval
├── src/looktwice/
│   ├── data/           # parse metadata, split, frame extraction, face crop, dataset class
│   ├── models/         # build_model(name) via timm
│   ├── train.py
│   ├── evaluate.py     # frame & video metrics, robustness, Grad-CAM
│   └── export_onnx.py
├── backend/            # FastAPI + ONNX Runtime + face detector
├── extension/          # Chrome MV3: manifest, content script, service worker, overlay
├── demo/               # halaman demo + klip uji
├── results/            # CSV log, gambar kurva, confusion matrix (di-commit)
└── docs/               # ai_usage_log.md, user_testing/, report assets
```

---

## 8. Yang perlu diputuskan tim / dicek

- [ ] **Tanggal deadline pasti** (brief lama: Week 14; brief baru: ikut jadwal ujian)
- [ ] Rubrik mana yang berlaku (asumsi: Form Submit 2026/2027) & AI Use Type yang dipilih dosen
- [ ] Format nama file DFD & hasil hitungan split aktor
- [ ] Jumlah tester (proposal: 5; idealnya ada beberapa orang 50+ sesuai target user)
- [ ] Kaggle account siapa yang dipakai untuk menyimpan dataset crop
