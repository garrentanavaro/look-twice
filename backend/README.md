# Backend (FastAPI + ONNX Runtime)

```bash
# dari root repo, venv aktif, package sudah `pip install -e .`
.venv/Scripts/python -m pip install -r backend/requirements.txt
.venv/Scripts/python -m uvicorn backend.app:create_app --factory --port 8000
```

Butuh:
- `weights/looktwice.onnx` + `weights/looktwice.json`, hasil dari `python -m looktwice.export_onnx`.
- `weights/face_detection_yunet_2023mar.onnx`. Kalau belum ada, file ini otomatis di-download dari repo resmi `opencv_zoo` saat server pertama kali jalan.

## Endpoint

| Method | Path | Isi |
|---|---|---|
| GET | `/health` | nama model, ukuran input, threshold `t/lo/hi`, provider ONNX, detektor |
| POST | `/predict` | body = byte JPEG satu frame → `{face_found, prob, badge, box, thresholds, latency_ms{decode,detect,infer,total}}` |
| GET | `/stats` | latency server mean/p50/p95 dari 2000 request terakhir + rasio "no face" (untuk laporan) |
| POST | `/stats/reset` | kosongkan statistik sebelum sesi pengukuran |
| GET | `/api/clips?set=A` | daftar klip di `demo/clips/A/` |
| GET | `/demo/` | halaman demo untuk user testing |

Pemrosesan per frame:
1. Decode JPEG.
2. YuNet mencari wajah dan mengambil wajah terbesar.
3. Crop persegi ×1.2 ke 256, lalu resize ke ukuran input model. Langkah ini **identik dengan saat training** (parameter dibaca dari `configs/preprocess.yaml`).
4. ONNX menghasilkan P(fake), lalu badge ditentukan dengan threshold dari `.json`.

Smoothing (rata-rata 10 frame) dilakukan di extension, per video.

**Privasi:** frame hanya diproses di memori, tidak pernah disimpan.

## Variabel lingkungan

| Variabel | Default |
|---|---|
| `LOOKTWICE_MODEL` | `weights/looktwice.onnx` |
| `LOOKTWICE_DETECTOR` | `yunet` (atau `modul:Kelas` untuk detektor lain/dummy) |
| `LOOKTWICE_YUNET` | `weights/face_detection_yunet_2023mar.onnx` |
| `LOOKTWICE_THREADS` | otomatis |
| `LOOKTWICE_CLIPS` | `demo/clips` |

## Mengukur latency end-to-end (target proposal < 200 ms)

1. `POST /stats/reset`.
2. Putar video di halaman demo dengan extension aktif selama ±1 menit.
3. Buka `GET /stats` untuk latency server, dan popup extension untuk round-trip (capture → server → hasil).
