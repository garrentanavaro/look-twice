# LookTwice

Real-time browser-integrated deepfake video detection for scam and fraud prevention.
Final project COMP6826001 Deep Learning — Marco Oden Leo, Garren Tannavaro, Jason Hermawan.

Rencana kerja lengkap: [PLAN.md](PLAN.md).

## Struktur

| Folder | Isi | Status |
|---|---|---|
| `src/looktwice/data/` | scan metadata DFD, split per aktor, ekstraksi frame, crop wajah (YuNet) | ✅ siap, ter-test |
| `src/looktwice/preprocess.py` | CLI `inspect` / `split` / `extract` | ✅ |
| `configs/preprocess.yaml` | parameter preprocessing | ✅ |
| `notebooks/01_preprocess_kaggle.ipynb` | menjalankan preprocessing di Kaggle | ✅ |
| `src/looktwice/data/dataset.py`, `models/`, `train.py` | Dataset + augmentasi, model timm, loop training | ✅ siap, ter-test di CPU |
| `src/looktwice/run_experiments.py`, `configs/train/` | recipe A/B/C, rencana stage 1–3, runner yang bisa resume | ✅ |
| `notebooks/02_train_kaggle.ipynb` | training di Kaggle + kurva | ✅ |
| `evaluate.py`, `thresholds.py`, `gradcam.py`, `plotting.py` | test set, threshold & pita Uncertain, robustness, error analysis, Grad-CAM, grafik laporan | ✅ ter-test di CPU |
| `export_onnx.py` | export ONNX + verifikasi vs PyTorch + benchmark latency | ✅ |
| `notebooks/03_evaluate_kaggle.ipynb` | evaluasi akhir + export | ✅ |
| `backend/` | FastAPI + ONNX Runtime: `/predict`, `/health`, `/stats`, menyajikan halaman demo | ✅ ter-test (pytest) |
| `extension/` | Chrome extension MV3: capture 2 FPS, badge Hijau/Kuning/Merah, popup statistik | ✅ diuji di browser pane, belum dipasang di Chrome |
| `demo/` | halaman user testing: video berdampingan, jawaban + keyakinan, export CSV | ✅ |
| `results/` | CSV metric, kurva, confusion matrix (di-commit) | — |
| `docs/ai_usage_log.md` | AI Usage Log (wajib, 10% nilai) | 🟡 isi terus |

## Setup lokal (Windows)

```bash
python -m venv .venv
.venv/Scripts/python -m pip install numpy pandas pyyaml "opencv-python-headless>=4.8" tqdm pytest
.venv/Scripts/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python -m pip install timm "albumentations>=2.0" scikit-learn matplotlib onnx onnxruntime
.venv/Scripts/python -m pip install -e .
.venv/Scripts/python -m pytest -q
```

Test memakai video & gambar sintetis, detektor dummy, dan model kecil tanpa bobot pretrained, jadi tidak butuh dataset asli maupun GPU. Torch versi CPU cukup untuk laptop. Training sungguhan dilakukan di Kaggle.

## Preprocessing (di Kaggle)

Dataset DFD terlalu besar untuk laptop, jadi preprocessing dijalankan di Kaggle Notebook
(lihat `notebooks/01_preprocess_kaggle.ipynb`; nyalakan **Internet** di settings notebook
supaya `pip install` dan download model YuNet bisa jalan).

```bash
# 1. scan video, parse ID aktor -> data/meta/videos.csv
python -m looktwice.preprocess inspect --root <folder DFD> --out data/meta

# 2. cari seed split yang menyisakan video fake terbanyak di val/test, lalu split
python -m looktwice.preprocess split --out data/meta --search-seeds 500
python -m looktwice.preprocess split --out data/meta --seed <seed terbaik> --policy strict

# 3. smoke test dulu (5 video per kelas), cek hasil crop-nya, baru full run
python -m looktwice.preprocess extract --root <folder DFD> --meta data/meta --out data/faces --limit 5
python -m looktwice.preprocess extract --root <folder DFD> --meta data/meta --out data/faces --workers 4
```

Output:
- `data/meta/videos.csv`, `videos_split.csv`, `actor_split.json`: metadata & pembagian aktor
- `data/faces/crops/<split>/<real|fake>/*.jpg`: crop wajah 256×256
- `data/faces/crops_index.csv`: satu baris per crop (path, label, split, video, aktor, frame)
- `data/faces/failures.csv`: frame yang tidak ada wajahnya

Simpan `data/` sebagai **Kaggle Dataset** baru supaya notebook training tinggal attach.

### Policy split
- `strict`: video fake hanya dipakai kalau aktor target **dan** donor ada di split yang sama → tidak ada kebocoran wajah, tapi banyak video fake terbuang.
- `target`: video fake ikut split aktor target → semua video terpakai, tapi wajah donor bisa muncul di split lain.

`split` selalu mencetak jumlah video untuk kedua policy. Pilih berdasarkan angka asli, lalu tulis alasannya di laporan.

## Training (di Kaggle)

Lihat `notebooks/02_train_kaggle.ipynb`. Intinya:

```bash
# satu run manual (config digabung berurutan, --set menimpa)
python -m looktwice.train --config configs/train/base.yaml --config configs/train/recipe_B.yaml \
    --set model.name=xception --set data.index=<crops_index.csv>

# stage dari configs/train/experiments.yaml (run yang sudah selesai dilewati)
python -m looktwice.run_experiments --stage stage1 --set data.index=<crops_index.csv> --set out_dir=runs
python -m looktwice.run_experiments --collect     # -> results/runs_summary.csv
```

| Stage | Isi | Jumlah run |
|---|---|---|
| stage1 | ResNet-50 (baseline), Xception, EfficientNet-B4 × recipe A/B/C | 9 |
| stage2 | ablation satu variabel pada model+recipe terbaik: tanpa augmentasi, backbone beku, tanpa label smoothing, LR ×10 / ÷10 | 5 |
| stage3 | konfigurasi final diulang dengan seed 1 & 2 (mean ± SD bersama seed 42) | 2 |

Setiap run menulis `runs/<nama>/`: `config.yaml`, `history.csv` (loss/metric/LR/grad-norm per epoch),
`best.pt`, `val_preds.csv`, `summary.json` (metric terbaik, parameter, waktu, memori GPU).

Catatan desain:
- Normalisasi mengikuti checkpoint pretrained (Xception mean/std 0.5, ResNet & EfficientNet ImageNet).
- Semua model memakai input 224 agar perbandingan adil (resolusi asli: Xception 299, EfficientNet-B4 320).
- Loss: BCE + label smoothing 0.05; `loss.pos_weight: auto` tersedia kalau kelas train tidak seimbang.
- Checkpoint & early stopping memakai `val_auc` (bisa diganti `train.monitor`). Test set **tidak** disentuh saat training.
- Recipe B: `t_max` cosine = jumlah epoch (proposal menulis 20 untuk 25 epoch).

## Evaluasi & export

Lihat `notebooks/03_evaluate_kaggle.ipynb`. Test set dipakai **sekali**, setelah model dipilih dari hasil validasi.

```bash
# perbandingan arsitektur di test set (ROC, metrik frame & video, confusion matrix)
python -m looktwice.evaluate --run runs/s1_resnet50_B --run runs/s1_xception_B --run runs/s1_efficientnet_b4_B \
    --index <crops_index.csv> --out results/eval

# analisis model final: robustness, pola error, Grad-CAM
python -m looktwice.evaluate --run runs/<final> --index <crops_index.csv> --out results/eval_final \
    --robustness --errors --gradcam 6

# ONNX untuk backend + cek selisih vs PyTorch + latency (append ke results/latency.csv)
python -m looktwice.export_onnx --run runs/<final> --out weights/looktwice.onnx --bench
```

| Output | Untuk bab laporan |
|---|---|
| `comparison_test.csv`, `roc_test.png`, `confusion_*.png` | Results: perbandingan model (brief §13) |
| `thresholds.json`, kolom `badge` di `test_preds.csv` | System design: aturan badge Hijau/Amber/Merah |
| `robustness.csv/.png` (JPEG 90→15, blur 3→11, downscale 0.75→0.25) | Real-world suitability: robustness |
| `errors_by_{scene,target_actor,face_size,detector_conf}.csv`, `false_*.png` | Error analysis (brief §14) |
| `gradcam.png` | Explainability: model melihat area mana |
| `results/latency.csv` | Computational cost: ms/frame CPU vs GPU, PyTorch vs ONNX |

**Threshold & pita Uncertain** dipilih dari prediksi *validation*:
- `t` = threshold dengan F1 tertinggi (diambil di titik tengah celah antar skor).
- Pita `[t−w, t+w)` = pita tersempit yang membuat akurasi prediksi di luar pita ≥ 95% (atur dengan `--target-acc`).

Badge: `p < lo` → Authentic, `lo ≤ p < hi` → Uncertain, `p ≥ hi` → Manipulated.

**Kontrak ONNX:**
- Input `image`: float32 `[N,3,S,S]`, RGB, nilai 0–1. Normalisasi sudah di dalam graph.
- Output `prob`: `[N]` = P(fake).
- File `.json` di sebelahnya menyimpan ukuran input, threshold, dan asal run.

Latency di `export_onnx` hanya mengukur model. Latency end-to-end (decode + deteksi wajah + model) diukur di backend.

## Aplikasi (laptop)

```bash
# 1. taruh weights/looktwice.onnx + .json hasil notebook 03, lalu
.venv/Scripts/python -m pip install -r backend/requirements.txt
.venv/Scripts/python -m uvicorn backend.app:create_app --factory --port 8000
# 2. Chrome -> chrome://extensions -> Developer mode -> Load unpacked -> folder extension/
# 3. buka http://localhost:8000/demo/
```

Detail: [backend/README.md](backend/README.md), [extension/README.md](extension/README.md), [demo/README.md](demo/README.md).
