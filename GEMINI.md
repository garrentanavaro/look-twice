# GEMINI.md

This file provides guidance to Gemini (Antigravity) when working with code in this repository.

## Project

LookTwice is a final project for COMP6826001 Deep Learning (BINUS). A Chrome extension samples frames from videos the user is watching, a local FastAPI server crops the face and runs a CNN deepfake classifier (ONNX), and the extension overlays a badge: likely authentic / uncertain / likely deepfake. Three ImageNet-pretrained backbones are compared on the DFD (Deepfake Detection, FaceForensics++) dataset: ResNet-50 (baseline), Xception, EfficientNet-B4. Team: Marco Oden Leo, Garren Tannavaro, Jason Hermawan.

Grading (brief "Form Submit Final Project COMP6826001", AY 2026/2027): data prep & experimental workflow 20%, model construction 20%, behavior/performance analysis 20%, real-world suitability 15%, working application 15%, AI usage & verification 10%. The report must not contain fabricated metrics, runs or user feedback; every number must come from an actual run.

## Commands

Run everything from the repo root using Python >= 3.10 (`.venv` uses Python 3.12). `pyproject.toml` declares no dependencies; install from requirements.

```bash
uv venv --python 3.12 .venv
.venv\Scripts\python -m pip install -r requirements.txt -r backend/requirements.txt   # torch CPU wheel is enough locally
.venv\Scripts\python -m pip install -e .
.venv\Scripts\python -m pytest -q                           # 45 tests, ~44 s on CPU, no dataset/GPU needed
.venv\Scripts\python -m pytest tests/test_training.py::test_metrics -q   # single test
```

Pipeline CLIs (all `python -m looktwice.<module>`):

```bash
python -m looktwice.preprocess [--config FILE] inspect --root <dfd> --out data/meta
python -m looktwice.preprocess [--config FILE] split --out data/meta [--seed N] [--policy strict|target] [--search-seeds N]
python -m looktwice.preprocess [--config FILE] extract --root <dfd> --meta data/meta --out data/faces [--limit N] [--workers 4]
python -m looktwice.train --config configs/train/base.yaml --config configs/train/recipe_B.yaml --set model.name=xception --set data.index=<crops_index.csv>
python -m looktwice.run_experiments --stage stage1|stage2|stage3 --set data.index=<crops_index.csv> [--only <substr>] [--dry-run]
python -m looktwice.run_experiments --collect                 # -> results/runs_summary.csv
python -m looktwice.evaluate --run runs/<name> [--run ...] --index <crops_index.csv> [--robustness --errors --gradcam 6]
python -m looktwice.export_onnx --run runs/<name> --out weights/looktwice.onnx --bench
uvicorn backend.app:create_app --factory --port 8000          # needs weights/looktwice.onnx + .json
```

The extension is loaded unpacked from `extension/` in `chrome://extensions`; the demo page is served by the backend at `/demo/`.

## Architecture

The stages communicate through files, and each file format is a contract the next stage relies on:

1. **Preprocessing** (`src/looktwice/data/`, `preprocess.py`). `metadata.py` parses DFD file names: originals `01__scene.mp4`, fakes `TARGET_DONOR__scene__ID.mp4` (verified on the real dataset). `split.py` assigns whole actors to train/val/test. Policy `strict` drops a fake whose donor actor is in a different split than its target (prevents face leakage); `target` keeps all fakes. `extract.py` samples evenly spaced frames, detects the largest face with YuNet (`faces.py`), saves a square crop enlarged by `bbox_scale` and zero-padded at frame edges. Output: `data/faces/crops/<split>/<real|fake>/*.jpg` + `crops_index.csv` (path relative to the CSV's folder, label 0=real/1=fake, split, video_id, target, donor, scene, frame, face_score, face_w, frame_w).
2. **Training** (`train.py`, `run_experiments.py`, `config.py`). Configs are merged in order (`base.yaml` + a recipe) and then `--set key.sub=value` overrides. `configs/train/experiments.yaml` defines stage1 (3 models × recipes A/B/C), stage2 (single-variable ablations on the best model, `model`/`recipe` must be filled in first) and stage3 (extra seeds). The runner skips runs whose `runs/<name>/summary.json` exists, so interrupted Kaggle sessions can resume. Each run writes `config.yaml`, `history.csv`, `best.pt`, `val_preds.csv`, `summary.json`. `best.pt` stores the state dict plus `model_name`, `input_size`, `mean`, `std`; `models.load_checkpoint` rebuilds the model from it without downloading weights.
3. **Evaluation** (`evaluate.py`, `thresholds.py`, `gradcam.py`, `plotting.py`). Thresholds come only from the run's `val_preds.csv`: `t` = best-F1 threshold (midpoint of the gap between scores), Uncertain band `[t-w, t+w)` = narrowest band where predictions outside it reach `--target-acc` on validation. The test split is used only here. Metrics are reported per frame and per video (mean of frame probabilities).
4. **Export** (`export_onnx.py`). ONNX contract: input `image` float32 `[N,3,S,S]` RGB in [0,1], normalization baked into the graph; output `prob` `[N]` = P(fake). A sidecar `<name>.json` holds input size, thresholds and source run.
5. **Serving** (`backend/app.py`). `Predictor` decodes the JPEG, runs YuNet, crops with the same `bbox_scale`/`save_size` read from `configs/preprocess.yaml`, resizes to the model input with INTER_AREA (same as training), runs ONNX. Frames are never written to disk. `/stats` reports end-to-end server latency.
6. **Extension** (`extension/`). `content.js` captures playing, visible videos at 2 FPS (canvas → JPEG data URL), sends them via `background.js` (the service worker can reach `http://localhost` thanks to `host_permissions`), and decides the badge from the mean of the last 10 frames with a face, using `lo`/`hi` returned by the backend. Cross-origin/DRM videos taint the canvas and show "Can't analyze".

## Gotchas

- Normalization mean/std must follow each pretrained checkpoint (`timm.data.resolve_model_data_config`); Xception uses 0.5/0.5, not ImageNet stats. Never hard-code ImageNet stats.
- PyYAML reads `1e-4` as a string; `config.py` coerces it, but write floats as `1.0e-4` in YAML anyway.
- OpenCV 5 has no Haar cascades; YuNet (`cv2.FaceDetectorYN`, OpenCV ≥ 4.8) is the only built-in detector. `ensure_yunet` downloads the model into `weights/` on first use.
- `build_detector` accepts `"module:Class"`; tests use `test_data_pipeline:DummyDetector` (also usable as `LOOKTWICE_DETECTOR` for the backend) to run without the YuNet model.
- `run_extract` builds the detector in the main process before starting the `Pool`: a failing Pool initializer makes Windows respawn workers forever and the run hangs silently.
- On Kaggle, attached datasets live under `/kaggle/input/datasets/<owner>/<slug>`; notebooks locate the DFD folder with a recursive glob.
- Kaggle CLI (`kaggle>=1.8.0`, `.venv` uses `2.2.5`) reads `~/.kaggle/access_token.txt` (generated via "Generate New Token") as well as legacy `kaggle.json`; `kaggle 2.2.5` also needs `six` installed. Phone verification on the Kaggle account is required for GPU + outbound Internet/DNS (`git clone`, `huggingface.co`).
- Line endings are mixed: Python, YAML and requirements files are CRLF, while JS/HTML/CSS, READMEs and some notebooks are LF. Keep each file's existing endings when editing so diffs stay small (`sed -i` in Git Bash silently converts to LF).

## Current state (2026-11-19)

- All 45 unit tests pass locally on CPU (`.\.venv\Scripts\python.exe -m pytest -q`, Python `3.12.10`).
- Real DFD preprocessing finished on Kaggle (`looktwice-01-preprocess`): 3,431 videos, 28 actors. Chosen split: `strict`, seed 185 (val actors 03, 06, 11, 14; test actors 02, 04, 07, 09; fake videos train 1,148 / val 143 / test 137, 1,640 dropped). `frames_fake` = 4, `frames_real` = 18; actual crop counts in `data/faces/crops_index.csv` are train `4,604` real / `4,590` fake (`9,194`), val `1,026` real / `572` fake (`1,598`), test `899` real / `548` fake (`1,447`). Dataset `garrentanavaro/looktwice-faces` is also extracted locally under `data/`.
- Training Stages 1–3 (16/16 runs) completed on Kaggle `Tesla T4` (`marcoodenleo/looktwice-02-stage1`, `looktwice-02-stage2`, `looktwice-02-stage3`); results are in `results/runs_summary.csv`, `results/curves/`, `results/runs_meta/`, and `runs/` (git-ignored):
  - **Stage 1 (9 runs):** `s1_xception_B` (`xception` + Recipe `B`) won with `val_auc=0.9602`, `val_f1=0.8589`, `val_acc=0.8999`, `val_video_auc=0.9838`, `val_video_f1=0.9333` (runner-up: `s1_efficientnet_b4_A` at `val_auc=0.9536`, `val_f1=0.8501`). Local ONNX CPU benchmark (`results/stage1_latency_check.csv`, Intel i7-13650HX): `s1_xception_B` `mean 14.1 ms` / `p95 14.8 ms` (`< 200 ms` satisfied).
  - **Stage 2 (5 ablations on `xception` + Recipe `B`):** All ablations scored below baseline (`E1_no_aug` `0.9564`, `E3_no_ls` `0.9554`, `E4_lr_div10` `0.9530`, `E4_lr_x10` `0.9517`, `E2_frozen` `0.7828`), so `configs/train/experiments.yaml` keeps `stage3.overrides: {}`.
  - **Stage 3 (seeds `1`, `2`, `42`):** `val_auc = 0.9566 ± 0.0037`, `val_f1 = 0.8441 ± 0.0144`, `val_acc = 0.8928 ± 0.0066`, `val_video_auc = 0.9829 ± 0.0018`, `val_video_f1 = 0.9137 ± 0.0270`.
- Next steps:
  - Marco: commit `results/` (branch `stage1`), PR to `main`, share `marcoodenleo/looktwice-02-stage3` output (`runs/`) with Jason & Garren, draft report chapters (*DL Fundamentals & Model*, *Experimental Methodology*, *Results*).
  - Jason: Stage 4 test-set evaluation + ONNX export (`notebooks/03_evaluate_kaggle.ipynb`; note `s1_efficientnet_b4_A` outperformed `s1_efficientnet_b4_B` on validation) & Stage 6 user study.
  - Garren: Stage 5 live Chrome extension + FastAPI backend testing.

## Repository conventions

- No comments or docstrings in code; explanations belong in READMEs and the report.
- Do not add AI attribution (Co-Authored-By lines or similar) to commits; commits are made under the team members' own git identities.
- Datasets, crops, `runs/`, `*.pt`, `*.onnx` and video files are git-ignored; only small result tables and figures in `results/` are committed.
