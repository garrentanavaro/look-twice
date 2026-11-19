# LookTwice

A browser extension powered by deep learning that flags face-swapped (deepfake) videos while you watch them, to help older adults spot manipulated media used in scams.

Final project for COMP6826001 Deep Learning, BINUS University.
Team: Marco Oden Leo, Garren Tannavaro, Jason Hermawan.

## How it works

The Chrome extension grabs two frames per second from the video you are watching and sends them to a small FastAPI server running on your machine. The server finds the face, crops it, and runs a CNN classifier (exported to ONNX). The extension averages the last few predictions and shows a badge on the video: likely authentic, uncertain, or likely deepfake.

We train and compare three ImageNet-pretrained CNNs on the [DFD dataset](https://www.kaggle.com/datasets/sanikatiwarekar/deep-fake-detection-dfd-entire-original-dataset): ResNet-50 as the baseline, Xception, and EfficientNet-B4. Train, validation and test sets are split by actor, so the model is always tested on faces it has never seen.

## Repository layout

```
src/looktwice/   data pipeline, training, evaluation, ONNX export
configs/         preprocessing and training configs
notebooks/       Kaggle notebooks: preprocess, train, evaluate
backend/         FastAPI inference server
extension/       Chrome extension (Manifest V3)
demo/            demo page for the user study
tests/           unit tests
```

## Setup

Python 3.10 or newer.

```bash
python -m venv .venv
.venv\Scripts\activate        # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pytest
```

The tests use small synthetic data, so they run without the dataset or a GPU.

## Data preparation

The dataset is too large to download locally, so preprocessing runs on Kaggle (`notebooks/01_preprocess_kaggle.ipynb`).

```bash
python -m looktwice.preprocess inspect --root <dfd_folder> --out data/meta
python -m looktwice.preprocess split --out data/meta --seed 42
python -m looktwice.preprocess extract --root <dfd_folder> --meta data/meta --out data/faces
```

This samples frames from each video, detects the face with YuNet and saves 256×256 crops plus a `crops_index.csv`. Settings are in `configs/preprocess.yaml`.

## Training

See `notebooks/02_train_kaggle.ipynb`. A single run:

```bash
python -m looktwice.train --config configs/train/base.yaml --config configs/train/recipe_B.yaml \
    --set model.name=xception --set data.index=<path/to/crops_index.csv>
```

All planned runs are listed in `configs/train/experiments.yaml` and can be run per stage:

```bash
python -m looktwice.run_experiments --stage stage1 --set data.index=<path/to/crops_index.csv>
python -m looktwice.run_experiments --collect
```

All 16 runs (Stages 1–3) have been completed on Kaggle (`Tesla T4`); summary metrics and training curves are recorded in `results/runs_summary.csv`, `results/curves/`, and `results/runs_meta/`:

- **Stage 1 (3 architectures × 3 recipes):** `s1_xception_B` (`xception` + Recipe `B`) ranked highest on validation (`val_auc=0.9602`, `val_f1=0.8589`, `val_video_auc=0.9838`, `val_video_f1=0.9333`) with `14.8 ms` p95 CPU ONNX latency (`results/stage1_latency_check.csv`, under the `200 ms` budget).
- **Stage 2 (5 single-variable ablations on `xception` + Recipe `B`):** All ablations scored below the baseline (`E1_no_aug` `0.9564`, `E3_no_ls` `0.9554`, `E4_lr_div10` `0.9530`, `E4_lr_x10` `0.9517`, `E2_frozen` `0.7828`).
- **Stage 3 (seeds `1`, `2`, `42`):** `val_auc = 0.9566 ± 0.0037`, `val_f1 = 0.8441 ± 0.0144`, `val_video_auc = 0.9829 ± 0.0018`.

## Evaluation and export

See `notebooks/03_evaluate_kaggle.ipynb`.

```bash
python -m looktwice.evaluate --run runs/<run_name> --index <path/to/crops_index.csv> \
    --robustness --errors --gradcam 6
python -m looktwice.export_onnx --run runs/<run_name> --out weights/looktwice.onnx --bench
```

## Running the app

1. Put `looktwice.onnx` and `looktwice.json` in `weights/`.
2. Start the server:
   ```bash
   pip install -r backend/requirements.txt
   uvicorn backend.app:create_app --factory --port 8000
   ```
3. Load the extension: open `chrome://extensions`, turn on Developer mode, click "Load unpacked" and select the `extension/` folder.
4. Open http://localhost:8000/demo/ or any page with a video.

More details in [backend](backend/README.md), [extension](extension/README.md) and [demo](demo/README.md).
