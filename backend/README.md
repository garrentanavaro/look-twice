# Backend

FastAPI server that receives video frames from the extension and returns a deepfake probability.

## Run

From the repository root:

```bash
pip install -r backend/requirements.txt
uvicorn backend.app:create_app --factory --port 8000
```

It expects `weights/looktwice.onnx` and `weights/looktwice.json` (created by `looktwice.export_onnx`). The YuNet face detector model is downloaded to `weights/` on first start if it is missing.

Frames are only processed in memory and are never saved.

## Endpoints

- `GET /health`: model info and decision thresholds
- `POST /predict`: send one JPEG frame as the request body; returns `face_found`, `prob`, `badge`, the face box and timing
- `GET /stats`: latency of recent requests (mean, p50, p95)
- `POST /stats/reset`: clear the latency stats
- `GET /demo/`: demo page

## Configuration

Optional environment variables:

- `LOOKTWICE_MODEL`: path to the ONNX model (default `weights/looktwice.onnx`)
- `LOOKTWICE_YUNET`: path to the YuNet model
- `LOOKTWICE_THREADS`: number of CPU threads for ONNX Runtime
- `LOOKTWICE_CLIPS`: folder with demo clips (default `demo/clips`)
