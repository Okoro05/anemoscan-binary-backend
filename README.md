# AnemoScan Binary Backend

FastAPI backend for the AnemoScan binary anemia-screening application.

The live analysis path now follows the supervisor-approved pipeline used to train the 200-image Random Forest model:

```text
captured conjunctival image
        ↓
resize to 512×512
        ↓
grayscale + Triangle threshold
        ↓
3×3 morphology open/close
        ↓
ROI pixels
        ↓
RGB + HSV + LAB means/stds
        ↓
trained Random Forest
        ↓
anemia probability
        ↓
Anemia Detected / No Anemia Detected
```

The clinical Hb label is used only during model training. A live image is **not** compared pixel-by-pixel with the 200 training images. Instead, it is converted into the same 18 numerical colour features and passed to the trained Random Forest.

## Random Forest model contract

The backend expects this file:

```text
api/model/anemoscan_color_feature_random_forest.joblib
```

It must be the model produced by the approved 200-image pipeline. The backend validates that it exposes `predict_proba`, is binary with classes `0` and `1`, and expects exactly these features in this order:

```text
R_mean, R_std,
G_mean, G_std,
B_mean, B_std,
H_mean, H_std,
S_mean, S_std,
V_mean, V_std,
L_mean, L_std,
A_mean, A_std,
B_channel_mean, B_channel_std
```

The classification threshold remains **0.50**. The frontend camera-quality threshold is separate from the AI classification threshold.

## API endpoints

- `GET /api/health` — service, database and Random Forest model status.
- `GET /api/captures` — recent scans with pagination.
- `POST /api/captures` — validate and save a captured image and scan metadata.
- `GET /api/captures/{id}/image` — retrieve the stored image.
- `PATCH /api/captures/{id}` — rename a scan.
- `DELETE /api/captures/{id}` — delete a scan and its image.
- `POST /api/captures/{id}/analyze` — run ROI extraction, RGB/HSV/LAB feature extraction and Random Forest classification.
- `GET /api/export` — export stored images and metadata as a ZIP archive.

The analysis response continues to expose the existing frontend fields:

```json
{
  "anemiaDetected": true,
  "anemiaProbability": 0.78,
  "anemiaStatus": "Anemia Detected"
}
```

## Local development

From the repository root:

```bash
cd api
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements-dev.txt
pytest -q
uvicorn app.main:app --reload
```

The application defaults to a local SQLite database when `DATABASE_URL` is not supplied. For deployment, use PostgreSQL and set `DATABASE_URL` in the hosting provider's environment settings.

## Deployment

The supplied `render.yaml` defines this repository as a backend-only Render web service. The Docker build uses `api/` as its context and starts Uvicorn on the host-provided `PORT`.

Required deployment environment variables:

- `DATABASE_URL` — PostgreSQL connection string.
- `CORS_ORIGINS` — comma-separated frontend origin(s), without a trailing slash.

Optional:

- `MODEL_PATH` — defaults to `model/anemoscan_color_feature_random_forest.joblib` inside the API container.
- `MAX_IMAGE_BYTES` — defaults to 8 MiB.

## Important model file step

The trained `.joblib` file is generated in Google Colab at:

```text
My Drive/Anemia_Detection/models/anemoscan_color_feature_random_forest.joblib
```

Copy that exact file into this repository at:

```text
api/model/anemoscan_color_feature_random_forest.joblib
```

Do not retrain the model in the backend and do not substitute the EfficientNet ONNX model. The backend is now intentionally wired to the supervisor-required Random Forest pipeline.

## Safety and project scope

This is an academic screening prototype, not a diagnostic medical device. The model was evaluated on a small held-out test set and requires external validation before clinical use.

There is currently no authentication or authorization layer. Do not expose the service to real patient data on the public internet until appropriate privacy, security, access-control and institutional requirements have been addressed.
