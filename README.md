# AnemoScan Binary Backend

FastAPI backend for the AnemoScan binary anemia-screening application.

The backend accepts a captured conjunctival image, stores the scan and image, runs the final binary ONNX classifier, and returns:

```json
{
  "anemiaDetected": true,
  "anemiaProbability": 0.89,
  "anemiaStatus": "Anemia Detected"
}
```

The model threshold is **0.50**. The camera quality threshold of **70/100** is enforced by the frontend and is separate from the AI classification threshold.

## Repository structure

```text
api/
  app/
    config.py       environment/configuration
    db.py           SQLAlchemy engine and database initialization
    inference.py    ONNX preprocessing, validation and binary inference
    main.py         FastAPI application and CORS/lifespan setup
    models.py       capture and image database models
    routes.py       health, capture, image, analysis and export endpoints
    schemas.py      request/response validation
  model/
    model.onnx      final binary anemia classifier
  tests/             API and inference integration tests
  Dockerfile
  requirements.txt
  requirements-dev.txt
render.yaml
.github/workflows/ci.yml
```

## Binary inference contract

- Input: one RGB image, resized to 224x224.
- Preprocessing: `/255`, ImageNet mean/std normalization, CHW layout, batch dimension.
- Model: one ONNX input and one output; startup validation rejects an incompatible model.
- Output: probability in `[0, 1]`, binary result at threshold `0.50`, and status text.
- Patient age and gender are stored for scan context but are **not** model inputs.
- No predicted hemoglobin value or anemia-severity classification is produced by this backend.

## API endpoints

- `GET /api/health` — service, database and model status.
- `GET /api/captures` — recent scans with pagination.
- `POST /api/captures` — validate and save a captured image and scan metadata.
- `GET /api/captures/{id}/image` — retrieve the stored image.
- `PATCH /api/captures/{id}` — rename a scan.
- `DELETE /api/captures/{id}` — delete a scan and its image.
- `POST /api/captures/{id}/analyze` — run the binary anemia classifier.
- `GET /api/export` — export stored images and metadata as a ZIP archive.

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

- `MODEL_PATH` — defaults to `model/model.onnx` inside the API container.
- `MAX_IMAGE_BYTES` — defaults to 8 MiB.

## Safety and project scope

This is an academic screening prototype, not a diagnostic medical device. The model was evaluated on a small held-out test set and requires external validation before clinical use.

There is currently no authentication or authorization layer. Do not expose the service to real patient data on the public internet until appropriate privacy, security, access-control and institutional requirements have been addressed.
