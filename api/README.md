# AnemoScan API

This directory is the deployable FastAPI backend for the binary anemia-screening workflow.

Install dependencies with `pip install -r requirements.txt`, run tests with `pytest -q`, and start the service with `uvicorn app.main:app --host 0.0.0.0 --port 8000`.

The ONNX model is stored at `model/model.onnx` and is validated at application startup before inference is allowed.
