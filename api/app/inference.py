"""Binary anemia inference for the final AnemoScan classifier."""

import io
import threading
from dataclasses import dataclass

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

from .config import settings

IMAGE_SIZE = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
AI_THRESHOLD = 0.50

_session: ort.InferenceSession | None = None
_lock = threading.Lock()


class InferenceError(Exception):
    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


@dataclass
class Prediction:
    anemia_detected: bool
    anemia_probability: float
    status: str


def reset_session() -> None:
    global _session
    with _lock:
        _session = None


def _validate_session(session: ort.InferenceSession) -> None:
    inputs = session.get_inputs()
    if len(inputs) != 1:
        raise InferenceError(500, "The configured model is not the final binary image-only anemia classifier.")
    shape = inputs[0].shape
    if len(shape) != 4 or shape[1:] != [3, IMAGE_SIZE, IMAGE_SIZE]:
        raise InferenceError(500, f"Unexpected model input shape {shape}. Expected [1, 3, 224, 224].")
    if len(session.get_outputs()) != 1:
        raise InferenceError(500, "The configured model must expose exactly one binary output.")


def get_session() -> ort.InferenceSession:
    global _session
    with _lock:
        if _session is None:
            if not settings.model_path.exists():
                raise InferenceError(503, f"No binary model found at {settings.model_path}. Add model.onnx first.")
            try:
                _session = ort.InferenceSession(str(settings.model_path), providers=["CPUExecutionProvider"])
                _validate_session(_session)
            except InferenceError:
                _session = None
                raise
            except Exception as exc:
                _session = None
                raise InferenceError(500, "The binary anemia model could not be loaded.") from exc
        return _session


def model_info() -> dict:
    info: dict = {
        "modelPresent": settings.model_path.exists(),
        "modelLoaded": _session is not None,
        "modelType": "binary-anemia-classifier",
        "classificationThreshold": AI_THRESHOLD,
    }
    if _session is not None:
        info["modelInputs"] = [item.name for item in _session.get_inputs()]
        info["modelOutputs"] = [item.name for item in _session.get_outputs()]
    return info


def preprocess_image(raw_bytes: bytes) -> np.ndarray:
    """Match the final binary training pipeline: RGB, 224x224, /255, ImageNet normalisation."""
    try:
        image = ImageOps.exif_transpose(Image.open(io.BytesIO(raw_bytes))).convert("RGB")
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
        array = (array - IMAGENET_MEAN) / IMAGENET_STD
        return np.expand_dims(array.transpose(2, 0, 1), axis=0).astype(np.float32)
    except Exception as exc:
        raise InferenceError(400, "The stored image could not be read.") from exc


def _to_probability(raw_output: np.ndarray) -> float:
    value = float(np.asarray(raw_output).reshape(-1)[0])
    if 0.0 <= value <= 1.0:
        probability = value
    else:
        probability = 1.0 / (1.0 + np.exp(-value))
    return float(np.clip(probability, 0.0, 1.0))


def predict(image_bytes: bytes) -> Prediction:
    session = get_session()
    image_array = preprocess_image(image_bytes)
    input_name = session.get_inputs()[0].name
    try:
        output = session.run(None, {input_name: image_array})
    except Exception as exc:
        raise InferenceError(500, "The anemia classifier could not analyze this image.") from exc

    probability = _to_probability(output[0])
    detected = bool(probability >= AI_THRESHOLD)
    return Prediction(
        anemia_detected=detected,
        anemia_probability=round(probability, 4),
        status="Anemia Detected" if detected else "No Anemia Detected",
    )
