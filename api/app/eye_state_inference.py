"""ONNX inference for AnemoScan eye-state detection.

This model detects open vs closed eyes only. It is separate from the
binary anemia classifier in app.inference and any conjunctiva anatomy gate.
"""

from __future__ import annotations

import io
import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

from .config import settings

IMAGE_SIZE = 224
EXPECTED_CLASSES = ("open_eye", "closed_eye")
IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)

_session: ort.InferenceSession | None = None
_metadata: dict[str, Any] | None = None
_lock = threading.Lock()


class EyeStateInferenceError(Exception):
    """An expected error while loading or running the eye-state model."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


@dataclass(frozen=True)
class EyeStatePrediction:
    eye_state: str
    confidence: float
    probabilities: dict[str, float]


def reset_session() -> None:
    """Clear cached model state. Useful for tests and controlled reloads."""
    global _session, _metadata
    with _lock:
        _session = None
        _metadata = None


def _load_metadata() -> dict[str, Any]:
    path = settings.eye_state_metadata_path
    if not path.is_file():
        raise EyeStateInferenceError(503, f"Eye-state metadata was not found at {path}.")
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EyeStateInferenceError(500, "Eye-state model metadata could not be read.") from exc

    classes = metadata.get("classes", {})
    if classes.get("0") != "open_eye" or classes.get("1") != "closed_eye":
        raise EyeStateInferenceError(500, "Eye-state metadata must define 0=open_eye and 1=closed_eye.")
    if metadata.get("input_shape") != [1, 3, IMAGE_SIZE, IMAGE_SIZE]:
        raise EyeStateInferenceError(500, "Eye-state metadata has an unsupported input shape.")
    if metadata.get("input_color") != "RGB":
        raise EyeStateInferenceError(500, "Eye-state model must use RGB input.")
    if metadata.get("normalization_mean") != IMAGENET_MEAN.tolist():
        raise EyeStateInferenceError(500, "Eye-state normalization mean does not match the training pipeline.")
    if metadata.get("normalization_std") != IMAGENET_STD.tolist():
        raise EyeStateInferenceError(500, "Eye-state normalization standard deviation does not match the training pipeline.")
    return metadata


def _validate_session(session: ort.InferenceSession) -> None:
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1 or len(outputs) != 1:
        raise EyeStateInferenceError(500, "Eye-state ONNX model must have exactly one input and one output.")
    shape = inputs[0].shape
    if len(shape) != 4 or shape[1:] != [3, IMAGE_SIZE, IMAGE_SIZE]:
        raise EyeStateInferenceError(
            500, f"Unexpected eye-state model input shape {shape}; expected [batch, 3, 224, 224]."
        )
    output_shape = outputs[0].shape
    if len(output_shape) != 2 or output_shape[-1] != 2:
        raise EyeStateInferenceError(
            500, f"Unexpected eye-state output shape {output_shape}; expected [batch, 2] logits."
        )


def get_session() -> ort.InferenceSession:
    global _session, _metadata
    with _lock:
        if _session is None:
            model_path = settings.eye_state_model_path
            if not model_path.is_file():
                raise EyeStateInferenceError(503, f"Eye-state ONNX model was not found at {model_path}.")
            try:
                metadata = _load_metadata()
                session = ort.InferenceSession(
                    str(model_path),
                    providers=["CPUExecutionProvider"],
                )
                _validate_session(session)
                _metadata = metadata
                _session = session
            except EyeStateInferenceError:
                _session = None
                _metadata = None
                raise
            except Exception as exc:
                _session = None
                _metadata = None
                raise EyeStateInferenceError(500, "The eye-state ONNX model could not be loaded.") from exc
        return _session


def model_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "eyeStateModelPresent": settings.eye_state_model_path.is_file(),
        "eyeStateModelLoaded": _session is not None,
        "eyeStateModelType": "open-closed-eye-state-classifier",
        "eyeStateClasses": list(EXPECTED_CLASSES),
    }
    if _session is not None:
        info["eyeStateModelInputs"] = [item.name for item in _session.get_inputs()]
        info["eyeStateModelOutputs"] = [item.name for item in _session.get_outputs()]
    return info


def preprocess_image(raw_bytes: bytes) -> np.ndarray:
    """Match eval_transform: RGB, 224x224, ToTensor, ImageNet normalization."""
    try:
        with Image.open(io.BytesIO(raw_bytes)) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
            array = np.asarray(image, dtype=np.float32) / 255.0
    except Exception as exc:
        raise EyeStateInferenceError(422, "The supplied image could not be decoded.") from exc

    array = (array - IMAGENET_MEAN) / IMAGENET_STD
    return np.expand_dims(array.transpose(2, 0, 1), axis=0).astype(np.float32)


def _softmax_logits(raw_output: np.ndarray) -> np.ndarray:
    logits = np.asarray(raw_output, dtype=np.float32)
    if logits.ndim != 2 or logits.shape[0] != 1 or logits.shape[1] != 2:
        raise EyeStateInferenceError(500, "Eye-state model returned an invalid logits tensor.")
    if not np.isfinite(logits).all():
        raise EyeStateInferenceError(500, "Eye-state model returned non-finite logits.")
    shifted = logits[0] - np.max(logits[0])
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values)


def predict(image_bytes: bytes) -> EyeStatePrediction:
    session = get_session()
    image_array = preprocess_image(image_bytes)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    try:
        raw_output = session.run([output_name], {input_name: image_array})[0]
    except Exception as exc:
        raise EyeStateInferenceError(500, "The eye-state model could not analyze this image.") from exc

    probabilities_array = _softmax_logits(raw_output)
    index = int(np.argmax(probabilities_array))
    probabilities = {
        "open_eye": round(float(probabilities_array[0]), 6),
        "closed_eye": round(float(probabilities_array[1]), 6),
    }
    return EyeStatePrediction(
        eye_state=EXPECTED_CLASSES[index],
        confidence=round(float(probabilities_array[index]), 6),
        probabilities=probabilities,
    )
