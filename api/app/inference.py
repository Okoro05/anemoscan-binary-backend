"""Binary anemia inference using the approved RGB/HSV/LAB + Random Forest pipeline."""

import io
import threading
from dataclasses import dataclass

import cv2
import joblib
import numpy as np
from PIL import Image, ImageOps

from .config import settings

IMAGE_SIZE = 512
AI_THRESHOLD = 0.50
MIN_ROI_PIXELS = 100

FEATURE_COLUMNS = [
    "R_mean", "R_std",
    "G_mean", "G_std",
    "B_mean", "B_std",
    "H_mean", "H_std",
    "S_mean", "S_std",
    "V_mean", "V_std",
    "L_mean", "L_std",
    "A_mean", "A_std",
    "B_channel_mean", "B_channel_std",
]

_model = None
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


def reset_model() -> None:
    global _model
    with _lock:
        _model = None


def _validate_model(model) -> None:
    if not hasattr(model, "predict_proba"):
        raise InferenceError(500, "The configured model is not a probability-producing Random Forest classifier.")

    if not hasattr(model, "n_features_in_") or int(model.n_features_in_) != len(FEATURE_COLUMNS):
        raise InferenceError(
            500,
            f"Unexpected model feature count. Expected {len(FEATURE_COLUMNS)} visual colour features.",
        )

    if not hasattr(model, "classes_") or set(np.asarray(model.classes_).tolist()) != {0, 1}:
        raise InferenceError(500, "The configured model must be a binary anemia classifier with classes 0 and 1.")


def get_model():
    global _model
    with _lock:
        if _model is None:
            if not settings.model_path.exists():
                raise InferenceError(
                    503,
                    f"No Random Forest model found at {settings.model_path}. Add the trained joblib model first.",
                )
            try:
                _model = joblib.load(settings.model_path)
                _validate_model(_model)
            except InferenceError:
                _model = None
                raise
            except Exception as exc:
                _model = None
                raise InferenceError(500, "The Random Forest anemia model could not be loaded.") from exc
        return _model


def model_info() -> dict:
    return {
        "modelPresent": settings.model_path.exists(),
        "modelLoaded": _model is not None,
        "modelType": "rgb-hsv-lab-random-forest",
        "classificationThreshold": AI_THRESHOLD,
        "featureCount": len(FEATURE_COLUMNS),
        "roiSize": IMAGE_SIZE,
        "roiMethod": "triangle-threshold-with-morphology",
    }


def _prepare_image(raw_bytes: bytes) -> np.ndarray:
    """Decode the captured image to BGR for OpenCV processing."""
    try:
        image = ImageOps.exif_transpose(Image.open(io.BytesIO(raw_bytes))).convert("RGB")
        rgb = np.asarray(image, dtype=np.uint8)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except Exception as exc:
        raise InferenceError(400, "The stored image could not be read.") from exc


def prototype_roi(bgr_img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce the approved 200-image training ROI pipeline exactly."""
    resized = cv2.resize(bgr_img, (IMAGE_SIZE, IMAGE_SIZE))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    _, mask = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE,
    )

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    roi = cv2.bitwise_and(resized, resized, mask=mask)
    return roi, mask


def extract_features_from_roi(roi_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return the exact 18 visual features used to train the Random Forest."""
    if roi_bgr is None or mask is None:
        raise InferenceError(422, "A valid conjunctival ROI could not be extracted.")

    valid = mask == 255
    if int(valid.sum()) < MIN_ROI_PIXELS:
        raise InferenceError(422, "The captured image does not contain enough valid ROI pixels for analysis.")

    rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB)

    rgb_pixels = rgb[valid]
    hsv_pixels = hsv[valid]
    lab_pixels = lab[valid]

    features = [
        float(np.mean(rgb_pixels[:, 0])), float(np.std(rgb_pixels[:, 0])),
        float(np.mean(rgb_pixels[:, 1])), float(np.std(rgb_pixels[:, 1])),
        float(np.mean(rgb_pixels[:, 2])), float(np.std(rgb_pixels[:, 2])),
        float(np.mean(hsv_pixels[:, 0])), float(np.std(hsv_pixels[:, 0])),
        float(np.mean(hsv_pixels[:, 1])), float(np.std(hsv_pixels[:, 1])),
        float(np.mean(hsv_pixels[:, 2])), float(np.std(hsv_pixels[:, 2])),
        float(np.mean(lab_pixels[:, 0])), float(np.std(lab_pixels[:, 0])),
        float(np.mean(lab_pixels[:, 1])), float(np.std(lab_pixels[:, 1])),
        float(np.mean(lab_pixels[:, 2])), float(np.std(lab_pixels[:, 2])),
    ]

    return np.asarray([features], dtype=np.float32)


def extract_features(raw_bytes: bytes) -> np.ndarray:
    bgr = _prepare_image(raw_bytes)
    roi, mask = prototype_roi(bgr)
    return extract_features_from_roi(roi, mask)


def predict(image_bytes: bytes) -> Prediction:
    model = get_model()
    features = extract_features(image_bytes)

    try:
        probabilities = model.predict_proba(features)
    except Exception as exc:
        raise InferenceError(500, "The Random Forest anemia classifier could not analyze this image.") from exc

    classes = np.asarray(model.classes_)
    positive_indices = np.where(classes == 1)[0]
    if len(positive_indices) != 1:
        raise InferenceError(500, "The Random Forest model does not expose a binary anemia probability.")

    probability = float(probabilities[0, positive_indices[0]])
    probability = float(np.clip(probability, 0.0, 1.0))
    detected = bool(probability >= AI_THRESHOLD)

    return Prediction(
        anemia_detected=detected,
        anemia_probability=round(probability, 4),
        status="Anemia Detected" if detected else "No Anemia Detected",
    )
