"""Unit tests for the independent eye-state ONNX inference module."""

import io
import json

import numpy as np
import pytest
from PIL import Image

from api.app import eye_state_inference as eye


class FakeValue:
    def __init__(self, name, shape):
        self.name = name
        self.shape = shape


class FakeSession:
    def __init__(self, input_shape=(1, 3, 224, 224), output_shape=(1, 2)):
        self._inputs = [FakeValue("image", list(input_shape))]
        self._outputs = [FakeValue("logits", list(output_shape))]

    def get_inputs(self):
        return self._inputs

    def get_outputs(self):
        return self._outputs


def image_bytes():
    buffer = io.BytesIO()
    Image.new("RGB", (80, 60), color=(120, 80, 40)).save(buffer, format="JPEG")
    return buffer.getvalue()


def test_preprocess_matches_expected_tensor_shape_and_type():
    tensor = eye.preprocess_image(image_bytes())

    assert tensor.shape == (1, 3, 224, 224)
    assert tensor.dtype == np.float32
    assert np.isfinite(tensor).all()


def test_softmax_converts_two_logits_to_probabilities():
    probabilities = eye._softmax_logits(np.asarray([[0.0, 2.0]], dtype=np.float32))

    assert probabilities.shape == (2,)
    assert np.isclose(float(probabilities.sum()), 1.0)
    assert probabilities[1] > probabilities[0]


def test_softmax_rejects_wrong_output_shape():
    with pytest.raises(eye.EyeStateInferenceError):
        eye._softmax_logits(np.asarray([[0.1]], dtype=np.float32))


def test_softmax_rejects_non_finite_logits():
    with pytest.raises(eye.EyeStateInferenceError):
        eye._softmax_logits(np.asarray([[np.nan, 1.0]], dtype=np.float32))


def test_session_validation_accepts_expected_model_signature():
    eye._validate_session(FakeSession())


def test_session_validation_rejects_wrong_input_dimensions():
    with pytest.raises(eye.EyeStateInferenceError):
        eye._validate_session(FakeSession(input_shape=(1, 3, 192, 192)))


def test_session_validation_rejects_wrong_number_of_classes():
    with pytest.raises(eye.EyeStateInferenceError):
        eye._validate_session(FakeSession(output_shape=(1, 1)))


def test_metadata_loader_accepts_training_metadata(tmp_path, monkeypatch):
    metadata = {
        "model_name": "AnemoScan Eye-State Classifier",
        "architecture": "MobileNetV3-Small",
        "classes": {"0": "open_eye", "1": "closed_eye"},
        "input_shape": [1, 3, 224, 224],
        "input_color": "RGB",
        "normalization_mean": [0.485, 0.456, 0.406],
        "normalization_std": [0.229, 0.224, 0.225],
    }
    path = tmp_path / "model_metadata.json"
    path.write_text(json.dumps(metadata), encoding="utf-8")
    monkeypatch.setattr(eye.settings, "eye_state_metadata_path", path)

    loaded = eye._load_metadata()

    assert loaded["classes"]["0"] == "open_eye"
    assert loaded["classes"]["1"] == "closed_eye"
