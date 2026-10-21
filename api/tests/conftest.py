"""Test setup for the binary anemia API."""

import base64
import io
import os
import sys
import tempfile
from pathlib import Path

import onnx
import pytest
from onnx import TensorProto, helper
from PIL import Image

TMP = Path(tempfile.mkdtemp(prefix="anemoscan-tests-"))
os.environ["DATABASE_URL"] = os.getenv("TEST_DATABASE_URL") or f"sqlite:///{TMP / 'test.db'}"
os.environ["MODEL_PATH"] = str(TMP / "model.onnx")
os.environ.pop("WEB_DIST_DIR", None)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _value_info(name, shape):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)


def _build_model():
    bias = helper.make_tensor("bias", TensorProto.FLOAT, [1], [2.0])
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("ReduceMean", ["input"], ["mean"], axes=[1, 2, 3], keepdims=0),
                helper.make_node("Add", ["mean", "bias"], ["logit"]),
            ],
            "binary",
            [_value_info("input", [1, 3, 224, 224])],
            [_value_info("logit", [1])],
            initializer=[bias],
        ),
        opset_imports=[helper.make_opsetid("", 13)],
    )
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, TMP / "binary.onnx")


_build_model()


def make_image_bytes(fmt: str = "PNG") -> bytes:
    image = Image.new("RGB", (100, 100), (120, 80, 70))
    image.paste(Image.new("RGB", (70, 58), (235, 235, 235)), (15, 21))
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return buffer.getvalue()


def data_url(fmt: str = "PNG") -> str:
    mime = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp"}[fmt]
    return f"data:{mime};base64," + base64.b64encode(make_image_bytes(fmt)).decode()


def payload(patient="patient-001", age=25, gender="female", **extra):
    body = {
        "imageData": data_url(),
        "patientId": patient,
        "age": age,
        "gender": gender,
        "metrics": {"brightness": 60, "sharpness": 70, "contrast": 55, "overall": 72},
        "threshold": 70,
        "cameraLabel": "BACK",
        "status": "READY",
    }
    body.update(extra)
    return body


@pytest.fixture()
def client():
    from fastapi.testclient import TestClient
    from app import inference
    from app.db import Base, engine
    from app.main import app

    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    inference.reset_session()
    Path(os.environ["MODEL_PATH"]).unlink(missing_ok=True)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def use_model():
    from app import inference

    def _use(enabled=True):
        target = Path(os.environ["MODEL_PATH"])
        if enabled:
            target.write_bytes((TMP / "binary.onnx").read_bytes())
        else:
            target.unlink(missing_ok=True)
        inference.reset_session()

    return _use
