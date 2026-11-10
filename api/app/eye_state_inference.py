"""ONNX inference for AnemoScan eye-state detection.

This module is intentionally separate from app.inference, which handles
binary anemia classification.
"""

from __future__ import annotations

import io
import json
import threading
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps
