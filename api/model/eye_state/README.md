# AnemoScan Eye-State Model

The following files are deployed together:

- `anemoscan_eye_state.onnx`: ONNX model graph.
- `anemoscan_eye_state.onnx.data`: external ONNX tensor weights. Keep this file beside the ONNX graph.
- `model_metadata.json`: class mapping, input shape, RGB color order, and normalization values.

This artifact classifies only `open_eye` versus `closed_eye`. It is not the five-class anatomical capture model and does not verify lower palpebral conjunctiva exposure.

Do not delete or replace these files independently. Upload a new model package only after training, validation, and a matching metadata file are complete. GitHub's **Add file via upload** control is a generic directory action, not a missing-file warning.
