# AnemoScan

Non-invasive anemia screening from a photo of the inner eyelid (conjunctiva).
A phone camera takes the photo, a quality check makes sure it is usable, and a trained neural network
predicts the hemoglobin level (g/dL). This is a **screening estimate**, not a diagnosis.

## How it works

```
Patient ID, age, gender  ->  camera (live quality check)  ->  save  ->  analyze  ->  result
                                   web/                        api/       api/ + model.onnx
```

1. Enter the **patient ID, age and gender** (or import a photo instead).
2. The camera only allows a photo that passes the quality and framing check.
3. The scan is saved as `patient-042-scan-1`, `-scan-2`, ... and analyzed automatically.
4. The result screen shows the predicted hemoglobin and the anemia category.

## What is in this repository

```
anemoscan/
  web/            the web app (React + TypeScript + Vite + Tailwind)
  api/            the server (Python + FastAPI + SQLAlchemy + ONNX Runtime)
    model/        put your trained model.onnx here
    tests/        automatic tests
  notebooks/      the two Google Colab training notebooks
  docs/           guides (training, deployment, technology)
  tools/          check_dataset.py: finds spreadsheet problems before training
  Dockerfile      all-in-one build (web + API in one service)
  render.yaml     Render blueprint for the all-in-one service
```

## Where to start

| I want to... | Read |
|---|---|
| train the model, from data to a working app | `docs/TRAINING_GUIDE.md` |
| put the project online (Render, Vercel, Netlify) | `docs/DEPLOYMENT_GUIDE.md` |
| see every technology and endpoint used | `docs/TECH_STACK.md` |

## Run it locally (quick version)

```
# terminal 1: the API
cd api
python -m venv .venv
.venv\\Scripts\\activate        # Windows.   Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload

# terminal 2: the web app
cd web
npm ci
npm run dev                   # open http://localhost:5173
```

Without a `model.onnx` everything works except *Analyze*, which will say no trained model was found.
Details and hosting steps are in `docs/DEPLOYMENT_GUIDE.md`.

## Important

* **No login.** This is a prototype. Use patient IDs, never real names, and do not expose it publicly with real data.
* Prediction quality depends entirely on the training data. See *Understanding your results* in the training guide.
