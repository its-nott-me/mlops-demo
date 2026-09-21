"""
Section 5.5 / 7 - Serve the trained model with FastAPI.

Run (after running sklearn_pipeline.py at least once, to create model.joblib):
    uvicorn main:app --reload
Then open http://127.0.0.1:8000/docs and screenshot the Swagger UI.
"""
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="MLOps Demo - Income Prediction API")
model = joblib.load("model.joblib")  # produced by sklearn_pipeline.py

FEATURE_NAMES = [
    "Age", "Workclass", "Education-Num", "Marital Status", "Occupation",
    "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
    "Hours per week", "Country",
]


class PredictRequest(BaseModel):
    features: list[float]


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    df = pd.DataFrame([req.features], columns=FEATURE_NAMES)
    prediction = model.predict(df)[0]
    return {"prediction": bool(prediction)}
