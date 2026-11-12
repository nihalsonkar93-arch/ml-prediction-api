
import joblib
import numpy as np

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sklearn.datasets import load_breast_cancer

# Load the trained model
model = joblib.load("cancer_model.joblib")

# Load class names
dataset = load_breast_cancer()
class_names = dataset.target_names

# Create FastAPI application
app = FastAPI(title="ML Prediction API")

# Validate incoming data
class PredictionInput(BaseModel):
    features: list[float] = Field(min_length=30, max_length=30)

# Prediction endpoint
@app.post("/predict")
def predict(data: PredictionInput):

    features = np.array(data.features)

    if not np.isfinite(features).all():
        raise HTTPException(
            status_code=400,
            detail="Features must contain finite numbers."
        )

    prediction = int(model.predict([data.features])[0])
    probabilities = model.predict_proba([data.features])[0]

    return {
        "prediction": class_names[prediction],
        "confidence": float(probabilities[prediction])
    }
