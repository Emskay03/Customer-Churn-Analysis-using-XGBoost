from fastapi import FastAPI
import joblib
import json
import numpy as np


# Create API app
app = FastAPI(title="Churn Prediction API")


# Load trained model
model = joblib.load("model/churn_xgb.pkl")


# Load feature order
with open("model/columns.json") as f:
    columns = json.load(f)


# Health check
@app.get("/")
def home():
    return {"message": "Churn Prediction API is running"}


# Prediction endpoint
@app.post("/predict")
def predict(data: dict):

    # Convert input JSON to list in correct order
    values = [data[col] for col in columns]

    # Convert to numpy array
    arr = np.array(values).reshape(1, -1)

    # Get probability
    prob = model.predict_proba(arr)[0][1]

    # Apply tuned threshold
    churn = int(prob > 0.8)

    return {
        "churn_probability": float(prob),
        "churn_prediction": churn
    }
