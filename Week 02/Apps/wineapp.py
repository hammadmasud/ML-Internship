# main_api.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load model
model = joblib.load("wine_model.joblib")

# Define input format
class WineInput(BaseModel):
    volatile_acidity: float
    citric_acid: float
    chlorides: float
    density: float
    sulphates: float
    alcohol: float
    total_sulfur_dioxide: float


@app.post("/predict")
def predict_quality(data: WineInput):
    # Convert input to a 2D array for prediction
    input_features = np.array([[ 
        data.volatile_acidity,
        data.citric_acid,
        data.chlorides,
        data.density,
        data.sulphates,
        data.alcohol,
        data.total_sulfur_dioxide
    ]])

    # Make prediction using the loaded model
    prediction = int(model.predict(input_features)[0])

    # Map binary class to label
    label_map = {0: "Not Good", 1: "Good"}

    return {
        "prediction": prediction,
        "quality": label_map.get(prediction, "Unknown")
    }