# cancer_api.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load saved pipeline model
model = joblib.load("breast_cancer.joblib")

# Define the input schema based on your selected features
class CancerInput(BaseModel):
    concave_points_worst: float
    perimeter_worst: float
    concave_points_mean: float
    radius_worst: float
    perimeter_mean: float
    area_worst: float
    radius_mean: float
    area_mean: float
    concavity_mean: float
    concavity_worst: float
    compactness_mean: float
    compactness_worst: float
    radius_se: float
    perimeter_se: float
    area_se: float
    texture_worst: float
    smoothness_worst: float
    symmetry_worst: float
    texture_mean: float
    concave_points_se: float
    smoothness_mean: float
    symmetry_mean: float

@app.post("/predict")
def predict_diagnosis(data: CancerInput):
    input_data = np.array([[
        data.concave_points_worst,
        data.perimeter_worst,
        data.concave_points_mean,
        data.radius_worst,
        data.perimeter_mean,
        data.area_worst,
        data.radius_mean,
        data.area_mean,
        data.concavity_mean,
        data.concavity_worst,
        data.compactness_mean,
        data.compactness_worst,
        data.radius_se,
        data.perimeter_se,
        data.area_se,
        data.texture_worst,
        data.smoothness_worst,
        data.symmetry_worst,
        data.texture_mean,
        data.concave_points_se,
        data.smoothness_mean,
        data.symmetry_mean
    ]])

    prediction = int(model.predict(input_data)[0])
    label = "Malignant" if prediction == 1 else "Benign"

    return {
        "prediction": prediction,
        "diagnosis": label
    }
