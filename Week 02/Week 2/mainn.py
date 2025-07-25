from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load model and model name
model = joblib.load("titanic_model.joblib")
with open("model_name.txt", "r") as f:
    model_name = f.read()

# Create FastAPI app
app = FastAPI(title="Titanic Survival Predictor", description=f"Using model: {model_name}")

# Define input schema
class Passenger(BaseModel):
    Pclass: int
    Fare: float
    family_size: int
    Age_Child: int
    Age_Teen: int
    Age_Adult: int
    Age_Aged: int
    Sex_female: int
    Sex_male: int
    Embarked_C: int
    Embarked_Q: int
    Embarked_S: int
    HasCabin: int

# Prediction route
@app.post("/predict")
def predict(data: Passenger):
    features = np.array([[
        data.Pclass, data.Fare, data.family_size,
        data.Age_Child, data.Age_Teen, data.Age_Adult, data.Age_Aged,
        data.Sex_female, data.Sex_male,
        data.Embarked_C, data.Embarked_Q, data.Embarked_S,
        data.HasCabin
    ]])
    prediction = model.predict(features)
    result = "Survived" if prediction[0] == 1 else "Did Not Survive"
    return {
        "model_used": model_name,
        "prediction": int(prediction[0]),
        "result": result
    }
