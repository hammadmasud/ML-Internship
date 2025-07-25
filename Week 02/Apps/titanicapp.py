# titanic_api.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load saved pipeline
model = joblib.load("titanic_model.joblib")

# Define the input schema
class PassengerInput(BaseModel):
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

@app.post("/predict")
def predict_survival(data: PassengerInput):
    input_array = np.array([[
        data.Pclass,
        data.Fare,
        data.family_size,
        data.Age_Child,
        data.Age_Teen,
        data.Age_Adult,
        data.Age_Aged,
        data.Sex_female,
        data.Sex_male,
        data.Embarked_C,
        data.Embarked_Q,
        data.Embarked_S,
        data.HasCabin
    ]])

    prediction = int(model.predict(input_array)[0])
    label = "Survived" if prediction == 1 else "Did not survive"

    return {
        "prediction": prediction,
        "label": label
    }
