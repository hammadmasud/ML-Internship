from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

app = FastAPI(title="Customer Segmentation API")



label_map = {
    0: 'Medium Income -High Spending',
    1: '- Medium Income -Less Spending',
  
     }
model = joblib.load('hierarchical.joblib')


scaler = StandardScaler()  


class CustomerData(BaseModel):
    age: int
    annual_income: float
    spending_score: float
    gender: str  

@app.get("/")
def read_root():
    return {"message": "Customer Segmentation API is running"}

@app.post("/predict")
def predict_cluster(data: CustomerData):
    # Convert input to DataFrame
    gender_female = 1 if data.gender == "Female" else 0
    gender_male = 1 if data.gender == "Male" else 0

    df_input = pd.DataFrame([{
        'Age': data.age,
        'Annual Income (k$)': data.annual_income,
        'Spending Score (1-100)': data.spending_score,
        'Gender_Female': gender_female,
        'Gender_Male': gender_male
    }])

    # Ensure column order
    df_input = df_input[['Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'Gender_Female', 'Gender_Male']]

    scaler = joblib.load("scaler.pkl")
    scaled_input = scaler.transform(df_input)

    # Predict
    cluster = model.predict(scaled_input)[0]
    label = label_map.get(cluster, "Unknown Cluster")

    return {
        "predicted_cluster": int(cluster),
        "cluster_label": label
    }
