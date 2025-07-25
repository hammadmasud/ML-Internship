from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

app = FastAPI(title="Customer Clustering and Classification API")

# Load model, scaler, and PCA
model = joblib.load("kmeanspsa.joblib")
scaler = joblib.load("scaler_kmeans.pkl")
pca = joblib.load("pca_kmeans.pkl")  # NEW: Load PCA transformer

# Label mapping

label_map = {
    0: 'Male- Medium Income -Less Spending',
    1: 'Female - Medium Income -High Spending',
    2: 'Female- Medium Income -Less Spending',
    3: 'Male- Medium Income -High Spending',
    }


# Input schema
class CustomerData(BaseModel):
    age: int
    annual_income: float
    spending_score: float
    gender: str

@app.get("/")
def root():
    return {"message": "Customer cluster prediction API is live!"}

@app.post("/predict")
def predict_cluster(data: CustomerData):
    try:
        # Gender encoding
        gender_female = 1 if data.gender.lower() == "female" else 0
        gender_male = 1 if data.gender.lower() == "male" else 0

        # Construct input DataFrame
        input_df = pd.DataFrame([{
            "Age": data.age,
            "Annual Income (k$)": data.annual_income,
            "Spending Score (1-100)": data.spending_score,
            "Gender_Female": gender_female,
            "Gender_Male": gender_male
        }])

        # Ensure correct order
        input_df = input_df[["Age", "Annual Income (k$)", "Spending Score (1-100)", "Gender_Female", "Gender_Male"]]

        # Scale input
        scaled_input = scaler.transform(input_df)

        # Apply PCA
        pca_input = pca.transform(scaled_input)

        # Predict
        cluster = model.predict(pca_input)[0]
        cluster_label = label_map.get(cluster, "Unknown")

        return {
            "predicted_cluster": int(cluster),
            "cluster_label": cluster_label
        }

    except Exception as e:
        return {"error": str(e)}
