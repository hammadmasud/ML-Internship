import streamlit as st
import joblib
import numpy as np

# Load pipeline and feature names
pipeline = joblib.load('boston_pipeline.joblib')
features = joblib.load('feature_names.joblib')

# Flatten if nested list (prevents "unhashable type: list" error)
if isinstance(features[0], list):
    features = features[0]

st.title("🏠 Boston Housing Price Prediction")
st.write("Input the housing features to predict the **MEDV (house price)**")

# Create input fields for each feature
user_inputs = {}
for feature in features:
    user_inputs[feature] = st.number_input(f"{feature}", value=1.0, step=0.1)

# Predict button
if st.button("Predict"):
    input_array = np.array([list(user_inputs.values())])
    prediction = pipeline.predict(input_array)[0]
    st.success(f"🏡 Predicted House Price (MEDV): **{prediction:.2f}**")



