# app.py
import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load('iris_model.joblib')
scaler = joblib.load('scaler.joblib')

# UI
st.title("Iris Flower Species Prediction")
st.write("Enter the iris flower's measurements:")

# Input fields
sepal_length = st.number_input("Sepal Length (cm)", 0.0, 10.0, 5.1)
sepal_width = st.number_input("Sepal Width (cm)", 0.0, 10.0, 3.5)
petal_length = st.number_input("Petal Length (cm)", 0.0, 10.0, 1.4)
petal_width = st.number_input("Petal Width (cm)", 0.0, 10.0, 0.2)

# Predict button
if st.button("Predict"):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)[0]

    target_names = ['Setosa', 'Versicolor', 'Virginica']
    st.success(f"Predicted Species: **{target_names[prediction]}**")
