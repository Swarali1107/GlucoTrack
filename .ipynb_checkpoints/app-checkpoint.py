import streamlit as st
import numpy as np
import joblib

# Load trained model
model = joblib.load("diabetes_model.pkl")

st.title("🔍 Diabetes Risk Predictor")

# Input fields
glucose = st.number_input("Glucose", min_value=0)
bmi = st.number_input("BMI", min_value=0.0)
age = st.number_input("Age", min_value=0)
blood_pressure = st.number_input("Blood Pressure", min_value=0)
insulin = st.number_input("Insulin", min_value=0)
skin_thickness = st.number_input("Skin Thickness", min_value=0)
pregnancies = st.number_input("Pregnancies", min_value=0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, step=0.01)

# Predict
if st.button("Predict"):
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                            insulin, bmi, dpf, age]])
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.error("⚠️ High Risk of Diabetes")
    else:
        st.success("✅ Low Risk of Diabetes")
