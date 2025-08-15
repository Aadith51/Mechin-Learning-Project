import streamlit as st
import joblib
import numpy as np
import pandas as pd

# === Load model and preprocessors ===
model = joblib.load("heart_attack_prediction.pkl")
scaler = joblib.load("scaler.pkl")
feature_order = joblib.load("features.pkl")

# Ensure feature_order is a list
if not isinstance(feature_order, list):
    feature_order = list(feature_order)

# === Page config ===
st.set_page_config(page_title="Heart Attack Risk Predictor", layout="centered")
st.title("❤ Heart Attack Risk Prediction App")

st.markdown("Provide the patient's health data below to predict the risk of a heart attack.")

# === User Inputs ===
st.header("🧾 Patient Information")
age = st.number_input("Age", min_value=1, max_value=120)
gender = st.selectbox("Gender", ['Male', 'Female'])
hypertension = st.selectbox("Hypertension", ['No', 'Yes'])
diabetes = st.selectbox("Diabetes", ['No', 'Yes'])
cholesterol_level = st.number_input("Total Cholesterol Level")
obesity = st.selectbox("Obesity", ['No', 'Yes'])
smoking_status = st.selectbox("Smoking Status", ['Never', 'Former', 'Current'])
physical_activity = st.selectbox("Physical Activity", ['Low', 'Moderate', 'High'])
stress_level = st.selectbox("Stress Level", ['Low', 'Medium', 'High'])
blood_pressure_systolic = st.number_input("Systolic Blood Pressure")
blood_pressure_diastolic = st.number_input("Diastolic Blood Pressure")
fasting_blood_sugar = st.number_input("Fasting Blood Sugar")
cholesterol_hdl = st.number_input("HDL Cholesterol")
cholesterol_ldl = st.number_input("LDL Cholesterol")
triglycerides = st.number_input("Triglycerides")
previous_heart_disease = st.selectbox("Previous Heart Disease", ['No', 'Yes'])

# === Encode and prepare data ===
data = {
    "age": age,
    "gender": 1 if gender == "Male" else 0,
    "hypertension": 1 if hypertension == "Yes" else 0,
    "diabetes": 1 if diabetes == "Yes" else 0,
    "cholesterol_level": cholesterol_level,
    "obesity": 1 if obesity == "Yes" else 0,
    "smoking_status": 0 if smoking_status == "Never" else (1 if smoking_status == "Former" else 2),
    "physical_activity": 0 if physical_activity == "Low" else (1 if physical_activity == "Moderate" else 2),
    "stress_level": 0 if stress_level == "Low" else (1 if stress_level == "Medium" else 2),
    "blood_pressure_systolic": blood_pressure_systolic,
    "blood_pressure_diastolic": blood_pressure_diastolic,
    "fasting_blood_sugar": fasting_blood_sugar,
    "cholesterol_hdl": cholesterol_hdl,
    "cholesterol_ldl": cholesterol_ldl,
    "triglycerides": triglycerides,
    "previous_heart_disease": 1 if previous_heart_disease == "Yes" else 0,
}

# Create DataFrame and scale
input_df = pd.DataFrame([data])[feature_order]
scaled_input = scaler.transform(input_df)

# === Prediction ===
st.subheader("📊 Prediction Result")
threshold = 0.4  # Adjust threshold as needed

if st.button("Predict Risk"):
    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][1]  # probability of class 1 (high risk)

    st.write(f"🔍 *Prediction Probability (High Risk):* {probability:.2%}")

    if probability >= threshold:
        st.error(f"⚠ High Risk of Heart Attack\n\nModel Confidence: *{probability:.2%}*")
    else:
        st.success(f"✅ Low Risk of Heart Attack\n\nModel Confidence: *{probability:.2%}*")

    st.markdown("---")
    st.caption("🔧 Threshold set to {:.0f}%. Adjust based on model performance.".format(threshold * 100))



