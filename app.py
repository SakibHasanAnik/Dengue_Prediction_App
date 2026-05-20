import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Page setup
st.set_page_config(
    page_title="Dengue Predictor",
    page_icon="🦟"
)

st.title("🦟 Dengue Clinical Outcome Predictor")
st.write("Enter patient information to predict dengue status")

# Load model
@st.cache_resource
def load_model():
    model = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

try:
    model, scaler = load_model()
    st.success("✅ Model loaded successfully!")
except:
    st.error("⚠️ Model files not found. Please ensure model.pkl and scaler.pkl are in the same folder.")

# Input form
with st.form("patient_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        age = st.number_input("Age", 1, 100, 30)
        location = st.text_input("Location", "Balasur")
        
    with col2:
        platelet = st.number_input("Platelet Count", 50000, 350000, 130000)
        wbc = st.number_input("WBC Count", 3000, 15000, 6000)
        
    fever = st.selectbox("Fever", ["No", "Yes"])
    fever_duration = st.number_input("Duration of Fever (days)", 0, 14, 3)
    
    col3, col4, col5 = st.columns(3)
    with col3:
        headache = st.selectbox("Headache", ["No", "Yes"])
    with col4:
        muscle_pain = st.selectbox("Muscle Pain", ["No", "Yes"])
    with col5:
        rash = st.selectbox("Rash", ["No", "Yes"])
    
    vomiting = st.selectbox("Vomiting", ["No", "Yes"])
    
    submitted = st.form_submit_button("Predict")
    
if submitted:
    # Convert inputs to numbers
    gender_enc = 1 if gender == "Male" else 0
    fever_enc = 1 if fever == "Yes" else 0
    headache_enc = 1 if headache == "Yes" else 0
    muscle_pain_enc = 1 if muscle_pain == "Yes" else 0
    rash_enc = 1 if rash == "Yes" else 0
    vomiting_enc = 1 if vomiting == "Yes" else 0
    
    # Feature engineering
    symptom_count = sum([fever_enc, headache_enc, muscle_pain_enc, rash_enc, vomiting_enc])
    platelet_wbc_ratio = platelet / (wbc + 1)
    
    # Create feature vector
    features = pd.DataFrame({
        'Gender': [gender_enc],
        'Platelet Count': [platelet],
        'WBC': [wbc],
        'Location': [0.1],  # Simplified for demo
        'Fever': [fever_enc],
        'Duration_of_Fever': [fever_duration],
        'Headache': [headache_enc],
        'Muscle_Pain': [muscle_pain_enc],
        'Rash': [rash_enc],
        'Vomiting': [vomiting_enc],
        'Platelet_WBC_ratio': [platelet_wbc_ratio],
        'Symptom_count': [symptom_count]
    })
    
    # Scale
    scaled_features = scaler.transform(features)
    
    # Predict
    pred = model.predict(scaled_features)[0]
    prob = model.predict_proba(scaled_features)[0]
    
    # Show result
    st.subheader("📊 Prediction Result")
    
    if pred == 1:
        st.error(f"🔴 **Dengue POSITIVE** (Probability: {prob[1]:.2%})")
    else:
        st.success(f"🟢 **Dengue NEGATIVE** (Probability: {prob[0]:.2%})")