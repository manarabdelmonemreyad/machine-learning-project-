"""
Chronic Kidney Disease Prediction — Streamlit deployment app.

Run locally:
    streamlit run app.py

Deploy for free on Streamlit Community Cloud (share.streamlit.io):
    1. Push this whole project folder to a public GitHub repo.
    2. Go to share.streamlit.io -> New app -> pick the repo -> main file: app.py -> Deploy.
    3. You will get a public URL (this is the link required by the project criteria).
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json

st.set_page_config(page_title="CKD Risk Predictor", page_icon="🩺", layout="centered")


@st.cache_resource
def load_artifacts():
    with open("best_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("label_encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
    with open("selected_features.json") as f:
        selected_features = json.load(f)
    with open("feature_medians.json") as f:
        medians = json.load(f)
    return model, scaler, encoders, selected_features, medians


model, scaler, encoders, selected_features, medians = load_artifacts()

st.title("🩺 Chronic Kidney Disease Risk Predictor")
st.write(
    "Enter a patient's clinical values below. The model was trained on the UCI "
    "Chronic Kidney Disease dataset (400 patients) using a Logistic Regression / "
    "Random Forest / SVM comparison — see the project notebook for the full pipeline."
)
st.caption(
    "⚠️ Educational / course-project demo only. This is **not** a certified "
    "medical device and must not be used for real diagnosis."
)

st.header("Patient inputs")

col1, col2 = st.columns(2)
with col1:
    sg = st.selectbox("Urine Specific Gravity (sg)", [1.005, 1.010, 1.015, 1.020, 1.025], index=2)
    al = st.slider("Albumin (al)", 0, 5, 0)
    bgr = st.number_input("Blood Glucose Random (bgr, mg/dl)", value=120.0, step=1.0)
    hemo = st.number_input("Hemoglobin (hemo, g/dl)", value=13.0, step=0.1)
    pcv = st.number_input("Packed Cell Volume (pcv, %)", value=42.0, step=1.0)
    rc = st.number_input("Red Blood Cell count (rc, millions/cmm)", value=5.0, step=0.1)

with col2:
    pc = st.selectbox("Pus Cell (pc)", ["normal", "abnormal"])
    htn = st.selectbox("Hypertension (htn)", ["yes", "no"])
    dm = st.selectbox("Diabetes Mellitus (dm)", ["yes", "no"])
    appet = st.selectbox("Appetite (appet)", ["good", "poor"])
    pe = st.selectbox("Pedal Edema (pe)", ["yes", "no"])
    bu = st.number_input("Blood Urea (bu, mg/dl)", value=40.0, step=1.0)
    sc = st.number_input("Serum Creatinine (sc, mg/dl)", value=1.2, step=0.1)

if st.button("Predict CKD risk", type="primary"):
    # bun/creatinine ratio - the engineered feature from the notebook
    ratio = bu / sc if sc != 0 else medians.get("bu", 40.0) / medians.get("sc", 1.2)

    raw = {
        "sg": sg, "al": al, "pc": pc, "bgr": bgr, "hemo": hemo, "pcv": pcv, "rc": rc,
        "htn": htn, "dm": dm, "appet": appet, "pe": pe,
        "bun_creatinine_ratio": ratio,
    }

    row = {}
    for feat in selected_features:
        if feat in encoders:
            row[feat] = encoders[feat].transform([raw[feat]])[0]
        else:
            row[feat] = raw[feat]

    X_new = pd.DataFrame([row])[selected_features]
    X_new_scaled = scaler.transform(X_new)

    pred = model.predict(X_new_scaled)[0]
    proba = model.predict_proba(X_new_scaled)[0][1]

    st.header("Result")
    if pred == 1:
        st.error(f"⚠️ Model predicts **CKD** (probability {proba:.1%})")
    else:
        st.success(f"✅ Model predicts **No CKD** (probability of CKD {proba:.1%})")

    st.progress(min(max(proba, 0.0), 1.0))
    st.caption("Probability shown is the model's estimated likelihood of CKD, not a diagnosis.")

st.divider()
st.caption("Source: CBIO313 course project — see the GitHub repository README for full methodology.")
