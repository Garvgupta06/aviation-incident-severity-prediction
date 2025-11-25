import streamlit as st
import pandas as pd
import joblib
import numpy as np

# =========================================================
# MUST be first Streamlit command
# =========================================================
st.set_page_config(page_title="Aviation Severity Predictor", layout="centered")

# =========================================================
# Load Model
# =========================================================
@st.cache_resource
def load_model():
    return joblib.load("models/aviation_severity_model.pkl")

model = load_model()

# =========================================================
# Load Dataset for Defaults & Dropdown Options
# =========================================================
@st.cache_resource
def load_data():
    df = pd.read_csv("data/cleaned/final_aviation_data.csv")
    if "Unnamed: 0.1" in df.columns:
        df = df.drop(columns=["Unnamed: 0.1"])
    return df

df_train = load_data()

# =========================================================
# Helper for dropdown options
# =========================================================
def cat_options(col):
    return sorted(df_train[col].dropna().unique().tolist())

# =========================================================
# Default values for removed features
# =========================================================
default_make = df_train["make"].mode()[0]
default_category = df_train["aircraft_category"].mode()[0]
default_engine = df_train["engine_type"].mode()[0]
default_amateur_built = df_train["amateur_built"].mode()[0]
default_month = df_train["month"].mode()[0]
default_day = df_train["day"].mode()[0]

# =========================================================
# UI
# =========================================================

st.title("✈️ Aviation Incident Severity Prediction")
st.write("Estimate the fatality risk of an aviation incident using machine learning.")

st.divider()

# ============================
# Input Form (Simplified UI)
# ============================
with st.form("prediction_form"):

    col1, col2 = st.columns(2)

    country = col1.selectbox("Country", cat_options("country"))

    aircraft_damage = col2.selectbox("Aircraft Damage", cat_options("aircraft_damage"))
    purpose_of_flight = col1.selectbox("Purpose of Flight", cat_options("purpose_of_flight"))

    weather_condition = col2.selectbox("Weather Condition", cat_options("weather_condition"))
    broad_phase_of_flight = col1.selectbox("Phase of Flight", cat_options("broad_phase_of_flight"))

    number_of_engines = col2.number_input("Number of Engines", min_value=0, max_value=10, value=1)

    year = col1.number_input("Year of Event", min_value=1945, max_value=2030, value=2010)

    submitted = st.form_submit_button("Predict Risk")

# =========================================================
# Prediction
# =========================================================
if submitted:

    # Build input row with defaults for removed features
    input_data = pd.DataFrame([{
        "country": country,
        "aircraft_damage": aircraft_damage,
        "aircraft_category": default_category,
        "make": default_make,
        "amateur_built": default_amateur_built,
        "number_of_engines": number_of_engines,
        "engine_type": default_engine,
        "purpose_of_flight": purpose_of_flight,
        "weather_condition": weather_condition,
        "broad_phase_of_flight": broad_phase_of_flight,
        "year": year,
        "month": default_month,
        "day": default_day
    }])

    # Run prediction
    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    st.subheader("🔎 Prediction Result")

    if pred == 1:
        st.error(f"🚨 **High Fatality Risk**\nProbability: **{prob*100:.2f}%**")
    else:
        st.success(f"🟢 **Low Fatality Risk**\nFatality Probability: **{prob*100:.2f}%**")

    st.progress(float(prob))

    st.caption("Prediction generated using a trained XGBoost model with aviation incident data.")
