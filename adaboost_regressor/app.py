import streamlit as st
import pandas as pd
import pickle
import os

st.title("Insurance Cost Prediction using AdaBoost Regressor")

# Check model existence
model_path = "models/adaboost_model.pkl"

if not os.path.exists(model_path):
    st.error(f"Model file not found: {model_path}")
    st.write("Current Directory:", os.getcwd())
    st.write("Files:", os.listdir())
    st.stop()

# Load model
with open(model_path, "rb") as f:
    model = pickle.load(f)

# User Inputs
age = st.number_input("Age", min_value=18, max_value=100, value=25)

sex = st.selectbox(
    "Sex",
    ["male", "female"]
)

bmi = st.number_input(
    "BMI",
    min_value=10.0,
    max_value=60.0,
    value=25.0
)

children = st.number_input(
    "Children",
    min_value=0,
    max_value=10,
    value=0
)

smoker = st.selectbox(
    "Smoker",
    ["yes", "no"]
)

region = st.selectbox(
    "Region",
    ["northeast", "northwest", "southeast", "southwest"]
)

if st.button("Predict Insurance Cost"):

    input_data = pd.DataFrame({
        "age": [age],
        "bmi": [bmi],
        "children": [children],
        "sex_male": [1 if sex == "male" else 0],
        "smoker_yes": [1 if smoker == "yes" else 0],
        "region_northwest": [1 if region == "northwest" else 0],
        "region_southeast": [1 if region == "southeast" else 0],
        "region_southwest": [1 if region == "southwest" else 0]
    })

    prediction = model.predict(input_data)

    st.success(
        f"Predicted Insurance Charges: ₹ {prediction[0]:,.2f}"
    )
