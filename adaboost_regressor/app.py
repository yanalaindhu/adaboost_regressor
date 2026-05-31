import streamlit as st
import pickle
import pandas as pd

# Load model
with open("models/adaboost_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Insurance Cost Prediction")
st.write("AdaBoost Regressor")

age = st.number_input("Age", 18, 100, 25)

sex = st.selectbox("Sex", ["male", "female"])

bmi = st.number_input("BMI", 10.0, 60.0, 25.0)

children = st.number_input("Children", 0, 10, 0)

smoker = st.selectbox("Smoker", ["yes", "no"])

region = st.selectbox(
    "Region",
    ["northeast", "northwest", "southeast", "southwest"]
)

if st.button("Predict"):

    data = {
        "age": age,
        "bmi": bmi,
        "children": children,
        "sex_male": 1 if sex == "male" else 0,
        "smoker_yes": 1 if smoker == "yes" else 0,
        "region_northwest": 1 if region == "northwest" else 0,
        "region_southeast": 1 if region == "southeast" else 0,
        "region_southwest": 1 if region == "southwest" else 0
    }

    df = pd.DataFrame([data])

    prediction = model.predict(df)

    st.success(f"Predicted Insurance Charge: ₹ {prediction[0]:,.2f}")