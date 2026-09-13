import streamlit as st
import joblib
import pandas as pd
import numpy as np

st.set_page_config(page_title="Social Media Impact on Health", layout="centered")

@st.cache_resource
def load_models():
    return {
        "Logistic Regression": joblib.load("classifier_model.joblib"),
        "Gradient Boosting Regressor": joblib.load("regression_model.joblib"),
    }

models = load_models()

model_lr = models["Logistic Regression"]
model_gbr = models["Gradient Boosting Regressor"]

st.title("Social Media Impact on Health")
st.write("Fill in the details below to get an estimated mental health impact rating.")

st.subheader("Your Details")

col1, col2 = st.columns(2)
with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age", min_value=0, value=15)
    academic_level = st.selectbox("Academic Level", ["High School", "Undergraduate","Graduate"])
    avg_daily_usage_hours = st.number_input("Average Daily Usage (Hours)", min_value = 0.0, value = 3.5)
    most_used_platform = st.selectbox("Platform", ["Facebook","Instagram","LinkedIn","Snapchat", "Youtube", "Tiktok" , "Twitter"])
    sleep_hours_per_night = st.number_input("Sleep Hours / Night", min_value = 0, value = 8)


if st.button("Submit", type="primary"):
    input_df = pd.DataFrame([{
        "Gender": gender,
        "Age": age,
        "Academic_Level": academic_level,
        "Avg_Daily_Usage_Hours": avg_daily_usage_hours,
        "Most_Used_Platform": most_used_platform,
        "Sleep_Hours_Per_Night": sleep_hours_per_night,
    }])

    pred_sentiment = model_lr.predict(input_df)[0]
    pred_score = model_gbr.predict(input_df)[0]
    with col2:
            st.subheader("Result")
            st.success(f"### Predicted sentiment: {pred_sentiment}")
            st.success(f"### Predicted score: {pred_score:.2f}")