import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder


model = pickle.load(open("customer_churn_model.pkl", "rb"))
feature_names = pickle.load(open("feature_names.pkl", "rb"))


st.set_page_config(page_title="Customer Churn Prediction", page_icon="📊", layout="centered")

st.title("📊 Customer Churn Prediction App")
st.markdown("This web app predicts whether a telecom customer is likely to **churn** or **stay** based on their details.")

st.markdown("---")


st.header(" Enter Customer Details")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", ["Yes", "No"])
    partner = st.selectbox("Has Partner", ["Yes", "No"])
    dependents = st.selectbox("Has Dependents", ["Yes", "No"])
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])

with col2:
    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    payment_method = st.selectbox("Payment Method", [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ])
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
    monthly_charges = st.number_input("Monthly Charges (Rupees)", min_value=0.0, max_value=2000.0, value=70.0)
    total_charges = st.number_input("Total Charges (Rupees)", min_value=0.0, max_value=100000.0, value=800.0)

st.markdown("---")


input_data = pd.DataFrame({
    "gender": [gender],
    "SeniorCitizen": [1 if senior == "Yes" else 0],
    "Partner": [1 if partner == "Yes" else 0],
    "Dependents": [1 if dependents == "Yes" else 0],
    "PhoneService": [1 if phone_service == "Yes" else 0],
    "MultipleLines": [multiple_lines],
    "InternetService": [internet_service],
    "OnlineSecurity": [online_security],
    "OnlineBackup": [online_backup],
    "DeviceProtection": [device_protection],
    "TechSupport": [tech_support],
    "StreamingTV": [streaming_tv],
    "StreamingMovies": [streaming_movies],
    "PaperlessBilling": [paperless_billing],
    "Contract": [contract],
    "PaymentMethod": [payment_method],
    "tenure": [tenure],
    "MonthlyCharges": [monthly_charges],
    "TotalCharges": [total_charges]
})


for col in input_data.select_dtypes(include="object").columns:
    le = LabelEncoder()
    input_data[col] = le.fit_transform(input_data[col])


input_data = input_data.reindex(columns=feature_names, fill_value=0)


if st.button("🔍 Predict Churn"):
    try:
        prediction = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0][1] * 100

        st.subheader("🎯 Prediction Result")
        if prediction == 1:
            st.error(f"❌ The customer is **likely to churn.** (Confidence: {proba:.2f}%)")
        else:
            st.success(f"✅ The customer is **not likely to churn.** (Confidence: {100-proba:.2f}%)")

        st.markdown("### 🧾 Input Summary")
        st.dataframe(input_data)

    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")
