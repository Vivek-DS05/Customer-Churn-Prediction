import streamlit as st
import pandas as pd
import joblib

model  = joblib.load('logistic_regression_model.joblib')
scaler = joblib.load('customer_data_scaler.joblib')
FEATURES = [
    'gender', 'SeniorCitizen', 'Partner',
    'Dependents', 'tenure', 'PhoneService',
    'PaperlessBilling', 'MonthlyCharges',
    'TotalCharges',
    'InternetService_Fiber optic',
    'InternetService_No',
    'Contract_One year',
    'Contract_Two year',
    'PaymentMethod_Credit card (automatic)',
    'PaymentMethod_Electronic check',
    'PaymentMethod_Mailed check',
    'MultipleLines_No phone service',
    'MultipleLines_Yes',
    'OnlineSecurity_No internet service',
    'OnlineSecurity_Yes',
    'OnlineBackup_No internet service',
    'OnlineBackup_Yes',
    'DeviceProtection_No internet service',
    'DeviceProtection_Yes',
    'TechSupport_No internet service',
    'TechSupport_Yes',
    'StreamingTV_No internet service',
    'StreamingTV_Yes',
    'StreamingMovies_No internet service',
    'StreamingMovies_Yes'
]

st.title("Customer Churn Predictor")
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Personal")
    gender           = st.selectbox("Gender",
                       ["Male", "Female"])
    senior_citizen   = st.selectbox("Senior Citizen",
                       ["No", "Yes"])
    partner          = st.selectbox("Partner",
                       ["Yes", "No"])
    dependents       = st.selectbox("Dependents",
                       ["Yes", "No"])
    tenure           = st.slider("Tenure (months)",
                       0, 72, 12)

with col2:
    st.subheader("Services")
    phone_service    = st.selectbox("Phone Service",
                       ["Yes", "No"])
    multiple_lines   = st.selectbox("Multiple Lines",
                       ["No", "Yes", "No phone service"])
    internet_service = st.selectbox("Internet Service",
                       ["DSL", "Fiber optic", "No"])
    online_security  = st.selectbox("Online Security",
                       ["Yes", "No", "No internet service"])
    online_backup    = st.selectbox("Online Backup",
                       ["Yes", "No", "No internet service"])
    device_protection= st.selectbox("Device Protection",
                       ["Yes", "No", "No internet service"])
    tech_support     = st.selectbox("Tech Support",
                       ["Yes", "No", "No internet service"])
    streaming_tv     = st.selectbox("Streaming TV",
                       ["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies",
                       ["Yes", "No", "No internet service"])

with col3:
    st.subheader("Account")
    contract         = st.selectbox("Contract",
                       ["Month-to-month",
                        "One year", "Two year"])
    paperless        = st.selectbox("Paperless Billing",
                       ["Yes", "No"])
    payment          = st.selectbox("Payment Method",
                       ["Electronic check",
                        "Mailed check",
                        "Bank transfer (automatic)",
                        "Credit card (automatic)"])
    monthly_charges  = st.number_input(
                       "Monthly Charges ($)",
                       0.0, 200.0, 65.0)
    total_charges    = st.number_input(
                       "Total Charges ($)",
                       0.0, 10000.0,
                       float(monthly_charges * tenure))

def build_features():

    data = {
        'gender'          : 1 if gender == 'Male' else 0,
        'SeniorCitizen'   : 1 if senior_citizen == 'Yes' else 0,
        'Partner'         : 1 if partner == 'Yes' else 0,
        'Dependents'      : 1 if dependents == 'Yes' else 0,
        'tenure'          : tenure,
        'PhoneService'    : 1 if phone_service == 'Yes' else 0,
        'PaperlessBilling': 1 if paperless == 'Yes' else 0,
        'MonthlyCharges'  : monthly_charges,
        'TotalCharges'    : total_charges,

        'InternetService_Fiber optic':
            1 if internet_service == 'Fiber optic' else 0,
        'InternetService_No':
            1 if internet_service == 'No' else 0,

        'Contract_One year':
            1 if contract == 'One year' else 0,
        'Contract_Two year':
            1 if contract == 'Two year' else 0,

        'PaymentMethod_Credit card (automatic)':
            1 if payment == 'Credit card (automatic)' else 0,
        'PaymentMethod_Electronic check':
            1 if payment == 'Electronic check' else 0,
        'PaymentMethod_Mailed check':
            1 if payment == 'Mailed check' else 0,

        'MultipleLines_No phone service':
            1 if multiple_lines == 'No phone service' else 0,
        'MultipleLines_Yes':
            1 if multiple_lines == 'Yes' else 0,

        'OnlineSecurity_No internet service':
            1 if online_security == 'No internet service' else 0,
        'OnlineSecurity_Yes':
            1 if online_security == 'Yes' else 0,

        'OnlineBackup_No internet service':
            1 if online_backup == 'No internet service' else 0,
        'OnlineBackup_Yes':
            1 if online_backup == 'Yes' else 0,

        'DeviceProtection_No internet service':
            1 if device_protection == 'No internet service' else 0,
        'DeviceProtection_Yes':
            1 if device_protection == 'Yes' else 0,

        'TechSupport_No internet service':
            1 if tech_support == 'No internet service' else 0,
        'TechSupport_Yes':
            1 if tech_support == 'Yes' else 0,

        'StreamingTV_No internet service':
            1 if streaming_tv == 'No internet service' else 0,
        'StreamingTV_Yes':
            1 if streaming_tv == 'Yes' else 0,

        'StreamingMovies_No internet service':
            1 if streaming_movies == 'No internet service' else 0,
        'StreamingMovies_Yes':
            1 if streaming_movies == 'Yes' else 0,
    }

    input_df = pd.DataFrame([data])
    input_df = input_df[FEATURES]

    return input_df

if st.button("Predict"):
    input_df     = build_features()

    input_scaled = scaler.transform(input_df)

    prediction   = model.predict(input_scaled)[0]
    churn_prob   = model.predict_proba(
                   input_scaled)[0][1] * 100

    if churn_prob >= 70:
        risk = "HIGH RISK"
    elif churn_prob >= 40:
        risk = "MEDIUM RISK"
    else:
        risk = "LOW RISK"
    r1, r2, r3 = st.columns(3)

    with r1:
        if prediction == 1:
            st.error("WILL CHURN")
        else:
            st.success("WILL NOT CHURN")

    with r2:
        st.info(f"{risk}")

    with r3:
        st.metric("Churn Probability",
                  f"{churn_prob:.1f}%")