import streamlit as st
import pandas as pd
from src.components.predict_pipeline import PredictPipeline

st.set_page_config(page_title="Loan Default Predictor", layout="centered")

st.title("Loan Default Prediction")

pipeline = PredictPipeline()

st.header("Applicant Information")

# Numeric inputs
age = st.number_input("Age", min_value=18, max_value=100, value=30)
income = st.number_input("Income", min_value=0, value=5000)
loan_amount = st.number_input("Loan Amount", min_value=0, value=200000)
credit_score = st.number_input("Credit Score", min_value=0, max_value=850, value=600)
months_employed = st.number_input("Months Employed", min_value=0, value=36)
num_credit_lines = st.number_input("Number of Credit Lines", min_value=0, value=3)
interest_rate = st.number_input("Interest Rate (%)", min_value=0.0, value=12.5)
loan_term = st.number_input("Loan Term (months)", min_value=0, value=36)
dti_ratio = st.number_input("Debt-to-Income Ratio", min_value=0.0, value=0.5)

# Categorical inputs
education = st.selectbox("Education", ["High School", "Bachelor", "Master", "PhD"])
employment_type = st.selectbox("Employment Type", ["Salaried", "Self-Employed", "Unemployed"])
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widowed"])
has_mortgage = st.selectbox("Has Mortgage", ["Yes", "No"])
has_dependents = st.selectbox("Has Dependents", ["Yes", "No"])
loan_purpose = st.selectbox("Loan Purpose", ["Car", "Home Improvement", "Business", "Education", "Other"])
has_cosigner = st.selectbox("Has Co-Signer", ["Yes", "No"])

if st.button("Predict"):
    input_data = {
        "Age": age,
        "Income": income,
        "LoanAmount": loan_amount,
        "CreditScore": credit_score,
        "MonthsEmployed": months_employed,
        "NumCreditLines": num_credit_lines,
        "InterestRate": interest_rate,
        "LoanTerm": loan_term,
        "DTIRatio": dti_ratio,
        "Education": education,
        "EmploymentType": employment_type,
        "MaritalStatus": marital_status,
        "HasMortgage": has_mortgage,
        "HasDependents": has_dependents,
        "LoanPurpose": loan_purpose,
        "HasCoSigner": has_cosigner
    }

    result = pipeline.predict(input_data)
    st.success(f"Prediction: {result['prediction']} | Probability of default: {result['probability']:.2f}")
