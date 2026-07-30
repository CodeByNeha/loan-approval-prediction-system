"""
Loan Approval Prediction System — Streamlit Interface
Loads the model trained in Loan_Approval_Source_Code.ipynb and serves
real-time predictions for loan applicants.

Run with:  streamlit run app.py
"""

import numpy as np
import pandas as pd
import joblib
import streamlit as st

st.set_page_config(page_title="Loan Approval Prediction System", page_icon="💰", layout="centered")


@st.cache_resource
def load_artifacts():
    model = joblib.load("models/loan_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    label_encoders = joblib.load("models/label_encoders.pkl")
    feature_columns = joblib.load("models/feature_columns.pkl")
    return model, scaler, label_encoders, feature_columns


def preprocess_input(raw, scaler, label_encoders, feature_columns):
    df = pd.DataFrame([raw])

    df["Dependents"] = df["Dependents"].replace("3+", 3).astype(int)

    for col, le in label_encoders.items():
        df[col] = le.transform(df[col])

    df = pd.get_dummies(df, columns=["Property_Area"])
    for col in feature_columns:
        if col.startswith("Property_Area_") and col not in df.columns:
            df[col] = False

    df["TotalIncome"] = df["ApplicantIncome"] + df["CoapplicantIncome"]
    df["DebtToIncomeRatio"] = df["LoanAmount"] / (df["TotalIncome"] / 1000 + 1e-3)
    df["LoanAmount_log"] = np.log1p(df["LoanAmount"])
    df["TotalIncome_log"] = np.log1p(df["TotalIncome"])

    numeric_cols = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term",
                     "TotalIncome", "DebtToIncomeRatio", "LoanAmount_log", "TotalIncome_log"]
    df[numeric_cols] = scaler.transform(df[numeric_cols])

    df = df.reindex(columns=feature_columns, fill_value=False)
    return df


def main():
    st.title("💰 Loan Approval Prediction System")
    st.caption("Enter applicant details to get a real-time approval prediction with confidence score.")

    model, scaler, label_encoders, feature_columns = load_artifacts()

    with st.form("applicant_form"):
        col1, col2 = st.columns(2)
        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            married = st.selectbox("Married", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
            education = st.selectbox("Education", ["Graduate", "Not Graduate"])
            self_employed = st.selectbox("Self Employed", ["Yes", "No"])
            property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
        with col2:
            applicant_income = st.number_input("Applicant Income (monthly)", min_value=0, value=5000, step=100)
            coapplicant_income = st.number_input("Co-applicant Income (monthly)", min_value=0, value=0, step=100)
            loan_amount = st.number_input("Loan Amount (in thousands)", min_value=1, value=120, step=1)
            loan_term = st.selectbox("Loan Term (months)", [360, 180, 120, 84, 60, 300], index=0)
            credit_history = st.selectbox("Credit History Meets Guidelines?", ["Yes", "No"])

        submitted = st.form_submit_button("Predict")

    if submitted:
        raw = {
            "Gender": gender,
            "Married": married,
            "Dependents": dependents,
            "Education": education,
            "Self_Employed": self_employed,
            "ApplicantIncome": applicant_income,
            "CoapplicantIncome": coapplicant_income,
            "LoanAmount": loan_amount,
            "Loan_Amount_Term": loan_term,
            "Credit_History": 1.0 if credit_history == "Yes" else 0.0,
            "Property_Area": property_area,
        }

        X_input = preprocess_input(raw, scaler, label_encoders, feature_columns)
        pred = model.predict(X_input)[0]
        proba = model.predict_proba(X_input)[0][pred]

        st.divider()
        if pred == 1:
            st.success(f"✅ Loan Approved — Confidence: {proba:.1%}")
        else:
            st.error(f"❌ Loan Rejected — Confidence: {proba:.1%}")


if __name__ == "__main__":
    main()
