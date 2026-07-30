"""
Generates a synthetic loan-application dataset that mirrors the schema of the
Kaggle "Loan Prediction Problem Dataset"
(https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset).

This is used so the notebook can be run and demonstrated end-to-end even
before the real Kaggle CSV is downloaded. To use the real data instead,
download train.csv from the link above, save it as data/loan_data.csv,
and skip running this script.
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 614  # matches the real dataset's training set size

gender = np.random.choice(["Male", "Female"], N, p=[0.8, 0.2])
married = np.random.choice(["Yes", "No"], N, p=[0.65, 0.35])
dependents = np.random.choice(["0", "1", "2", "3+"], N, p=[0.58, 0.17, 0.17, 0.08])
education = np.random.choice(["Graduate", "Not Graduate"], N, p=[0.78, 0.22])
self_employed = np.random.choice(["Yes", "No"], N, p=[0.14, 0.86])
property_area = np.random.choice(["Urban", "Semiurban", "Rural"], N, p=[0.38, 0.38, 0.24])

applicant_income = np.random.gamma(shape=3.0, scale=2000, size=N).round(0)
coapplicant_income = np.random.gamma(shape=1.5, scale=1200, size=N).round(0)
coapplicant_income[np.random.rand(N) < 0.4] = 0  # many applicants have no co-applicant

loan_amount = (0.02 * (applicant_income + coapplicant_income) +
               np.random.normal(0, 30, N)).clip(9, 700).round(0)

loan_term = np.random.choice([360, 180, 120, 60, 300, 84], N, p=[0.72, 0.08, 0.06, 0.06, 0.05, 0.03])
credit_history = np.random.choice([1.0, 0.0], N, p=[0.84, 0.16])

# introduce a small number of missing values, mirroring the real dataset
gender = gender.astype(object)
married = married.astype(object)
self_employed = self_employed.astype(object)
loan_amount = loan_amount.astype(float)
loan_term = loan_term.astype(float)

for col_arr, frac in [(gender, 0.02), (married, 0.005), (self_employed, 0.05),
                       (loan_amount, 0.035), (loan_term, 0.023), (credit_history, 0.08)]:
    idx = np.random.choice(N, int(N * frac), replace=False)
    col_arr[idx] = np.nan

# target variable: approval likelihood driven mainly by credit history and income
total_income = applicant_income + coapplicant_income
approval_score = (
    3.2 * np.nan_to_num(credit_history, nan=0.5)
    + 0.00012 * total_income
    - 0.004 * loan_amount
    + np.random.normal(0, 1.1, N)
)
loan_status = np.where(approval_score > 2.0, "Y", "N")

df = pd.DataFrame({
    "Loan_ID": [f"LP{str(i).zfill(6)}" for i in range(1, N + 1)],
    "Gender": gender,
    "Married": married,
    "Dependents": dependents,
    "Education": education,
    "Self_Employed": self_employed,
    "ApplicantIncome": applicant_income.astype(int),
    "CoapplicantIncome": coapplicant_income.astype(int),
    "LoanAmount": loan_amount,
    "Loan_Amount_Term": loan_term,
    "Credit_History": credit_history,
    "Property_Area": property_area,
    "Loan_Status": loan_status,
})

df.to_csv("data/loan_data.csv", index=False)
print(f"Synthetic dataset written to data/loan_data.csv  |  shape={df.shape}")
print(df["Loan_Status"].value_counts(normalize=True))
