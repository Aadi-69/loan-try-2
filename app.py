from flask import Flask, render_template, request
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

app = Flask(__name__)

# Load and preprocess approval data
def train_approval_model():
    data = pd.read_csv("loan_approval_data.csv")
    data.fillna(method='ffill', inplace=True)

    le = LabelEncoder()
    for col in data.select_dtypes(include='object').columns:
        data[col] = le.fit_transform(data[col])

    X = data.drop('Loan_Status', axis=1)
    y = data['Loan_Status']

    model = LogisticRegression()
    model.fit(X, y)

    joblib.dump(model, 'approval_model.joblib')

# Train dummy loan amount model
def train_amount_model():
    dummy_data = pd.DataFrame({
        'income': [4000, 6000, 8000, 3000, 10000],
        'credit_score': [650, 700, 720, 600, 750],
        'years_employed': [2, 5, 7, 1, 10],
        'loan_amount': [100000, 150000, 180000, 90000, 250000]
    })

    X = dummy_data.drop('loan_amount', axis=1)
    y = dummy_data['loan_amount']

    model = LinearRegression()
    model.fit(X, y)

    joblib.dump(model, 'amount_model.joblib')


# Train models if not already trained
if not os.path.exists("approval_model.joblib"):
    train_approval_model()

if not os.path.exists("amount_model.joblib"):
    train_amount_model()


@app.route('/')
def index():
    return render_template('form.html')


@app.route('/predict', methods=['POST'])
def predict():
    income = float(request.form['income'])
    credit_score = float(request.form['credit_score'])
    years_employed = float(request.form['years_employed'])

    # Predict loan amount
    amount_model = joblib.load("amount_model.joblib")
    predicted_amount = amount_model.predict([[income, credit_score, years_employed]])[0]

    # Prepare input for approval prediction
    approval_model = joblib.load("approval_model.joblib")
    features = pd.DataFrame([{
        'Gender': 1,
        'Married': 1,
        'Dependents': 0,
        'Education': 0,
        'Self_Employed': 0,
        'ApplicantIncome': income,
        'CoapplicantIncome': 0,
        'LoanAmount': predicted_amount / 1000,
        'Loan_Amount_Term': 360,
        'Credit_History': 1,
        'Property_Area': 1
    }])

    approval_prediction = approval_model.predict(features)[0]
    result = "Approved ✅" if approval_prediction == 1 else "Not Approved ❌"

    return f"""
    <h2>Predicted Loan Amount: ₹{predicted_amount:.2f}</h2>
    <h2>Loan Approval Status: {result}</h2>
    <a href='/'>Back</a>
    """


# DO NOT run app.run() here — gunicorn will use "app" as entry point
# This file is ready for `gunicorn app:app`
