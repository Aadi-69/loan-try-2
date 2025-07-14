from flask import Flask, render_template, request
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import LabelEncoder
import os

app = Flask(__name__)

# --- Train Loan Approval Model ---
def train_approval_model():
    df = pd.read_csv("loan_approval_data.csv")

    # Encode object columns
    label_encoders = {}
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = df[col].astype(str)
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Save encoders
    joblib.dump(label_encoders, "label_encoders.joblib")

    df = df.drop(columns=["Loan_ID"], errors="ignore")
    X = df.drop("Loan_Status", axis=1)
    y = df["Loan_Status"]

    # Use SimpleImputer + LogisticRegression in a pipeline
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('model', LogisticRegression(max_iter=1000))
    ])

    pipeline.fit(X, y)
    joblib.dump(pipeline, "approval_model.joblib")


# --- Train Dummy Loan Amount Model ---
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

    joblib.dump(model, "amount_model.joblib")


# --- Train Models if Not Exist ---
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
    amt_model = joblib.load("amount_model.joblib")
    predicted_amount = amt_model.predict([[income, credit_score, years_employed]])[0]

    # Predict approval status
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

    result = approval_model.predict(features)[0]
    message = "Approved ✅" if result == 1 else "Not Approved ❌"

    return f"""
    <h2>Predicted Loan Amount: ₹{predicted_amount:.2f}</h2>
    <h2>Loan Approval Status: {message}</h2>
    <a href='/'>Back</a>
    """

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)


