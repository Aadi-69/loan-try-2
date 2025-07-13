from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import os

app = Flask(__name__)

# --- Loan Approval Prediction ---

# Load and preprocess approval data
approval_data = pd.read_csv('loan_approval_data.csv')
approval_data.fillna(method='ffill', inplace=True)

le = LabelEncoder()
for col in approval_data.select_dtypes(include='object').columns:
    approval_data[col] = le.fit_transform(approval_data[col])

X = approval_data.drop(['Loan_Status'], axis=1)
y = approval_data['Loan_Status']

approval_model = LogisticRegression()
approval_model.fit(X, y)

joblib.dump(approval_model, 'approval_model.joblib')

# --- Loan Amount Prediction Model (Trained with dummy/synthetic data) ---

# Create and train a dummy amount prediction model
dummy_data = pd.DataFrame({
    'income': [4000, 6000, 8000, 3000, 10000],
    'credit_score': [650, 700, 720, 600, 750],
    'years_employed': [2, 5, 7, 1, 10],
    'loan_amount': [100000, 150000, 180000, 90000, 250000]
})

X_amt = dummy_data.drop('loan_amount', axis=1)
y_amt = dummy_data['loan_amount']

amount_model = LinearRegression()
amount_model.fit(X_amt, y_amt)

joblib.dump(amount_model, 'amount_model.joblib')


@app.route('/')
def home():
    return render_template('form.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Get data from form
    income = float(request.form['income'])
    credit_score = float(request.form['credit_score'])
    years_employed = float(request.form['years_employed'])

    # Loan Amount Prediction
    amt_model = joblib.load('amount_model.joblib')
    amt_pred = amt_model.predict([[income, credit_score, years_employed]])[0]

    # Loan Approval Prediction (using dummy features for demo)
    features = pd.DataFrame([{
        'Gender': 1,
        'Married': 1,
        'Dependents': 0,
        'Education': 0,
        'Self_Employed': 0,
        'ApplicantIncome': income,
        'CoapplicantIncome': 0.0,
        'LoanAmount': amt_pred / 1000,  # Scale for example
        'Loan_Amount_Term': 360,
        'Credit_History': 1,
        'Property_Area': 1
    }])

    model = joblib.load('approval_model.joblib')
    approval_pred = model.predict(features)[0]
    approval_result = "Approved ✅" if approval_pred == 1 else "Not Approved ❌"

    return f"""
    <h2>Predicted Loan Amount: ₹{amt_pred:.2f}</h2>
    <h2>Loan Approval Status: {approval_result}</h2>
    <a href='/'>Back</a>
    """


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
