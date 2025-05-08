import pickle
import numpy as np
from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE  # Make sure imbalanced-learn is installed
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load the saved model
with open("customer_churn_model.pkl", "rb") as f:
    model_data = pickle.load(f)

loaded_model = model_data["model"]
feature_names = model_data["features_names"]

# Load the saved encoders
with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

# Function to handle the form submission and predict churn
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from form
    input_data = {
        'gender': request.form['gender'],
        'SeniorCitizen': int(request.form['SeniorCitizen']),
        'Partner': request.form['Partner'],
        'Dependents': request.form['Dependents'],
        'tenure': int(request.form['tenure']),
        'PhoneService': request.form['PhoneService'],
        'MultipleLines': request.form['MultipleLines'],
        'InternetService': request.form['InternetService'],
        'OnlineSecurity': request.form['OnlineSecurity'],
        'OnlineBackup': request.form['OnlineBackup'],
        'DeviceProtection': request.form['DeviceProtection'],
        'TechSupport': request.form['TechSupport'],
        'StreamingTV': request.form['StreamingTV'],
        'StreamingMovies': request.form['StreamingMovies'],
        'Contract': request.form['Contract'],
        'PaperlessBilling': request.form['PaperlessBilling'],
        'PaymentMethod': request.form['PaymentMethod'],
        'MonthlyCharges': float(request.form['MonthlyCharges']),
        'TotalCharges': float(request.form['TotalCharges'])
    }

    # Convert input data into DataFrame
    input_data_df = pd.DataFrame([input_data])

    # Encode categorical features using saved encoders
    for column, encoder in encoders.items():
        input_data_df[column] = encoder.transform(input_data_df[column])

    # Make a prediction
    prediction = loaded_model.predict(input_data_df)
    pred_prob = loaded_model.predict_proba(input_data_df)

    # Convert prediction to a human-readable result
    result = "Churn" if prediction[0] == 1 else "No Churn"
    prob = pred_prob[0][1] * 100  # Churn probability in percentage

    return render_template('index.html', prediction=result, prob=prob)

if __name__ == '__main__':
    app.run(debug=True)
