# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
df = pd.read_csv(r"telco_churn.csv.csv")

# Drop customerID and missing values
df.drop(['customerID'], axis=1, inplace=True)
df.dropna(inplace=True)

# Encode categorical features
for col in df.select_dtypes(include='object'):
    if col != 'Churn':
        df[col] = LabelEncoder().fit_transform(df[col])

# Encode target
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Split features and target
X = df.drop('Churn', axis=1)
y = df['Churn']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'churn_model.pkl')
print("✅ Model saved as churn_model.pkl")
# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

st.title("📊 Telco Customer Churn Prediction App")
st.markdown("This app predicts whether a customer is likely to **churn** based on their details.")

# Load model
model = joblib.load("churn_model.pkl")

# Input form
def user_input():
    gender = st.selectbox("Gender", ['Male', 'Female'])
    SeniorCitizen = st.selectbox("Senior Citizen", ['No', 'Yes'])
    Partner = st.selectbox("Has Partner", ['Yes', 'No'])
    Dependents = st.selectbox("Has Dependents", ['Yes', 'No'])
    tenure = st.slider("Tenure (Months)", 0, 72, 12)
    PhoneService = st.selectbox("Phone Service", ['Yes', 'No'])
    MultipleLines = st.selectbox("Multiple Lines", ['No', 'Yes', 'No phone service'])
    InternetService = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
    OnlineSecurity = st.selectbox("Online Security", ['Yes', 'No', 'No internet service'])
    OnlineBackup = st.selectbox("Online Backup", ['Yes', 'No', 'No internet service'])
    DeviceProtection = st.selectbox("Device Protection", ['Yes', 'No', 'No internet service'])
    TechSupport = st.selectbox("Tech Support", ['Yes', 'No', 'No internet service'])
    StreamingTV = st.selectbox("Streaming TV", ['Yes', 'No', 'No internet service'])
    StreamingMovies = st.selectbox("Streaming Movies", ['Yes', 'No', 'No internet service'])
    Contract = st.selectbox("Contract", ['Month-to-month', 'One year', 'Two year'])
    PaperlessBilling = st.selectbox("Paperless Billing", ['Yes', 'No'])
    PaymentMethod = st.selectbox("Payment Method", [
        'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'
    ])
    MonthlyCharges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
    TotalCharges = st.number_input("Total Charges", 0.0, 10000.0, 2000.0)

    # Create dataframe
    data = {
        'gender': gender,
        'SeniorCitizen': 1 if SeniorCitizen == 'Yes' else 0,
        'Partner': Partner,
        'Dependents': Dependents,
        'tenure': tenure,
        'PhoneService': PhoneService,
        'MultipleLines': MultipleLines,
        'InternetService': InternetService,
        'OnlineSecurity': OnlineSecurity,
        'OnlineBackup': OnlineBackup,
        'DeviceProtection': DeviceProtection,
        'TechSupport': TechSupport,
        'StreamingTV': StreamingTV,
        'StreamingMovies': StreamingMovies,
        'Contract': Contract,
        'PaperlessBilling': PaperlessBilling,
        'PaymentMethod': PaymentMethod,
        'MonthlyCharges': MonthlyCharges,
        'TotalCharges': TotalCharges
    }
    return pd.DataFrame([data])

# Take input
input_df = user_input()

# Encode features like in training
encoded_df = input_df.copy()
for col in encoded_df.select_dtypes(include='object'):
    encoded_df[col] = LabelEncoder().fit_transform(encoded_df[col])

# Predict
if st.button("Predict"):
    prediction = model.predict(encoded_df)
    prob = model.predict_proba(encoded_df)[0][1]
    if prediction[0] == 1:
        st.error(f"⚠️ The customer is likely to churn. (Probability: {prob:.2f})")
    else:
        st.success(f"✅ The customer is not likely to churn. (Probability: {1 - prob:.2f})")

