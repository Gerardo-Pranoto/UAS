import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model
MODEL_PATH = 'best_churn_model.pkl'

try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    st.error(f"Model file '{MODEL_PATH}' not found. Please run the notebook to generate it.")
    st.stop()

def main():
    st.title("Telco Customer Churn Prediction")
    st.write("Enter customer details to predict if they will churn.")

    # Create form for user input
    with st.form("prediction_form"):
        st.header("Customer Demographics")
        col1, col2 = st.columns(2)
        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        with col2:
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])

        st.header("Service Details")
        col3, col4 = st.columns(2)
        with col3:
            tenure = st.number_input("Tenure (Months)", min_value=0, max_value=100, value=1)
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        with col4:
            online_security = st.selectbox("Online Security", ["No internet service", "No", "Yes"])
            online_backup = st.selectbox("Online Backup", ["No internet service", "No", "Yes"])
            device_protection = st.selectbox("Device Protection", ["No internet service", "No", "Yes"])
            tech_support = st.selectbox("Tech Support", ["No internet service", "No", "Yes"])

        st.header("Streaming & Contract")
        col5, col6 = st.columns(2)
        with col5:
            streaming_tv = st.selectbox("Streaming TV", ["No internet service", "No", "Yes"])
            streaming_movies = st.selectbox("Streaming Movies", ["No internet service", "No", "Yes"])
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        with col6:
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment_method = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
            ])
            monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=0.0)
            total_charges = st.number_input("Total Charges", min_value=0.0, value=0.0)

        submit_button = st.form_submit_button("Predict Churn")

    if submit_button:
        # Create DataFrame from input
        data = {
            'gender': [gender],
            'SeniorCitizen': [senior_citizen],
            'Partner': [partner],
            'Dependents': [dependents],
            'tenure': [tenure],
            'PhoneService': [phone_service],
            'MultipleLines': [multiple_lines],
            'InternetService': [internet_service],
            'OnlineSecurity': [online_security],
            'OnlineBackup': [online_backup],
            'DeviceProtection': [device_protection],
            'TechSupport': [tech_support],
            'StreamingTV': [streaming_tv],
            'StreamingMovies': [streaming_movies],
            'Contract': [contract],
            'PaperlessBilling': [paperless_billing],
            'PaymentMethod': [payment_method],
            'MonthlyCharges': [monthly_charges],
            'TotalCharges': [total_charges]
        }
        
        input_df = pd.DataFrame(data)
        
        # Display input data
        st.subheader("Input Data")
        st.dataframe(input_df)

        # Predict
        try:
            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0][1] if hasattr(model, 'predict_proba') else None

            st.subheader("Prediction Result")
            if prediction == 'Yes':
                st.error(f"⚠️ Churn Prediction: **YES**")
            else:
                st.success(f"✅ Churn Prediction: **NO**")
            
            if probability is not None:
                st.write(f"Churn Probability: **{probability:.2%}**")
                
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

if __name__ == "__main__":
    main()