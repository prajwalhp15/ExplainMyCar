import streamlit as st
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt

# Load the trained model
with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="Car Price Predictor", layout="centered")
st.title("🚗 Car Price Prediction with Explainable AI")
st.markdown("Enter your car details to get an estimated selling price and understand what features influenced the prediction.")

# 🔧 User Inputs
car_age = st.slider("Car Age (in years)", 0, 20, 5)
km_driven = st.number_input("Kilometers Driven", 0, 300000, step=1000, value=30000)
seats = st.slider("Number of Seats", 2, 10, 5)

fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel", "LPG"])
seller_type = st.selectbox("Seller Type", ["Dealer", "Individual", "Trustmark Dealer"])
transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
owner = st.selectbox("Owner", [
    "First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner", "Test Drive Car"
])

# Expected columns from training
expected_columns = [
    'km_driven', 'car_age', 'seats',
    'fuel_Diesel', 'fuel_Petrol', 'fuel_LPG',
    'seller_type_Individual', 'seller_type_Trustmark Dealer',
    'transmission_Manual',
    'owner_Second Owner', 'owner_Third Owner', 'owner_Fourth & Above Owner', 'owner_Test Drive Car'
]

# Prepare input DataFrame with correct structure
input_data = pd.DataFrame(columns=expected_columns)
input_data.loc[0] = 0

# Fill in user-selected values
input_data.at[0, 'km_driven'] = km_driven
input_data.at[0, 'car_age'] = car_age
input_data.at[0, 'seats'] = seats

if fuel == "Diesel":
    input_data.at[0, 'fuel_Diesel'] = 1
elif fuel == "Petrol":
    input_data.at[0, 'fuel_Petrol'] = 1
elif fuel == "LPG":
    input_data.at[0, 'fuel_LPG'] = 1

if seller_type == "Individual":
    input_data.at[0, 'seller_type_Individual'] = 1
elif seller_type == "Trustmark Dealer":
    input_data.at[0, 'seller_type_Trustmark Dealer'] = 1

if transmission == "Manual":
    input_data.at[0, 'transmission_Manual'] = 1

if owner == "Second Owner":
    input_data.at[0, 'owner_Second Owner'] = 1
elif owner == "Third Owner":
    input_data.at[0, 'owner_Third Owner'] = 1
elif owner == "Fourth & Above Owner":
    input_data.at[0, 'owner_Fourth & Above Owner'] = 1
elif owner == "Test Drive Car":
    input_data.at[0, 'owner_Test Drive Car'] = 1

# 🚀 Predict and Explain
if st.button("🔍 Predict Price"):
    try:
        # Match feature order
        model_features = model.feature_names_in_
        input_data = input_data[model_features]

        # Predict
        prediction = model.predict(input_data)[0]
        st.success(f"💰 Estimated Selling Price: ₹{prediction:,.2f}")

        # SHAP Explanation
        st.subheader("🔎 Why this prediction?")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_data)

        # ✅ Future-proofed: use figure + show=False
        fig, ax = plt.subplots()
        shap.plots._waterfall.waterfall_legacy(
            explainer.expected_value[0], shap_values[0], input_data.iloc[0], show=False
        )
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Something went wrong: {e}")
