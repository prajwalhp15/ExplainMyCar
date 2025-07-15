# ExplainMyCar – Car Price Prediction with Explainable AI

**ExplainMyCar** is a Streamlit-based web application that predicts the selling price of a used car using a trained machine learning model, and provides visual explanations of the prediction using SHAP (SHapley Additive Explanations).

This project demonstrates how explainability can help users understand how different car features contribute to pricing decisions, improving transparency and trust in AI-powered tools.

---

## Features

- Predict the selling price of a car based on user inputs
- SHAP-based waterfall visualization to show feature influence
- Trained Random Forest Regression model
- Interactive user interface built with Streamlit

---

## Project Structure

explainmycar/
├── app.py # Streamlit web app
├── train_model.py # Model training script
├── models/
│ └── model.pkl # Saved trained model
├── data/
│ └── Car_details_v3.csv # Training dataset
├── shap_explanation.png # Sample output visual (optional)
├── requirements.txt # Python dependencies
└── README.md # Project overview and usage


---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/prajwalhp15/ExplainMyCar.git
cd ExplainMyCar

python -m venv venv
venv\Scripts\activate  # On Windows

pip install -r requirements.txt

streamlit run app.py

Model
The model is trained using RandomForestRegressor on the "Car_details_v3.csv" dataset. The dataset includes features such as:

Car age

Kilometers driven

Fuel type

Seller type

Transmission type

Number of seats

Ownership status

The model is saved as model.pkl and used in app.py for predictions.

Explainability
SHAP (SHapley Additive Explanations) is used to explain individual predictions by showing how each feature affects the output. The app visualizes this using a SHAP waterfall plot, helping users understand which factors influenced the car's estimated price.

Example Output
You can include a screenshot here (e.g. shap_explanation.png) showing:

The estimated price

SHAP explanation chart

License
This project is open source and available for educational and personal use. Attribution is appreciated.

Credits
Developed by Prajwal H P
ML model and dashboard built using scikit-learn, SHAP, and Streamlit.
