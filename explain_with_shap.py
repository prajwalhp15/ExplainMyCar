import pandas as pd
import shap
import pickle
import matplotlib.pyplot as plt

# Load dataset and model
df = pd.read_csv("data/Car_details_v3.csv")
with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

# Prepare data (same as training)
df.drop(['name', 'torque', 'max_power', 'engine', 'mileage'], axis=1, inplace=True, errors='ignore')
df.dropna(inplace=True)
df['car_age'] = 2020 - df['year']
df.drop('year', axis=1, inplace=True)
df = pd.get_dummies(df, columns=['fuel', 'seller_type', 'transmission', 'owner'], drop_first=True)

X = df.drop('selling_price', axis=1)

# Use a sample input (first row)
sample = X.iloc[[0]]

# SHAP explain
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(sample)

# Plot SHAP explanation
shap.initjs()
shap.summary_plot(shap_values, sample, plot_type="bar")
plt.savefig("shap_explanation.png")
print("✅ SHAP explanation generated and saved as shap_explanation.png")
