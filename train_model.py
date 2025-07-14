import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle
import os

# Step 1: Load dataset
df = pd.read_csv("data/Car_details_v3.csv")

# Step 2: Drop unused or messy columns
df.drop(['name', 'torque', 'max_power', 'engine', 'mileage'], axis=1, inplace=True, errors='ignore')
df.dropna(inplace=True)

# Step 3: Feature Engineering
df['car_age'] = 2020 - df['year']
df.drop('year', axis=1, inplace=True)

# Step 4: Encode categorical features
df = pd.get_dummies(df, columns=['fuel', 'seller_type', 'transmission', 'owner'], drop_first=True)

# Step 5: Define X, y
X = df.drop('selling_price', axis=1)
y = df['selling_price']

# Step 6: Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 7: Save model
os.makedirs("models", exist_ok=True)
with open("models/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved as models/model.pkl")
