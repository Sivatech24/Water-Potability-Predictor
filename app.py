import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
data = pd.read_csv("water_quality.csv")  # Ensure you replace this with the actual file path

# Handling missing values
data = data.dropna()

# Splitting features and target variable
X = data.drop(columns=["Potability"])
y = data["Potability"]

# Splitting into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardizing the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Training logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

def predict_potability(features):
    features = np.array(features).reshape(1, -1)
    features = scaler.transform(features)
    return model.predict(features)[0]

# Streamlit UI
st.title("Water Potability Prediction")
st.write("Enter the water quality parameters to predict potability.")

ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=7.0)
hardness = st.number_input("Hardness", min_value=0.0, value=150.0)
solids = st.number_input("Solids", min_value=0.0, value=20000.0)
chloramines = st.number_input("Chloramines", min_value=0.0, value=4.0)
sulfate = st.number_input("Sulfate", min_value=0.0, value=300.0)
conductivity = st.number_input("Conductivity", min_value=0.0, value=400.0)
organic_carbon = st.number_input("Organic Carbon", min_value=0.0, value=10.0)
trihalomethanes = st.number_input("Trihalomethanes", min_value=0.0, value=50.0)
turbidity = st.number_input("Turbidity", min_value=0.0, value=4.0)

if st.button("Predict Potability"):
    input_features = [ph, hardness, solids, chloramines, sulfate, conductivity, organic_carbon, trihalomethanes, turbidity]
    result = predict_potability(input_features)
    st.write("The water is potable" if result == 1 else "The water is not potable")
