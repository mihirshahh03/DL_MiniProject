import streamlit as st
import pickle
import numpy as np

# Load the necessary files (only the ones that work)
with open(r'C:\Users\ASUS\Desktop\College\Sem7\DL\M1_DL_Project\DL_MiniProject\notebooks\polynomial_features.pkl', 'rb') as pf_file:
    poly_features = pickle.load(pf_file)

with open(r'C:\Users\ASUS\Desktop\College\Sem7\DL\M1_DL_Project\DL_MiniProject\notebooks\scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)
    print(type(scaler))  # Should print something like <class 'sklearn.preprocessing._data.StandardScaler'>


# Streamlit App
st.title("Medical Insurance Cost Prediction")

# Input fields for the user
age = st.number_input("Age", min_value=18, max_value=100, value=25)

# Extract only the numeric part of the tuple (1 for Male, 0 for Female)
sex = st.selectbox("Sex", options=[("Male", 1), ("Female", 0)])[1]

bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)

# Extract only the numeric part of the tuple (1 for Yes, 0 for No)
smoker = st.selectbox("Smoker", options=[("Yes", 1), ("No", 0)])[1]

# Extract only the numeric part of the tuple (0 for Northeast, 1 for Northwest, etc.)
region = st.selectbox("Region", options=[("Northeast", 0), ("Northwest", 1), ("Southeast", 2), ("Southwest", 3)])[1]

# Placeholder message in case models are not available
st.warning("Note: The prediction models are unavailable due to file issues.")

# Predict button
if st.button("Predict Charges"):
    # Convert inputs to array
    user_input = np.array([[age, sex, bmi, children, smoker, region]])

    # Scale the input data
    user_input_scaled = scaler.transform(np.array(user_input))


    # Apply polynomial features
    user_input_poly = poly_features.transform(user_input_scaled)

    # Placeholder for prediction since no working model is available
    st.error("Model for prediction is currently unavailable. Please load a valid model.")
