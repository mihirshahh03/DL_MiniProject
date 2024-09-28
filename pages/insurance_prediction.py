import streamlit as st
import joblib
import numpy as np

# Load the saved RandomForest model
model = joblib.load(r'C:\Users\ronit\OneDrive\Desktop\College\Sem_7\NNDL\NNDL_Repo\DL_MiniProject\random_forest_model.pkl')

# Create a Streamlit app for prediction
st.title("Insurance Charges Prediction")

# Input features
age = st.number_input("Age", min_value=18, max_value=100, step=1)
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, step=0.1)
children = st.number_input("Number of children", min_value=0, max_value=10, step=1)

# Map sex to numerical values
sex_map = {
    "Female": 0,
    "Male": 1
}
sex_input = st.selectbox("Sex", list(sex_map.keys()))
sex = sex_map[sex_input]  # Get the corresponding value

# Map smoker status to numerical values
smoker_map = {
    "No": 0,
    "Yes": 1
}
smoker_input = st.selectbox("Smoker", list(smoker_map.keys()))
smoker = smoker_map[smoker_input]  # Get the corresponding value

# Map region names to corresponding values
region_map = {
    "northeast": 0,
    "northwest": 1,
    "southeast": 2,
    "southwest": 3
}
region_input = st.selectbox("Region", list(region_map.keys()))
region = region_map[region_input]  # Get the corresponding value

# Create a feature array based on user inputs
features = np.array([[age, bmi, children, sex, smoker, region]])

# Make prediction
if st.button("Predict Charges"):
    prediction = model.predict(features)
    st.write(f"Predicted Insurance Charges: ${prediction[0]:.2f}")
