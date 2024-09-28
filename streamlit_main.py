import streamlit as st

# App title
st.title("Machine Learning Models Dashboard")

# Display buttons for different models
if st.button("Text Classification"):
    st.experimental_set_query_params(page="text_classification")

if st.button("Medical Pill Classification"):
    st.experimental_set_query_params(page="pill_classification")

if st.button("Insurance Cost Prediction"):
    st.experimental_set_query_params(page="insurance_prediction")

if st.button("Image Segmentation"):
    st.experimental_set_query_params(page="image_segmentation")

if st.button("Netflix Title Generation"):
    st.experimental_set_query_params(page="netflix_title_generation")
