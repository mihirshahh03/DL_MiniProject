import streamlit as st
import os
from kaggle.api.kaggle_api_extended import KaggleApi
import subprocess

# Define the path to your Kaggle credentials
os.environ['KAGGLE_CONFIG_DIR'] = r"C:\Users\ASUS\kaggle.json"

# Initialize Kaggle API
api = KaggleApi()
api.authenticate()

# Download dataset from Kaggle if required
def download_kaggle_dataset(dataset_name):
    # Example Kaggle dataset download command
    with st.spinner(f"Downloading dataset: {dataset_name} from Kaggle..."):
        subprocess.run(f"kaggle datasets download -d {dataset_name} -p datasets/", shell=True)

# Define function to run Jupyter Notebooks
def run_notebook(notebook_path):
    subprocess.run(f"jupyter nbconvert --to notebook --execute {notebook_path}", shell=True)

# App title
st.title("Machine Learning Models Dashboard")

# Sidebar for model selection
st.sidebar.title("Choose a Model")
models = ["Text Classification", "Medical Pill Classification", "Insurance Cost Prediction", 
          "Image Segmentation", "Model 5 (WIP)"]
selected_model = st.sidebar.selectbox("Select a model to run:", models)

# Description based on model selection
model_descriptions = {
    "Text Classification": "This model classifies different languages based on text.",
    "Medical Pill Classification": "EfficientNetB3 model used to classify medical pill images.",
    "Insurance Cost Prediction": "Linear regression model for predicting medical insurance costs.",
    "Image Segmentation": "UNet model for performing image segmentation on fish images.",
    "Model 5 (WIP)": "Currently under development."
}
st.write(model_descriptions[selected_model])

# Dataset management: use local dataset or Kaggle API to download
if selected_model == "Medical Pill Classification":
    # Use Kaggle dataset for pill classification model
    if st.button("Download Pill Classification Dataset from Kaggle"):
        download_kaggle_dataset("dataset-name")

elif selected_model == "Insurance Cost Prediction":
    # No Kaggle dataset required, assuming local dataset is available
    st.write("Using local dataset for Insurance Cost Prediction.")

# Running the model notebook
if st.button(f"Run {selected_model} Model"):
    if selected_model == "Text Classification":
        run_notebook("notebooks/manualtry.ipynb")
    elif selected_model == "Medical Pill Classification":
        run_notebook("notebooks/drugsimageclassification.ipynb")
    elif selected_model == "Insurance Cost Prediction":
        run_notebook("notebooks/Medical_cost.ipynb")
    elif selected_model == "Image Segmentation":
        run_notebook("notebooks/imageseg.ipynb")
    else:
        st.write("Model 5 is still under development.")

# Display results here after the notebook runs (you can modify this section based on outputs)
st.write("Model results will be displayed here after execution.")
