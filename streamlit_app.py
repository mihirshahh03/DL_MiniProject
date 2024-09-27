import streamlit as st
import os
import json
from kaggle.api.kaggle_api_extended import KaggleApi
import subprocess

# Define the path to your Kaggle credentials
# Define the path to your Kaggle credentials
kaggle_json_path = r"C:\Users\ronit\kaggle.json"

# Check if kaggle.json exists
if not os.path.exists(kaggle_json_path):
    st.error(f"Could not find kaggle.json at {kaggle_json_path}")
else:
    st.success(f"kaggle.json found at {kaggle_json_path}")
    
    # Load kaggle.json credentials
    with open(kaggle_json_path, 'r') as f:
        kaggle_creds = json.load(f)
        os.environ['KAGGLE_USERNAME'] = kaggle_creds['username']
        os.environ['KAGGLE_KEY'] = kaggle_creds['key']

    # Initialize Kaggle API
    api = KaggleApi()
    api.authenticate()

    # Dataset management: use local dataset or Kaggle API to download
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
        # download_kaggle_dataset("dataset-name")
        pass

elif selected_model == "Insurance Cost Prediction":
    # No Kaggle dataset required, assuming local dataset is available
    # st.write("Using local dataset for Insurance Cost Prediction.")
    pass

elif selected_model == "Image Segmentation":
    if st.button("Download Fish Segmentation Dataset from Kaggle"):
        download_kaggle_dataset("A Large-Scale Dataset for Fish Segmentation and Classification")

# Running the model notebook
if st.button(f"Run {selected_model} Model"):
    if selected_model == "Text Classification":
        run_notebook("notebooks/manualtry.ipynb")
    elif selected_model == "Medical Pill Classification":
        run_notebook("notebooks/drugsimageclassification.ipynb")
    elif selected_model == "Insurance Cost Prediction":
        run_notebook("notebooks/Medical_cost.ipynb")
    elif selected_model == "Image Segmentation":
        run_notebook(r"C:\Users\ronit\OneDrive\Desktop\College\Sem_7\NNDL\NNDL_Repo\DL_MiniProject\notebooks\imageseg.ipynb")
        
    else:
        st.write("Model 5 is still under development.")

# Display results here after the notebook runs (you can modify this section based on outputs)
st.write("Model results will be displayed here after execution.")
