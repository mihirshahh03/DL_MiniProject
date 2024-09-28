# image_segmentation.py
import streamlit as st
import cv2
from PIL import Image
import numpy as np
import tensorflow as tf

# Function to load the TensorFlow model
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model(r'C:\Users\ronit\OneDrive\Desktop\College\Sem_7\NNDL\NNDL_Repo\DL_MiniProject\best_model.h5')
        return model
    except OSError as e:
        st.error(f"Error loading model: {e}")
        return None

# Load the model once at the start of the app
model = load_model()

# Page title
st.title("Fish Image Segmentation")

# Create a section for uploading an image
st.header("Upload an Image")
uploaded_image = st.file_uploader("Select an image", type=["jpg", "jpeg", "png"])

# Create a section for displaying the output
st.header("Segmentation Output")

# Display a placeholder for the output
output_placeholder = st.container()

# Function to preprocess the uploaded image
def preprocess_image(image):
    try:
        # Convert image to numpy array
        image = np.array(image)

        # Check if the image is valid and has 3 color channels (RGB)
        if image is None or image.size == 0:
            st.error("Invalid image. Please upload a valid image.")
            return None
        if len(image.shape) != 3 or image.shape[2] != 3:
            st.error("Uploaded image must be in RGB format with 3 channels.")
            return None

        # Resize image to the model's expected input size (128x128 assumed)
        image_resized = cv2.resize(image, (128, 128))

        # Normalize the image
        image_normalized = image_resized / 255.0

        # Add batch dimension
        return np.expand_dims(image_normalized, axis=0)
    except Exception as e:
        st.error(f"Error during image preprocessing: {e}")
        return None

# Function to create the mask from the predicted output
def create_mask(pred_mask):
    try:
        # Convert model output to a mask (argmax over the last axis)
        pred_mask = tf.argmax(pred_mask, axis=-1)
        pred_mask = pred_mask[..., tf.newaxis]  # Add a channel dimension
        return pred_mask[0]  # Remove batch dimension
    except Exception as e:
        st.error(f"Error creating mask from prediction: {e}")
        return None

if uploaded_image is not None:
    try:
        # Read the uploaded image
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess the image for model input
        processed_image = preprocess_image(image)

        if processed_image is not None and model is not None:
            # Predict the segmentation mask
            pred_mask = model.predict(processed_image)

            # Add debugging information for `pred_mask`
            st.write(f"Prediction mask shape: {pred_mask.shape}")
            st.write(f"Prediction mask dtype: {pred_mask.dtype}")

            # Create the segmentation mask
            segmented_mask = create_mask(pred_mask)

            if segmented_mask is not None:
                # Debugging information for `segmented_mask`
                st.write(f"Segmented mask shape: {segmented_mask.shape}")
                st.write(f"Segmented mask dtype: {segmented_mask.dtype}")

                # Convert mask to uint8 for OpenCV compatibility
                segmented_mask = segmented_mask.numpy().astype(np.uint8) * 255  # Scale mask to 0-255

                # Post-process the mask (resize back to original image size)
                segmented_mask_resized = cv2.resize(segmented_mask, (image.width, image.height))

                # Display the segmented mask
                st.image(segmented_mask_resized, caption="Segmented Output", use_column_width=True)
            else:
                st.error("Failed to create segmentation mask.")
        else:
            st.error("Image preprocessing failed or model not loaded.")
    except Exception as e:
        st.error(f"An error occurred during segmentation: {e}")
else:
    st.info("Please upload an image for segmentation.")
