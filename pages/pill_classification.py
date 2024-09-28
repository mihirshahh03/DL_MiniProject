# import streamlit as st
# import numpy as np
# import joblib
# from PIL import Image
# import tensorflow as tf

# # Load the saved model (replace with your model path)
# model = tf.keras.models.load_model(r'C:\Users\ronit\OneDrive\Desktop\College\Sem_7\NNDL\NNDL_Repo\DL_MiniProject\Malaria Cells.h5')

# # Class labels (replace with your actual class labels)
# class_labels = ['Alaxan', 'Bactidol', 'Bioflu', 'Biogesic', 'DayZinc', 
#                 'Decolgen', 'Fish Oil', 'Kremil S', 'Medicol', 'Neozep']

# # Create a Streamlit app for prediction
# st.title("Pill Image Classification")

# # Input section for uploading an image
# uploaded_image = st.file_uploader("Upload an image of the pill", type=["jpg", "jpeg", "png"])

# if uploaded_image is not None:
#     # Display the uploaded image
#     image = Image.open(uploaded_image)
#     st.image(image, caption="Uploaded Image", use_column_width=True)

#     # Preprocess the image
#     img = image.resize((224, 224))  # Adjust size as per your model's requirements
#     img_array = tf.keras.preprocessing.image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

#     # Make predictions
#     predictions = model.predict(img_array)
#     score = tf.nn.softmax(predictions[0])

#     # Get predicted class
#     predicted_class = class_labels[np.argmax(score)]
#     confidence = np.max(score)

#     # Display results
#     st.write(f"Predicted Class: {predicted_class}")
#     st.write(f"Confidence: {confidence:.2f}")

# # Optional: Add more features, like additional information about the pill or links to resources.


import streamlit as st
import numpy as np
import joblib
from PIL import Image
import tensorflow as tf

# Load the saved model (replace with your model path)
model = tf.keras.models.load_model(r'C:\Users\ronit\OneDrive\Desktop\College\Sem_7\NNDL\NNDL_Repo\DL_MiniProject\Malaria Cells.h5')

# Class labels (replace with your actual class labels)
class_labels = ['Alaxan', 'Bactidol', 'Bioflu', 'Biogesic', 'DayZinc', 
                'Decolgen', 'Fish Oil', 'Kremil S', 'Medicol', 'Neozep']

# Create a Streamlit app for prediction
st.title("Pill Image Classification")

# Input section for uploading an image
uploaded_image = st.file_uploader("Upload an image of the pill", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Display the uploaded image
    image = Image.open(uploaded_image)

    # Convert the image to RGB (in case it has an alpha channel)
    if image.mode != 'RGB':
        image = image.convert('RGB')
        
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    img = image.resize((224, 224))  # Adjust size as per your model's requirements
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make predictions
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    # Get predicted class
    predicted_class = class_labels[np.argmax(score)]
    confidence = np.max(score)

    # Display results
    st.write(f"Predicted Class: {predicted_class}")
    # st.write(f"Confidence: {confidence:.2f}")

# Optional: Add more features, like additional information about the pill or links to resources.
