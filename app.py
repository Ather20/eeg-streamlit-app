import gdown
import os

MODEL_ID = "1ADhTn3DtLvL-dskVXEJWK-iYgppMdd4s"
MODEL_URL = f"https://drive.google.com/uc?id={MODEL_ID}"
MODEL_PATH = "model.h5"

# Download the model if not exists
if not os.path.exists(MODEL_PATH):
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import sys
import logging


logging.basicConfig(filename="app.log", level=logging.DEBUG)

# Helper function to get the correct path of resources (for standalone executables)
def resource_path(relative_path):
    """ Get the absolute path to resource files when running as a standalone executable. """
    try:
        base_path = sys._MEIPASS  # For PyInstaller and Nuitka standalone mode
    except Exception:
        base_path = os.path.abspath(".")
    
    return os.path.join(base_path, relative_path)

# Load the model and logo using resource_path
model_path = resource_path("model.h5")
logo_path = resource_path("logo.png")

# Load the model
try:
    model = tf.keras.models.load_model("model.h5")
except Exception as e:
    st.error(f"Error loading model: {e}")
    logging.debug(f"Error loading model: {e}")

# Define the class labels with more readable text
class_labels = ['Epilepsy Detected', 'No Epilepsy Detected']

# Function to preprocess the image and make it compatible with the model
def preprocess_image(image):
    img_height = 224
    img_width = 224

    try:
        # Ensure the image is in RGB format
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize the image to match model input shape
        image = image.resize((img_width, img_height))
        
        # Convert the image to a numpy array and normalize it
        image_array = np.array(image) / 255.0
        
        # Add a batch dimension
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array
    except Exception as e:
        st.error(f"Error in preprocessing image: {e}")
        return None

# Function to predict the class using the pre-trained model
def predict_eeg(image):
    preprocessed_image = preprocess_image(image)
    if preprocessed_image is None:
        return None
    try:
        # Get the predictions from the model
        logits = model.predict(preprocessed_image)
        
        # Since it's binary classification, use sigmoid for binary classification
        predicted_class_index = int(np.round(logits[0][0]))  # Assuming logits are raw predictions
        predicted_class = class_labels[predicted_class_index]
        
        return predicted_class
    except Exception as e:
        st.error(f"Error in making prediction: {e}")
        return None

# Load and display the logo at the top
try:
    logo = Image.open(logo_path)  # Path to the uploaded logo using resource_path
    st.image(logo, use_column_width=True)  # Display the logo
except Exception as e:
    st.error(f"Error loading logo: {e}")
    logging.debug(f"Error loading logo: {e}")

# Streamlit app layout
st.title("Epilepsy Detection from EEG Scalograms")
st.write("Upload an EEG scalogram image for epilepsy prediction.")

# File uploader to allow the user to upload an image
uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])

# Add some custom CSS to style the Predict button and center it
st.markdown(
    """
    <style>
    div.stButton {
        display: flex;
        justify-content: center;
        margin-top: 20px;
    }
    div.stButton > button {
        background-color: #007bff;
        color: white;
        padding: 15px 20px;
        border: none;
        border-radius: 5px;
        font-size: 16px;
        cursor: pointer;
        width: 200px;
        text-align: center;
    }

    div.stButton > button:hover {
        background-color: #0056b3;
    }
    .result-box {
        margin: 20px auto; /* This centers the box horizontally */
        padding: 20px;
        border-radius: 5px;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
        text-align: center;
        font-size: 20px;
        font-weight: bold;
        width: 30%;  /* Control the width of the result box */
        height: auto; /* Set to auto to adjust height based on content */
        background-color: #f1f1f1; /* Neutral background color */
    }
    .result-title {
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .result-text {
        font-size: 24px;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Store the prediction result in the session state so it persists after the button is clicked
if 'prediction' not in st.session_state:
    st.session_state.prediction = None

# Display the image if uploaded
if uploaded_file is not None:
    try:
        # Open and display the image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Scalogram', use_column_width=True)
    except Exception as e:
        st.error(f"Error loading image: {e}")

    # Center the Predict button using st.markdown and CSS for Flexbox centering
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Detect"):
            try:
                # Get the prediction and store it in the session state
                prediction = predict_eeg(image)
                st.session_state.prediction = prediction
            except Exception as e:
                st.error(f"Error in prediction process: {e}")

    # Define the text color based on the prediction
    if st.session_state.prediction is not None:
        if st.session_state.prediction == "Epilepsy Detected":
            text_color = "#8B0000"  # Dark red for epilepsy detected
        else:
            text_color = "#4CAF50"  # Green for no epilepsy detected

        # Display the result in a visually appealing, styled result box if a prediction has been made
        st.markdown(f"""
        <div class="result-box">
            <p class="result-title">Result:</p>
            <p class="result-text" style="color:{text_color};">{st.session_state.prediction}</p>
        </div>
        """, unsafe_allow_html=True)
