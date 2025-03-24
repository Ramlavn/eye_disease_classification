import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import os

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
import tensorflow as tf

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress low-level logs
warnings.filterwarnings("ignore")  # Suppress Python warnings
tf.get_logger().setLevel('ERROR')  # Suppress TensorFlow logs

# Set Streamlit page title
st.title("EyeZenX - Ocular Disease Recognition")

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_Model.h5", compile=False)

# Read class labels
with open('labels.txt', 'r') as file:
    class_names = [line.strip() for line in file.readlines()]

# Streamlit file uploader
uploaded_file = st.file_uploader("Upload an eye image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Open the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    
    # Display the uploaded image
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Preprocess the image

    #  The model expects images of size 224x224 pixels, so we define this as the target size.
    size = (224, 224)

    #  Resizes & crops the uploaded image to exactly 224x224 pixels using the LANCZOS filter
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # Converts the PIL Image to a NumPy array for numerical processing.    
    image_array = np.asarray(image)

    # Scale the pixel values or normalize the values from [0,255] to  [-1,1]
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    
    # Prepare the image for model input

    # Creates a NumPy array with shape (1, 224, 224, 3) to match the input format expected by the model
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array
    
    # Predict using the model
    prediction = model.predict(data)
    
    # Get model prediction
    index = int(np.argmax(prediction))  # Ensure it's an integer

    # Get the class name from the index

    # class_name = class_names[index] if index < len(class_names) else "Unknown"
    # confidence_score = prediction[0][index] if index < len(prediction[0]) else 0.0

    if index < len(class_names):
        class_name = class_names[index]
    else:
        class_name = "Unknown"
    
    # Get the confidence score from the index
    if index < len(prediction[0]):
        confidence_score = prediction[0][index]
    else:
        confidence_score = 0.0
    
    # Display prediction result
    st.write(f"**Prediction:** {class_name}")
    st.write(f"**Confidence Score:** {confidence_score:.2f}")

    # Plot confidence scores using seaborn
    fig, ax = plt.subplots()
    prediction = prediction.flatten()
    
    sns.barplot(x=prediction, y=class_names, ax=ax)
    ax.set_xlabel("Confidence Score (%)")
    ax.set_title("Model Confidence Scores for Each Class")
    plt.gca().invert_yaxis()
    
    st.pyplot(fig)

    # Sidebar validation section
    st.sidebar.header("Validate Prediction")
    st.sidebar.write("If the prediction is incorrect, select the correct class.")

    # Ensure the index is within the valid range
    if index >= 0 and index < len(class_names):  
        # If the index is valid, use it as the default  
        default_index = index  
    else:  
        # If the index is out of bounds, default to 0  
        default_index = 0  

    # default_index = index if 0 <= index < len(class_names) else 0
    correct_label = st.sidebar.selectbox("Select the correct class", class_names, index=default_index)


    if st.sidebar.button("Validate & Save"):
        # Create base dir directory to store the validated images
        base_dir = "validated_images"
        class_folder = os.path.join(base_dir, correct_label)
        os.makedirs(class_folder, exist_ok=True)  # Create if not exists

        # Save the image inside the correct class folder
        image_path = os.path.join(class_folder, uploaded_file.name)
        image.save(image_path)

        st.sidebar.success(f"✅ Image saved in `validated_images/{correct_label}/` for future training.")
