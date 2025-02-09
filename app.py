from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import tensorflow as tf
import os
import streamlit as st
import threading

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('D:/Desktop/Flask_app/saved_model/cats_vs_dogs_model.h5')

# Define preprocessing function
def preprocess_image(img_path, target_size=(128, 128)):
    # Open the image and convert to RGB (to handle grayscale or other formats)
    img = Image.open(img_path).convert('RGB')
    # Resize the image
    img = img.resize(target_size)
    # Convert the image to an array and normalize it
    img_array = img_to_array(img) / 255.0
    # Add a batch dimension
    return np.expand_dims(img_array, axis=0)

@app.route('/frontend')
def serve_frontend():
    return app.send_static_file('index.html')


@app.route('/')
def home():
    return "Welcome to the Cat vs Dog Classifier API!"


# Define the route for image upload and prediction
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return "Please use POST with an image file for prediction."

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    # Check if the file has a valid filename
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Save the uploaded file to a temporary location
        file_path = os.path.join('uploads', 'temp_image')
        file.save(file_path)

        # Convert to .jpg format if needed
        jpg_path = os.path.join('uploads', 'temp.jpg')
        img = Image.open(file_path).convert('RGB')
        img.save(jpg_path, 'JPEG')

        # Preprocess the image
        img_preprocessed = preprocess_image(jpg_path)

        # Perform prediction
        prediction = model.predict(img_preprocessed)

        # Classify prediction: output 0 is cat, 1 is dog
        label = 'dog' if prediction[0][0] > 0.5 else 'cat'

        # Delete the temporary files after prediction
        os.remove(file_path)
        os.remove(jpg_path)

        # Return the prediction result
        return jsonify({'prediction': label})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/test_predict', methods=['GET'])
def test_predict():
    import requests

    # Localhost URL for your /predict route
    url = "http://127.0.0.1:5000/predict"
    test_image_path = "D:/Desktop/Flask_app/cat.jpg"  # Replace with your test image path

    # Open and send the image file as a POST request
    with open(test_image_path, 'rb') as f:
        response = requests.post(url, files={'file': f})

    # Return the response for testing
    return response.json()


# Start the Flask app in a thread
def run_flask():
    app.run(debug=True, host='0.0.0.0', port=5000)


# Streamlit App Code
def run_streamlit():
    st.title("Cat vs Dog Classifier 🐱🐶")
    st.write("Upload an image to classify as cat or dog.")
    
    # Maximum file size (5 MB = 5 * 1024 * 1024 bytes)
    MAX_FILE_SIZE = 5 * 1024 * 1024
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        # Check the size of the uploaded file
        if uploaded_file.size > MAX_FILE_SIZE:
            st.error("The file is too large. Please upload a file smaller than 5MB.")
        else:
            # Display uploaded image with a specified width
            st.image(uploaded_file, caption="Uploaded Image", width=400)  # You can change the width to your preference

            st.write("")
            st.write("Classifying...")

            # Save the uploaded file
            with open("uploads/temp_image", "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Preprocess the image and make a prediction
            img_preprocessed = preprocess_image("uploads/temp_image")
            prediction = model.predict(img_preprocessed)

            # Classify prediction
            label = 'dog' if prediction[0][0] > 0.5 else 'cat'

            # Display prediction result
            st.write(f"The uploaded image is a {label}!")


# Run Flask in a separate thread
flask_thread = threading.Thread(target=run_flask)
flask_thread.start()

# Run Streamlit app
run_streamlit()
