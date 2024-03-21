from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained Keras model
model = load_model('modelCATR.h5')

# Define a function to preprogitcess the image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Rescale to match the training data
    return img_array

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get the image file from the request
    img_file = request.files['image']
    
    # Preprocess the image
    img_array = preprocess_image(img_file)
    
    # Make prediction
    prediction = model.predict(img_array)
    
    # Decode prediction
    if prediction[0] > 0.5:
        predicted_class = 'rabbit'
    else:
        predicted_class = 'cat'
    
    # Return prediction as JSON response
    return jsonify({'prediction': predicted_class})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
