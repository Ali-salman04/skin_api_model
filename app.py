from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS to allow frontend to connect

# Load trained model
model = load_model('disease_classification_model.h5')

# Make sure this exactly matches your model's class order
CATEGORIES = ['Acne', 'Eczema', 'Psoriasis', 'Melanoma', 'BCC', 'Rosacea', 'Warts']

@app.route('/')
def home():
    return 'Skin Disease Prediction API is running!'

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['image']

        # Read image and decode it to OpenCV format
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

        # Resize image to 128x128 (based on your training code)
        img = cv2.resize(img, (128, 128))

        # Normalize pixel values
        img = img.astype('float32') / 255.0

        # Expand dimensions for batch size
        img = np.expand_dims(img, axis=0)

        # Make prediction
        prediction = model.predict(img)
        pred_index = np.argmax(prediction)
        predicted_label = CATEGORIES[pred_index]
        confidence = float(prediction[0][pred_index]) * 100

        return jsonify({
            'prediction': predicted_label,
            'confidence': round(confidence, 2)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), "Upload correct img"

if __name__ == '__main__':
    app.run(debug=True)
