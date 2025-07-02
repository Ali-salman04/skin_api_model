# 📍 app/app.py

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import pickle
import os

app = Flask(__name__)
CORS(app)

# ✅ Load the scikit-learn model from .pkl
model_path = os.path.join(os.path.dirname(__file__), 'EN0_model.pkl')

with open(model_path, 'rb') as f:
    model = pickle.load(f)

# ✅ Define your class labels
CATEGORIES = ['Acne', 'Eczema', 'Psoriasis', 'Melanoma', 'BCC', 'Rosacea', 'Warts']

@app.route('/')
def home():
    return '✅ Skin Disease Prediction API is running (pkl model)!'

@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("✅ Image received")
        file = request.files['image']
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        print("✅ Image decoded")

        # Preprocess the image: resize, normalize, flatten
        img = cv2.resize(img, (128, 128))
        img = img.astype('float32') / 255.0
        img = img.flatten().reshape(1, -1)
        print("✅ Image preprocessed")

        # Predict
        prediction = model.predict_proba(img)[0]  # Get confidence per class
        pred_index = np.argmax(prediction)
        predicted_label = CATEGORIES[pred_index]
        confidence = float(prediction[pred_index]) * 100

        print("✅ Prediction successful")
        return jsonify({
            'prediction': predicted_label,
            'confidence': round(confidence, 2)
        })

    except Exception as e:
        print("❌ ERROR:", e)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
