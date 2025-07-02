from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import pickle
import os

app = Flask(__name__)
CORS(app)

# ✅ Load the .pkl model safely
model_path = os.path.join(os.path.dirname(__file__), 'EN0_model.pkl')
with open(model_path, 'rb') as f:
    loaded = pickle.load(f)
    if isinstance(loaded, dict) and 'model' in loaded:
        model = loaded['model']
    else:
        model = loaded

CATEGORIES = ['Acne', 'Eczema', 'Psoriasis', 'Melanoma', 'BCC', 'Rosacea', 'Warts']

@app.route('/')
def home():
    return '✅ Skin Disease Prediction API is running with .pkl model!'

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['image']
        if not file:
            return jsonify({'error': 'No file uploaded'}), 400

        print("✅ Image received")

        # Decode image
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        print("✅ Image decoded")

        # Preprocess
        img = cv2.resize(img, (128, 128))
        img = img.astype('float32') / 255.0
        img = img.flatten().reshape(1, -1)
        print("✅ Image preprocessed")

        # Predict
        if hasattr(model, 'predict_proba'):
            prediction = model.predict_proba(img)[0]
            pred_index = np.argmax(prediction)
            predicted_label = CATEGORIES[pred_index]
            confidence = float(prediction[pred_index]) * 100

            return jsonify({
                'prediction': predicted_label,
                'confidence': round(confidence, 2)
            })
        else:
            return jsonify({'error': 'Loaded object is not a classifier with predict_proba()'}), 500

    except Exception as e:
        print("❌ ERROR:", str(e))
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
