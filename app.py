from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from flask_cors import CORS
import traceback

app = Flask(__name__)
CORS(app)

# Load the trained model
model = load_model('mobilenet_skin_disease_model.h5')

# Categories used during training
CATEGORIES = [
    'ACNEREF',
    'Actinic KeratosisREF',
    'Basal Cell CarcinomaREF',
    'DermatographiaREF',
    'Melanocytic_NevusREF',
    'MelanomaREF',
    'NevusREF',
    'Pigmented_Benign_KeratosisREF'
]

@app.route('/')
def home():
    return '✅ Skin Disease Prediction API is running!'

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files.get('image')
        if not file:
            return jsonify({'error': 'No image uploaded'}), 400

        # Decode image
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({'error': 'Invalid image format'}), 400

        # ⚠️ Resize to 96x96 instead of 128x128
        img = cv2.resize(img, (96, 96))
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)

        # Predict
        prediction = model.predict(img)
        pred_index = np.argmax(prediction)

        if pred_index >= len(CATEGORIES):
            return jsonify({'error': 'Prediction index out of range'}), 500

        predicted_label = CATEGORIES[pred_index]
        confidence = float(prediction[0][pred_index]) * 100

        return jsonify({
            'prediction': predicted_label,
            'confidence': round(confidence, 2)
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
