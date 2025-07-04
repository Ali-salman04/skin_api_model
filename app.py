from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from flask_cors import CORS
import traceback

app = Flask(__name__)
CORS(app)

model = load_model('mobilenet_skin_disease_model.h5')
CATEGORIES = ['Acne', 'Melanoma', 'Nevus', 'AK', 'BCC']  # 👈 update this to match your new model

@app.route('/')
def home():
    return 'Skin Disease Prediction API is running!'

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['image']

        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({'error': 'Invalid image file'}), 400

        img = cv2.resize(img, (128, 128))
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)

        prediction = model.predict(img)
        print("Prediction Output:", prediction)

        pred_index = np.argmax(prediction)
        print("Predicted Index:", pred_index)

        if pred_index >= len(CATEGORIES):
            return jsonify({'error': f'Prediction index {pred_index} is out of bounds for class list of size {len(CATEGORIES)}'}), 500

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
