from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

# Define the model architecture exactly as it was during training
model = models.mobilenet_v2(weights=None)  # weights=None avoids loading pretrained ImageNet weights

# Important: Match this with your training architecture
model.classifier = nn.Sequential(
    nn.Dropout(0.2),
    nn.Sequential(
        nn.Linear(1280, 7)
    )
)

# Load the trained weights (state_dict)
model.load_state_dict(torch.load('mobilenet.pt', map_location=torch.device('cpu')))
model.eval()

# Define the class labels in the same order used during training
CATEGORIES = ['Acne', 'Eczema', 'Psoriasis', 'Melanoma', 'BCC', 'Rosacea', 'Warts']

# Define image preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((128, 128)),        # Resize to match training input size
    transforms.ToTensor(),                # Convert to Tensor and scale [0,1]
    transforms.Normalize([0.5]*3, [0.5]*3)  # Normalize to [-1, 1] if trained that way
])

@app.route('/')
def home():
    return '✅ Skin Disease Prediction API (PyTorch) is running!'

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400

        file = request.files['image']
        image = Image.open(io.BytesIO(file.read())).convert('RGB')

        # Apply preprocessing
        input_tensor = transform(image).unsqueeze(0)  # Shape: [1, 3, 128, 128]

        # Run inference
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            pred_index = torch.argmax(probabilities).item()
            predicted_label = CATEGORIES[pred_index]
            confidence = probabilities[pred_index].item() * 100

        return jsonify({
            'prediction': predicted_label,
            'confidence': round(confidence, 2)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
