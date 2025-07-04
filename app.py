from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from torchvision import models, transforms
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

# 1. Define or load the same model architecture
model = models.mobilenet_v2(pretrained=False)  # Or your custom model
model.classifier[1] = torch.nn.Linear(model.last_channel, 7)  # Adjust final layer for 7 classes

# 2. Load weights into model
model.load_state_dict(torch.load('mobilenet.pt', map_location=torch.device('cpu')))
model.eval()  # Set to evaluation mode

CATEGORIES = ['Acne', 'Eczema', 'Psoriasis', 'Melanoma', 'BCC', 'Rosacea', 'Warts']

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

@app.route('/')
def home():
    return 'Skin Disease Prediction API (PyTorch with state_dict) is running!'

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400

        image_file = request.files['image']
        image = Image.open(io.BytesIO(image_file.read())).convert('RGB')
        input_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.nn.functional.softmax(outputs[0], dim=0)
            pred_index = torch.argmax(probs).item()
            predicted_label = CATEGORIES[pred_index]
            confidence = probs[pred_index].item() * 100

        return jsonify({
            'prediction': predicted_label,
            'confidence': round(confidence, 2)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
