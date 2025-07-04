from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torchvision.transforms as transforms
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

# Load PyTorch model
model = torch.load('mobilenet.pt', map_location=torch.device('cpu'))
model.eval()  # Set to evaluation mode

CATEGORIES = ['Acne', 'Eczema', 'Psoriasis', 'Melanoma', 'BCC', 'Rosacea', 'Warts']

# Define preprocessing
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),  # Converts to [0,1] and CHW format
    transforms.Normalize([0.5]*3, [0.5]*3)  # Adjust based on your training setup
])

@app.route('/')
def home():
    return 'Skin Disease Prediction API (PyTorch) is running!'

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400

        image_file = request.files['image']
        image_bytes = image_file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

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
