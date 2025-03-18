from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
import logging
import os

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load TFLite model
model_path = os.getenv("MODEL_PATH", "disease_detection_model_quantized.tflite")
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

@app.route('/predict', methods=['POST'])
def predict():
try:
    # Get the image from the request
    data = request.json
    if not data or 'image' not in data:
        return jsonify({'error': 'No image data provided'}), 400
    
    # Log that we received data
    print("Received image data, processing...")
    
    # Extract the base64 image
    image_data = data['image']
    if 'base64' not in image_data:
        return jsonify({'error': 'Invalid image format'}), 400
    
    # Remove the data URL prefix (e.g., data:image/jpeg;base64,)
    base64_str = image_data.split('base64,')[1]
    
    # Decode the image
    img_bytes = base64.b64decode(base64_str)
    img = Image.open(BytesIO(img_bytes))
    
    # Preprocess the image for the model
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make prediction
    print("Running model prediction...")
    predictions = model.predict(img_array)
    
    # Process results
    results = []
    for i, prob in enumerate(predictions[0]):
        results.append({
            'className': class_names[i],
            'probability': float(prob)
        })
    
    # Sort by probability
    results.sort(key=lambda x: x['probability'], reverse=True)
    
    print("Prediction complete, returning results")
    return jsonify({'predictions': results})

except Exception as e:
    print(f"Error during prediction: {str(e)}")
    import traceback
    traceback.print_exc()
    return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)