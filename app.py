
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
import base64
from PIL import Image
import io
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="disease_detection_model_quantized.tflite")
interpreter.allocate_tensors()

# Define class names based on your model
class_names = ["test", "test2"]

@app.route('/test', methods=['GET'])
def test():
return "API is working!"

@app.route('/predict', methods=['POST'])
def predict():
try:
    logger.info("Received prediction request")
    image_data = request.json.get('image')  # Fixed variable name
    if not image_data:
        logger.error("No image data provided")
        return jsonify({"error": "No image data provided"}), 400

    # Decode base64 image
    try:
        image_bytes = base64.b64decode(image_data.split(',')[1])
        logger.info(f"Successfully decoded base64 image, size: {len(image_bytes)} bytes")
    except Exception as e:
        logger.error(f"Error decoding base64 image: {str(e)}")
        return jsonify({"error": f"Invalid image format: {str(e)}"}), 400

    # Process image
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB').resize((224, 224))
        image_array = np.array(image, dtype=np.float32) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        logger.info(f"Image processed, shape: {image_array.shape}")
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return jsonify({"error": f"Error processing image: {str(e)}"}), 400

    # Run inference
    try:
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        logger.info(f"Input details: {input_details}")
        logger.info(f"Output details: {output_details}")
        
        interpreter.set_tensor(input_details[0]['index'], image_array)
        interpreter.invoke()
        
        output_data = interpreter.get_tensor(output_details[0]['index'])
        results = output_data[0].tolist()
        logger.info(f"Inference complete, raw results: {results}")
    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")
        return jsonify({"error": f"Error during inference: {str(e)}"}), 500

    # Map to class names
    predictions = [{"className": class_names[i], "probability": float(results[i])} 
                  for i in range(min(len(class_names), len(results)))]
    predictions.sort(key=lambda x: x["probability"], reverse=True)
    
    logger.info(f"Returning predictions: {predictions}")
    return jsonify({"predictions": predictions})

except Exception as e:
    logger.error(f"Unexpected error during prediction: {str(e)}")
    import traceback
    logger.error(traceback.format_exc())
    return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
logger.info("Starting Flask server")
app.run(host='0.0.0.0', port=5000, debug=True)