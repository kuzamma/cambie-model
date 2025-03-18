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
        logger.info("Received prediction request")
        image_data = request.json.get('image')
        if not image_data:
            return jsonify({"error": "No image data provided"}), 400

        # Decode base64 image
        image_bytes = base64.b64decode(image_data.split(',')[1])
        image = Image.open(io.BytesIO(image_bytes)).resize((224, 224))
        image_array = np.array(image, dtype=np.float32) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        # Run inference
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        interpreter.set_tensor(input_details[0]['index'], image_array)
        interpreter.invoke()

        output_data = interpreter.get_tensor(output_details[0]['index'])
        results = output_data[0].tolist()

        # Map to class names (adjust based on your model)
        class_names = ["test1", "rest2"]
        predictions = [{"className": class_names[i], "probability": float(results[i])} 
                       for i in range(len(class_names))]
        predictions.sort(key=lambda x: x["probability"], reverse=True)

        return jsonify({"predictions": predictions})

    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)