from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64

app = Flask(__name__)

# Load your TFLite model
interpreter = tf.lite.Interpreter(model_path="disease_detection_model_quantized.tflite")
interpreter.allocate_tensors()

@app.route('/predict', methods=['POST'])
def predict():
    # Get image from request
    image_data = request.json['image']
    image_bytes = base64.b64decode(image_data.split(',')[1])
    
    # Process image
    image = Image.open(io.BytesIO(image_bytes)).resize((224, 224))
    image_array = np.array(image, dtype=np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    
    # Run inference
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    interpreter.set_tensor(input_details[0]['index'], image_array)
    interpreter.invoke()
    
    # Get results
    output_data = interpreter.get_tensor(output_details[0]['index'])
    results = output_data[0].tolist()
    
    # Map to class names (adjust based on your model)
    class_names = ["test1", "rest2"]
    
    predictions = [{"className": class_names[i], "probability": float(results[i])} 
                  for i in range(len(class_names))]
    
    # Sort by probability
    predictions.sort(key=lambda x: x["probability"], reverse=True)
    
    return jsonify({"predictions": predictions})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)