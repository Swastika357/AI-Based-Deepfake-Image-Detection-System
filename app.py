from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
CORS(app)

# Load your trained model
MODEL_PATH = "C:/Users/Jitendra Singh/Desktop/FlaskProject/deepfake_model.h5"

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print("Model loading failed:", str(e))
    model = None  # Safe fallback

@app.route('/')
def serve_frontend():
    return send_from_directory('.', 'Hi.html')  # Assumes IN.html is in the same folder as app.py

# ✅ Your prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    if model is None:
        return jsonify({"error": "Model not loaded properly"}), 500

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save temporarily to disk
    filepath = os.path.join("temp", file.filename)
    os.makedirs("temp", exist_ok=True)
    file.save(filepath)

    try:
        # Preprocess image
        img = image.load_img(filepath, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Predict
        pred = model.predict(img_array)[0][0]  # Binary classification

        prediction = "Fake" if pred > 0.5 else "Real"
        confidence = round(pred * 100, 2) if prediction == "Fake" else round((1 - pred) * 100, 2)

        os.remove(filepath)

        return jsonify({"prediction": prediction, "confidence": f"{confidence}%"})

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

# ✅ Start Flask server
if __name__ == "__main__":
    app.run(debug=True)
