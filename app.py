from flask import Flask, request, render_template, url_for
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
import uuid

app = Flask(__name__)

# Upload folder setup
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Make sure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load the trained model
model_path = os.path.join(os.getcwd(), "currency_prediction_model.h5")
print("🔍 Looking for model at:", model_path)

try:
    model = tf.keras.models.load_model(model_path)
    print("✅ Model loaded successfully!")
except Exception as e:
    print("❌ Model failed to load.")
    print("Error:", e)
    model = None  # Make sure model is defined even if loading fails

# Define class names
class_names = ['fake', 'real']

# Home route
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template('index.html', prediction="Error: Model not loaded.", img_path=None)

    if 'file' not in request.files:
        return render_template('index.html', prediction="Error: No file uploaded.", img_path=None)

    file = request.files['file']
    
    if file.filename == '':
        return render_template('index.html', prediction="Error: No file selected.", img_path=None)

    if file:
        try:
            # Secure and unique filename
            filename = secure_filename(file.filename)
            filename = str(uuid.uuid4()) + "_" + filename

            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            print("✅ Image saved at:", filepath)

            # Image preprocessing
            img = image.load_img(filepath, target_size=(128, 128))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0

            # Prediction
            predictions = model.predict(img_array)
            predicted_index = np.argmax(predictions[0])
            predicted_class = class_names[predicted_index]
            confidence = predictions[0][predicted_index]

            result_text = f"Result: {predicted_class.capitalize()} Note (Confidence: {confidence * 100:.2f}%)"
            img_url = url_for('static', filename=f'uploads/{filename}')

            return render_template('index.html', prediction=result_text, img_path=img_url)

        except Exception as e:
            return render_template('index.html', prediction=f"Prediction Error: {e}", img_path=None)

# Run Flask app
if __name__ == "__main__":
    app.run(debug=True)
