from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

# Load the trained model
model = tf.keras.models.load_model('mnist_cnn.h5')


def prepare_image(image, target_size):
    if image.mode != "L":
        image = image.convert("L")
    image = image.resize(target_size)
    image = np.array(image)
    image = image.reshape(1, 28, 28, 1)
    image = image.astype("float32")
    image /= 255.0
    return image


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file:
        image = Image.open(io.BytesIO(file.read()))
        prepared_image = prepare_image(image, target_size=(28, 28))

        predictions = model.predict(prepared_image)
        predicted_digit = np.argmax(predictions, axis=1)[0]

        return jsonify({'digit': int(predicted_digit)})


if __name__ == '__main__':
    app.run(debug=True)
