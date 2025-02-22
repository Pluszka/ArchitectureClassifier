import base64
import os

from keras.src.utils import img_to_array
from tensorflow.keras.models import load_model
from flask import Flask, render_template, request
import numpy as np
from PIL import Image

app = Flask(__name__)
model = load_model('static/model/architectureStylesClassifier.keras')
IMG_SIZE = (256, 256)
STYLES = [
    "Art deco",
    "Baroque",
    "Constructivism",
    "Gothic",
    "Minimalism",
    "Modernism",
    "Neoclassicism",
    "Postmodernism",
    "Renaissance"
]


@app.route("/")
def index():
    return render_template("index.html", file_data=None, mime_type=None, style=None)


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'filename' not in request.files:
        return 'No file part'

    file = request.files['filename']

    if file.filename == '':
        return 'No selected file'

    if file:
        image = Image.open(file).convert("RGB")
        image = image.resize(IMG_SIZE)
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)

        image = image / 255.0

        prediction = model.predict(image)
        class_index = np.argmax(prediction, axis=1)[0]
        style = STYLES[class_index]

        file.seek(0)
        file_data = file.read()
        encoded_file = base64.b64encode(file_data).decode("utf-8")
        mime_type = file.content_type

        return render_template("index.html", file_data=encoded_file, mime_type=mime_type, style=style)



if __name__ == "__main__":
    app.run(debug=True)