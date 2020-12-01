from flask import request, jsonify, send_from_directory, render_template, Flask
import base64
import numpy as np
import io
import os
from PIL import Image
import keras
from keras.models import Sequential, load_model
from keras.preprocessing.image import ImageDataGenerator, img_to_array

RES_FOLDER = os.path.join('static', 'resources')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = RES_FOLDER


def get_model():
    global model
    model = load_model(
        "ML_model/Xception_model_Full.h5")
    print(' * model loaded')


print("* loading keras model")
get_model()


def preprocess_image(image, ts):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(ts)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    # image = keras.preprocessing.image.load_img(image, target_size=ts)
    # image = keras.preprocessing.image.img_to_array(image, data_format='base64')
    return image


@app.route('/')
def home():
    full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'nCT24.jpg')
    return render_template("index.html", user_image=full_filename)


@app.route('/predict', methods=['POST', 'GET'])  # get not required
def predict():
    message = request.get_json(force=True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))
    processed_image = preprocess_image(image, ts=(200, 200))

    prediction = model.predict(processed_image).tolist()
    score = prediction[0]
    print("score:", score)

    response = {
        'prediction': {
            'Negative': 100 * (1 - score[0]),
            'Positive': 100 * score[0]
        }
    }
    return jsonify(response)  # , render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
