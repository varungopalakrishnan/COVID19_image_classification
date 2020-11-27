from flask import Flask
from flask import request, jsonify, send_from_directory
from flask_cors import CORS, cross_origin
import base64
import numpy as np
import io
from PIL import Image
import keras
from keras.models import Sequential, load_model
from keras.preprocessing.image import ImageDataGenerator, img_to_array


app = Flask(__name__)
cors = CORS(app)


def get_model():
    global model
    model = load_model(
        "ML_model/Xception_model_Full.h5")
    print(' * model loaded')


def preprocess_image(image, ts):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(ts)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    # image = keras.preprocessing.image.load_img(image, target_size=ts)
    # image = keras.preprocessing.image.img_to_array(image, data_format='base64')
    return image


print("* loading keras model")

get_model()


@app.route('/predict', methods=['POST', 'GET'])  # get not required
@cross_origin()
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
    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)
