import os

import numpy as np
import requests
import tensorflow as tf
from flask import Flask, jsoninfy, request
from scipy.misc import imread, imsave

# dataset
dataset = tf.keras.datasets.fashion_mnist

# Pre-processing
class_names = ['airplane', 'automobile', 'bird',
               'cat', 'dog', 'frog', 'horse', 'ship', 'truck']

((x_train, y_train), (x_test, y_test)) = dataset.load_data()

# normalization
x_train = x_train / 255.0
x_test = x_test / 255.0

for i in range(5):
    imsave(name="uploads/{}.png".format(i), arr=x_test[i])

with open("cnn_obj_model.json", "r") as f:
    model_json = f.read()

model = tf.keras.models.model_from_json(model_json)
model.load_weights("cnn_obj_model.h5")
model.summary()

app = Flask(__name__)


@app.route('/<string:img_name>', methods=['POST'])
def classify_image(img_name):
    upload_dir = 'uploads/'
    image = imread(upload_dir + img_name)

    predictions = model.predict([image])

    return jsoninfy({
        "object": class_names[np.argmax(predictions[0])]
    })


app.run(
    host="127.0.0.1",
    port=3000,
    debug=False
)

print("END")
