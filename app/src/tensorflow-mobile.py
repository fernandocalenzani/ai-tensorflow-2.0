import datetime

import numpy as np
import tensorflow as tf

# dataset
dataset = tf.keras.datasets.fashion_mnist

# Pre-processing

((x_train, y_train), (x_test, y_test)) = dataset.load_data()

# normalization
x_train = x_train / 255.0
x_test = x_test / 255.0

# reshaping
x_train = x_train.reshape(-1, 28 * 28)
x_test = x_test.reshape(-1, 28 * 28)

# Neural network

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Dense(
    units=128, activation='relu', input_shape=(784,)))
model.add(tf.keras.layers.Dense(
    units=128, activation='relu', input_shape=(784,)))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(
    units=10, activation='softmax'))

# Compiling

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])

model.summary()

model.fit(x_train, y_train, epochs=15)

test_loss, test_accuracy = model.evaluate(x_test, y_test)

print('test_loss: ', test_loss)
print('test_accuracy: ', test_accuracy)

model_dir = "app/models/1.0.0/fashion_mobile_model.h5"

tf.saved_model.save(
    model,
    export_dir=model_dir,
)

converter = tf.lite.TFLiteConverter.from_keras_model(model_dir)

tflite_model = converter.convert()

with open("tflite_model", "wb") as f:
    f.write(tflite_model)

""" Improving the accuracy and loss:
-2 increasing number of epochs
- add hide layers
- optimizer learning rate
- use other activation function
"""

"""
config 1: units=128, activation='relu', epochs=5, hideLayers=1 + dropout
result 1: test_loss:  0.3626554608345032 test_accuracy:  0.8672999739646912

config 2: units=128, activation='relu', epochs=10, hideLayers=1 + dropout
result 2: test_loss:  0.32938992977142334 test_accuracy:  0.8844000101089478

config 2: units=128, activation='relu', epochs=10, hideLayers=2 + dropout
result 2: test_loss:  0.32938992977142334 test_accuracy:  0.8844000101089478

config 2: units=128, activation='relu', epochs=15, hideLayers=2 + dropout
result 2: test_loss:  0.37883785367012024 test_accuracy:  0.8798999786376953
"""

print("END")
