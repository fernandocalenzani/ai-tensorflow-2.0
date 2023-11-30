# assincrona
# sincrona

# espelhada - dividir em gpus
# multi-workers mirrored strategy

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
distributed = tf.distribute.MirroredStrategy()

with distributed.scope():
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Dense(
        units=128, activation='relu', input_shape=(784,)))
    model.add(tf.keras.layers.Dense(
        units=128, activation='relu', input_shape=(784,)))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(
        units=10, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                  metrics=['sparse_categorical_accuracy'])

# Compiling
model.summary()
model.fit(x_train, y_train, epochs=15)

test_loss, test_accuracy = model.evaluate(x_test, y_test)
print('test_loss: ', test_loss)
print('test_accuracy: ', test_accuracy)
print("END")
