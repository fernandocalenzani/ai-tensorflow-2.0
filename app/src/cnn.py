import numpy as np
import tensorflow as tf
#from matplotlib.pyplot import plt

# dataset
dataset = tf.keras.datasets.cifar10

# Pre-processing
class_names = ['airplane', 'automobile', 'bird',
               'cat', 'dog', 'frog', 'horse', 'ship', 'truck']

((x_train, y_train), (x_test, y_test)) = dataset.load_data()

# normalization
x_train = x_train / 255.0
x_test = x_test / 255.0

# CNN

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Conv2D(
    filters=32, kernel_size=3, padding='same', activation='relu', input_shape=[32, 32, 3]))

model.add(tf.keras.layers.Conv2D(
    filters=32, kernel_size=3, padding='same', activation='relu'))

model.add(tf.keras.layers.MaxPool2D(
    pool_size=2, strides=2, padding='valid'))

model.add(tf.keras.layers.Conv2D(
    filters=64, kernel_size=3, padding='same', activation='relu'))

model.add(tf.keras.layers.Conv2D(
    filters=64, kernel_size=3, padding='same', activation='relu'))

model.add(tf.keras.layers.MaxPool2D(
    pool_size=2, strides=2, padding='valid'))

# Flattening
model.add(tf.keras.layers.Flatten())

# DENSE NN - Fully connected layers

model.add(tf.keras.layers.Dense(
    units=128,activation='relu'))

model.add(tf.keras.layers.Dense(
    units=10, activation='softmax'))

# Compilling

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])

model.summary()

model.fit(x_train, y_train, epochs=15)

test_loss, test_accuracy = model.evaluate(x_test, y_test)

print('test_loss: ', test_loss)
print('test_accuracy: ', test_accuracy)


print("END")
