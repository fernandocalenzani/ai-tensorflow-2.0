import os

import numpy as np
import tensorflow as tf
# from matplotlib.pyplot import plt
from tqdm import tqdm_notebook

# config
img_shape = (128, 128, 3)
fine_tunning_at = 100
dataset_path = "app/data/cats_and_dogs_filtered"
train_dir = os.path.join(dataset_path, "train")
validation_dir = os.path.join(dataset_path, "validation")

# dataset
dataGenerator = tf.keras.preprocessing.image.ImageDataGenerator

# build model
base_model = tf.keras.applications.MobileNetV2(
    input_shape=img_shape, include_top=False, weights="imagenet")

# Base model
base_model.trainable = True
print(len(base_model.layers))

# congelando as camadas 0 a fine_tunning_at
for layer in base_model.layers[:fine_tunning_at]:
  layer.trainable = False

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)

predict_layer = tf.keras.layers.Dense(
    units=1, activation='sigmoid')(global_average_layer)

model = tf.keras.models.Model(inputs=base_model.input, outputs=predict_layer)

base_model.summary()


# Compilling

model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.0001), loss='binary_crossentropy',
              metrics=['accuracy'])

# data generators
data_gen_train = dataGenerator(rescale=1/255.)
data_gen_valid = dataGenerator(rescale=1/255.)

train_generator = data_gen_train.flow_from_directory(
    train_dir, target_size=(128, 128), batch_size=128, class_mode='binary')

valid_generator = data_gen_train.flow_from_directory(
    validation_dir, target_size=(128, 128), batch_size=128, class_mode='binary')

model.fit_generator(train_generator, epochs=5, validation_data=valid_generator)

text_loss, test_accuracy = model.evaluate_generator(valid_generator)

print('test_loss: ', text_loss)
print('test_accuracy: ', test_accuracy)

print("END")
