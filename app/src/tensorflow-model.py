import os

import tensorflow as tf

"""
INSTALL TENSORFLOW SERVER
-------------------------
!echo "deb [arch=amd64] http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | sudo tee /etc/apt/sources.list.d/tensorflow-serving.list && \
curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | sudo apt-key add -


!wget 'http://storage.googleapis.com/tensorflow-serving-apt/pool/tensorflow-model-server-universal-2.8.0/t/tensorflow-model-server-universal/tensorflow-model-server-universal_2.8.0_all.deb'
!dpkg -i tensorflow-model-server-universal_2.8.0_all.deb
"""
model_dir = "app/models/"
version = "1.0.0"

epochs = 10

# dataset
dataset = tf.keras.datasets.cifar10

# Pre-processing

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
    units=128, activation='relu'))
model.add(tf.keras.layers.Dense(
    units=10, activation='softmax'))

# Compilling
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])
model.summary()
model.fit(x_train, y_train, epochs=epochs)

test_loss, test_accuracy = model.evaluate(x_test, y_test)

print('test_loss: ', test_loss)
print('test_accuracy: ', test_accuracy)

# export

export_path = os.path.join(model_dir, version)

""" if( os.path.isdir(export_path)):
  !rm -r {export_path}
"""

""" tf.saved_model.simple_save(
    tf.keras.backend.get_session(),
    export_dir=export_path,
    inputs={"input_image": model.input},
    outputs={t.name: t for t in model.outputs}
) """

tf.saved_model.save(
    model,
    export_dir=export_path,
)
