import numpy as np
import tensorflow as tf

maxlen = 100
n_words = 20000

# dataset
dataset = tf.keras.datasets.imdb

# Pre-processing

((x_train, y_train), (x_test, y_test)) = dataset.load_data(num_words=n_words)

# normalization
x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)

# CNN

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Embedding(
    input_dim=n_words, output_dim=128, input_shape=(x_train[1],)))


# LSTM
model.add(tf.keras.layers.LSTM(
    units=128, activation='tanh'))

# Output layer
model.add(tf.keras.layers.Dense(
    units=1, activation='sigmoid'))

model.add(tf.keras.layers.Dense(
    units=10, activation='softmax'))

# Compilling

model.compile(optimizer='rmsprop', loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

model.fit(x_train, y_train, epochs=3, batch_size=128)

test_loss, test_accuracy = model.evaluate(x_test, y_test)

print('test_loss: ', test_loss)
print('test_accuracy: ', test_accuracy)


print("END")
