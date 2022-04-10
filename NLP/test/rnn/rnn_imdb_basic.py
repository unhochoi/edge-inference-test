# Model: Simple RNN
# Dataset: imdb
# Reference : https://www.kaggle.com/tanyildizderya/imdb-dataset-sentiment-analysis-using-rnn

import tensorflow as tf

# Check GPU Availability
device_name = tf.test.gpu_device_name()
if not device_name:
    print('Cannot found GPU. Training with CPU')
else:
    print('Found GPU at :{}'.format(device_name))

# Get train/test dataset
num_words = 15000
maxlen = 130

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=num_words)
X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = tf.keras.preprocessing.sequence.pad_sequences(X_test, maxlen=maxlen)

# Build Simple RNN model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Embedding(num_words, 32, input_length=len(X_train[0])))
model.add(tf.keras.layers.SimpleRNN(16, input_shape=(num_words, maxlen), return_sequences=False, activation="relu"))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer="rmsprop",
              metrics=['accuracy'])

history = model.fit(X_train,
                    y_train,
                    validation_split=0.2,
                    epochs = 5,
                    batch_size=128,
                    verbose = 1)          

score = model.evaluate(X_test,
                       y_test,
                      batch_size=128)
