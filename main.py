import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pydot
import graphviz
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Input, Flatten, Reshape
from tensorflow.keras.utils import to_categorical


(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.cast(tf.expand_dims(x_train, axis=3), dtype=tf.float32) / 255.0
x_test = tf.cast(tf.expand_dims(x_test, axis=3), dtype=tf.float32) / 255.0

y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

# print(x_train.shape, y_train.shape, y_train_cat.shape, sep='\n')

input_img = Input(shape=(28, 28, 1))
en = Flatten()(input_img)
en = Dense(128, activation='relu')(en)
en = Dense(64, activation='relu')(en)
en = Dense(32, activation='relu')(en)
hidden_output = Dense(8, activation='linear')(en)

input_enc = Input(shape=(8,))
de = Dense(64, activation='relu')(input_enc)
de = Dense(128, activation='relu')(de)
de = Dense(28*28, activation='sigmoid')(de)
decoded = Reshape((28, 28, 1))(de)

input_rec = Input(shape=(8,))
cl = Dense(128, activation='relu')(input_rec)
classificator_output = Dense(10, activation='softmax')(cl)

encoder = keras.Model(input_img, hidden_output, name='encoder')
decoder = keras.Model(input_enc, decoded, name='decoder')
classificator = keras.Model(input_rec, classificator_output, name='classificator')

# model = keras.Model(input_img, [decoder(encoder(input_img)), classificator(encoder(input_img))])
# model.compile(optimizer='adam', loss={
#     'decoder': 'mean_squared_error',
#     'classificator': 'categorical_crossentropy'
# })


class NeuralNetwork(tf.keras.Model):
    def __init__(self, units):
        super().__init__()
        self.units = units
        self.model_layers = [keras.layers.Dense(n, activation='relu') for n in self.units]

    def call(self, inputs):
        x = inputs
        for layer in self.model_layers:
            x = layer(x)
        return x

    def get_config(self):
        return {'units': self.units}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class NeuralNetworkLinear(tf.keras.Model):
    def __init__(self, units):
        super().__init__()
        self.units = units
        self.model_layers = [keras.layers.Dense(n, activation='linear') for n in self.units]

    def call(self, inputs):
        x = inputs
        for layer in self.model_layers:
            x = layer(x)
        return x

    def get_config(self):
        return {'units': self.units}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


myModel = NeuralNetwork([128, 10])

# history = model.fit(x_train, {
#     'decoder': x_train,
#     'classificator': y_train_cat
# }, epochs=10, batch_size=32, validation_split=0.2)
model_loaded = keras.models.load_model('model')
model_loaded2 = keras.models.load_model('model')
weights = model_loaded.get_weights()
model_loaded2.set_weights(weights)
model_loaded.save_weights('model_weights')
# model_loaded = keras.models.load_model('model.h5', custom_objects={"NeuralNetwork": NeuralNetworkLinear})

gen_img, class_image = model_loaded.predict(tf.expand_dims(x_test, axis=3))

# print(history.history)
plt.imshow(np.expand_dims(np.squeeze(x_test[0]), axis=2), cmap='gray')
plt.show()
plt.imshow(np.expand_dims(np.squeeze(gen_img[0]), axis=2), cmap='gray')
plt.show()
print(tf.argmax(class_image, axis=1).numpy()[0])

# model.save('model')
# model.save('model.h5') - old format
