import keras
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import model_from_json
from data_manipulation import ImageDataManipulation


class ConvolutionalNN:
    def __init__(self, input_shape=(64, 64, 1), model_path='model_cnn.json', model_weights='model_cnn.h5'):
        self.default_model_path = model_path
        self.default_model_weights = model_weights
        self.model = None
        self.model_ready = False
        self.input_shape = input_shape
        self._init_model()

    def _init_model(self):
        model = Sequential()
        model.add(Conv2D(input_shape=self.input_shape, filters=32,
                         kernel_size=(7, 7)))
        model.add(MaxPooling2D())
        model.add(
            Conv2D(filters=64, kernel_size=(7, 7)))
        model.add(MaxPooling2D())
        model.add(Flatten())
        model.add(Dense(units=5, activation="softmax"))
        model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),
                      metrics=['accuracy'])
        self.model = model

    def fit(self, image_data, image_labels, validation_split, batch_size=64, epochs=30,
            random_state=33, json_filename='model_cnn.json', weights_filename='model_cnn.h5'):

        train_data, validation_data, train_labels, validation_labels = train_test_split(image_data,
                                                                                        image_labels,
                                                                                        test_size=validation_split,
                                                                                        random_state=random_state)
        train_labels = to_categorical(train_labels)
        validation_labels = to_categorical(validation_labels)
        self.model.fit(train_data, train_labels, batch_size=batch_size, epochs=epochs,
                       verbose=1, validation_data=(validation_data, validation_labels))
        self.model_ready = True

        # serialize model to JSON
        model_json = self.model.to_json()
        with open(json_filename, "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights(weights_filename)
        print("Saved model to disk")

    def predict(self, img, num_decimals=4):
        x = self.input_shape
        a = x[0]
        b = x[1]
        img = ImageDataManipulation.prepare_img(img, self.input_shape[0], self.input_shape[1], grayscale=True)
        if img is None:
            return None
        loaded = True
        if not self.model_ready:
            loaded = self.load_trained_model(self.default_model_path, self.default_model_weights)
        if loaded is False:
            raise Exception("No trained model is available")
        return np.round(self.model.predict(x=img), num_decimals)

    def load_trained_model(self, json_filename, weights_filename):
        try:
            json_file = open(json_filename, 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)
            loaded_model.load_weights(weights_filename)
            self.model = loaded_model
            self.model_ready = True
            return True
        except Exception as e:
            return False
