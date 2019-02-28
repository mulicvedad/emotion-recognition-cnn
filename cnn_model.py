import keras
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D, Activation
from keras.models import model_from_json
from keras.layers import Dropout
from data_manipulation import ImageDataManipulation
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard
from keras.utils import plot_model


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
                         kernel_size=(3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.25))#
        model.add(Conv2D(filters=64, kernel_size=(3, 3)))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(units=5, activation='softmax'))
        model.compile(loss=keras.losses.mean_squared_error, optimizer=keras.optimizers.Adam(),
                      metrics=['accuracy', keras.metrics.categorical_accuracy])
        print(model.summary())
        self.model = model

    def fit(self, image_data, image_labels, validation_split, batch_size=64, epochs=30,
            random_state=33, json_filename='model_cnn2.json', weights_filename='model_cnn2.h5'):

        train_data, validation_data, train_labels, validation_labels = train_test_split(image_data,
                                                                                        image_labels,
                                                                                        test_size=validation_split,
                                                                                        random_state=random_state)
        train_labels = to_categorical(train_labels)
        validation_labels = to_categorical(validation_labels)
        trained_model = self.model.fit(train_data, train_labels, batch_size=batch_size, epochs=epochs,
                       verbose=1, validation_data=(validation_data, validation_labels), callbacks=[TensorBoard(histogram_freq=1)])
        self.model_ready = True
        # serialize model to JSON
        model_json = self.model.to_json()
        with open(json_filename, 'w') as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights(weights_filename)
        print('Saved model to disk')
        return trained_model

    # Input is any image so this function includes preprocessing
    def predict(self, imgs, num_decimals=4):
        test_data = np.empty((len(imgs), self.input_shape[0], self.input_shape[1], self.input_shape[2]))
        i = 0
        for img in imgs:
            test_data[i] = ImageDataManipulation.prepare_img(img, self.input_shape[0], self.input_shape[1], grayscale=True)
            i += 1
        # Detect when there is no faces in single input image
        # Useful for GUI feedback
        if len(imgs) == 1 and test_data[0] is None:
            return None
        loaded = True
        if not self.model_ready:
            loaded = self.load_trained_model(self.default_model_path, self.default_model_weights)
        if loaded is False:
            raise Exception('No trained model is available')
        return np.round(self.model.predict(x=test_data), num_decimals)

    # This function takes preprocessed data
    def predict_simple(self, imgs, num_decimals=4):
        return np.round(self.model.predict(x=imgs), num_decimals)

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

    def plot_model(self):
        plot_model(self.model, to_file='model.png')

    # Assumption: there are subfolders called like emotions:
    # 'anger', 'happy', 'sadness', 'surprise', 'neutral'
    def test_model(self, test_img_root_dir):
        data_manip = ImageDataManipulation(src_folder=None, training_imgs_folder=None)
        data, labels = data_manip.load_training_data(root_img_folder=test_img_root_dir)
        return self.predict(imgs=data, num_decimals=6), labels


def show_training_history(model, title="Model accuracy", loss=False, ylabel="Accuracy"):
    if loss:
        plt.plot(model.history['loss'])  # loss
        plt.plot(model.history['val_loss'])  # val_loss
    else:
        plt.plot(model.history['acc'])  # loss
        plt.plot(model.history['val_acc'])  # val_loss
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def train():
    model = ConvolutionalNN()
    data_manip = ImageDataManipulation(None, 'training_data')
    train_data, train_labels = data_manip.load_training_data('training_data')
    dataset = data_manip.preprocess_data(train_data)
    trained_model = model.fit(dataset, train_labels, 0.15,json_filename='model_cnn_.json', weights_filename='model_cnn_.h5', epochs=50)
    show_training_history(trained_model)
    show_training_history(trained_model, title="Model loss", loss=True, ylabel="MSE")

#train()


