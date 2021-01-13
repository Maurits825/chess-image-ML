from utils import image_loader
from utils import visualize_data
import glob
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense

from keras.layers.advanced_activations import LeakyReLU

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import argparse
import random
import pickle
import cv2
import os

from sklearn.preprocessing import MultiLabelBinarizer


# ref: https://www.pyimagesearch.com/2018/05/07/multi-label-classification-with-keras/
class ChessPieceCNN:
    def __init__(self, image_dir, height, width, depth):
        self.input_shape = (height, width, depth)
        self.file_names = glob.glob(image_dir + "/*.png")

        self.model = None
        self.train_network = None
        self.training_data = None

    def build_neural_network(self, classes, final_act):
        model = Sequential()
        chan_dim = -1 # TODO needed? -- for batch normalization

        # CONV => RELU => POOL
        model.add(Conv2D(32, (3, 3), padding="same", input_shape=self.input_shape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chan_dim))
        model.add(MaxPooling2D(pool_size=(3, 3)))
        model.add(Dropout(0.25))

        # (CONV => RELU) * 2 => POOL
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chan_dim))
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chan_dim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # (CONV => RELU) * 2 => POOL
        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chan_dim))
        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chan_dim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # first (and only) set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(1024))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        # softmax classifier
        model.add(Dense(classes))
        model.add(Activation(final_act))

        # return the constructed network architecture
        return model

    def build_neural_network2(self, classes, final_act):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=self.input_shape, padding='same'))
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D((2, 2), padding='same'))
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(LeakyReLU(alpha=0.1))
        model.add(Dense(classes, activation=final_act))

        return model

    def train(self, model, data, labels, total_samples, epochs, init_lr, batch_size):
        (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, random_state=42)

        # construct the image generator for data augmentation TODO maybe needed or not?
        aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode="nearest")

        # initialize the optimizer (SGD is sufficient) TODO SGD?
        opt = Adam(lr=init_lr, decay=init_lr / epochs)

        # compile the model using binary cross-entropy TODO metric - binary?
        model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["binary_accuracy"])
        model.summary()

        # train the network
        # TODO removed steps_per_epoch=len(trainX) // batch_size this is default i think, with aug.flow, data is inf? so have to keep it
        #H = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=epochs, verbose=1, batch_size=batch_size)
        H = model.fit(
            x=aug.flow(trainX, trainY, batch_size=batch_size),
            validation_data=(testX, testY),
            steps_per_epoch=len(trainX) // batch_size,
            epochs=epochs, verbose=1)

        model.save("chess_piece_cnn.h5py", save_format="h5")

        return H

    def operate(self, total_samples, epochs, init_lr, batch_size):
        self.training_data, labels = image_loader.load_images(self.file_names, self.input_shape[1], self.input_shape[0], total_samples)
        #visualize_data.display_training_data(self.training_data, [0, 1])

        mlb = MultiLabelBinarizer()
        labels_binary = mlb.fit_transform(labels)
        print(mlb.classes_)

        # save the multi-label binarizer to disk
        f = open('mlb.pickle', "wb")
        f.write(pickle.dumps(mlb))
        f.close()

        # initialize the model using a sigmoid activation as the final layer
        self.model = self.build_neural_network2(classes=len(mlb.classes_), final_act="sigmoid")

        # train model
        self.train_network = self.train(self.model, self.training_data, labels_binary,
                                        total_samples=total_samples, epochs=epochs,
                                        init_lr=init_lr, batch_size=batch_size)

        visualize_data.display_training_results(self.train_network, 'binary')

    def predict_png(self, file_name):
        np_data = image_loader.load_image(file_name, self.input_shape[1], self.input_shape[0])
        predicted_classes = self.model.predict(np_data)
        print(np.around(predicted_classes, 2))


IMG_DIR = r"A:\repo\chess-sim\Chess Simulation\Images"
chessPieceCNN = ChessPieceCNN(IMG_DIR, 100, 100, 1)
chessPieceCNN.operate(total_samples=500, epochs=20, init_lr=1e-3, batch_size=1)
chessPieceCNN.predict_png(r"A:\repo\chess-sim\Chess Simulation\Images\black_knight_67.png")
chessPieceCNN.predict_png(r"A:\repo\chess-sim\Chess Simulation\Images\white_pawn_69.png")
