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

    def train(self, total_samples, epochs, init_lr, batch_size):
        # initialize the number of epochs to train for, initial learning rate,
        # batch size, and image dimensions

        data, labels = image_loader.load_images(self.file_names, self.input_shape[1], self.input_shape[0], total_samples)
        mlb = MultiLabelBinarizer()
        labels_binary = mlb.fit_transform(labels)

        (trainX, testX, trainY, testY) = train_test_split(data, labels_binary, test_size=0.2, random_state=42)

        # construct the image generator for data augmentation TODO maybe needed or not?
        aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
                                 height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                                 horizontal_flip=True, fill_mode="nearest")

        # initialize the model using a sigmoid activation as the final layer
        model = self.build_neural_network(classes=len(mlb.classes_), final_act="sigmoid")

        # initialize the optimizer (SGD is sufficient) TODO SGD?
        opt = Adam(lr=init_lr, decay=init_lr / epochs)

        # compile the model using binary cross-entropy TODO metric - binary?
        model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
        # train the network
        # TODO removed steps_per_epoch=len(trainX) // batch_size this is default i think, with aug.flow, data is inf? so have to keep it
        H = model.fit(
            x=aug.flow(trainX, trainY, batch_size=batch_size),
            steps_per_epoch=len(trainX) // batch_size,
            validation_data=(testX, testY), epochs=epochs, verbose=1)

        model.save("chess_piece_cnn", save_format="h5")

        return H


IMG_DIR = r"A:\repo\chess-sim\Chess Simulation\Images"
chessPieceCNN = ChessPieceCNN(IMG_DIR, 100, 100, 1)
chess_train = chessPieceCNN.train(total_samples=5, epochs=20, init_lr=1e-3, batch_size=1)
visualize_data.display_training_results(chess_train)
