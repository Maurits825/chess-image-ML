from PIL import Image
from PIL import ImageFilter
import glob
import numpy as np
import os
import csv
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import to_categorical
from keras import backend as K
from keras.layers.core import Activation
from keras.optimizers import Adam

total_samples = 2

#TODO fix this
def load_test():
    image_full = Image.open(r"A:\repo\chess-image-ML\test.png")
    image_grayscale = image_full.convert('L')
    image_np = np.asarray(image_grayscale)
    return image_np.reshape(1, 100, 100, 1)


def load_images_to_np_array():
    image_dir = r"A:\repo\chess-sim\Chess Simulation\Images"
    image_glob = glob.glob(image_dir + "/*.png")

    img_width = 100
    img_height = 100
    in_shape = (img_height, img_width, 1)
    image_data = np.zeros([total_samples, img_height, img_width, 1])  #len(os.listdir(image_dir)) - 1
    for idx, img in enumerate(image_glob):
        image_full = Image.open(img)
        image_grayscale = image_full.convert('L')
        image_np = np.asarray(image_grayscale)
        image_data[idx] = image_np.reshape(img_height, img_width, 1)

        if idx >= total_samples-1:
            break

    return image_data, in_shape


def load_image_labels():
    filename = r"A:\repo\chess-sim\Chess Simulation\Images\board_position.csv"
    labels = []
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        i=0
        for row in csv_reader:
            i = i + 1
            labels.append(row)
            if i >= total_samples*2:  # 12 rows of labels per sample
                break

    arr = np.asarray(labels).astype('float32').reshape(-1, 64)#reshape(total_samples, 2, 64)
    arr1 = arr[0::2].copy()
    arr2 = arr[1::2].copy()
    return arr1, arr2


def create_training_data(image_data):
    return image_data.astype('float32') / 255.


def display_training_data(data):
    plt.figure(figsize=[5, 5])
    plt.subplot(121)
    plt.imshow(data[0, :, :].squeeze(), cmap='gray')
    plt.title("title")

    plt.subplot(122)
    plt.imshow(data[1, :, :].squeeze(), cmap='gray')
    plt.title("title")

    plt.show()


def relu_advanced(x):
    return K.relu(x, max_value=2)


# ref: https://towardsdatascience.com/building-a-multi-output-convolutional-neural-network-with-keras-ed24c7bc1178
def make_default_hidden_layers(inputs):
    """
    Used to generate a default set of hidden layers. The structure used in this network is defined as:

    Conv2D -> BatchNormalization -> Pooling -> Dropout
    """
    x = Conv2D(128, (3, 3), padding="same")(inputs)
    x = Activation("relu")(x)
    x = LeakyReLU(alpha=0.1)(x)
    #x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(3, 3))(x)
    #x = Dropout(0.25)(x)

    x = Conv2D(256, (3, 3), padding="same")(x)
    x = Activation("relu")(x)
    x = LeakyReLU(alpha=0.1)(x)
    #x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    #x = Dropout(0.25)(x)

    x = Conv2D(512, (3, 3), padding="same")(x)
    x = Activation("relu")(x)
    x = LeakyReLU(alpha=0.1)(x)
    #x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    #x = Dropout(0.25)(x)

    return x


def build_generic_branch(inputs, num_classes, name):
    """
    Used to build the race branch of our face recognition network.
    This branch is composed of three Conv -> BN -> Pool -> Dropout blocks,
    followed by the Dense output layer.
    """
    x = make_default_hidden_layers(inputs)
    x = Flatten()(x)
    x = Dense(256)(x)
    x = Activation("relu")(x)
    x = LeakyReLU(alpha=0.1)(x)

    x = Flatten()(x)
    x = Dense(128)(x)
    x = Activation("relu")(x)
    x = LeakyReLU(alpha=0.1)(x)

    x = Dense(num_classes)(x)
    x = Activation("sigmoid", name=name)(x) #sigmoid is better for multi-label classification?
    return x


def assemble_full_model(input_shape, num_classes):
    """
    Used to assemble our multi-output model CNN.
    """
    inputs = Input(shape=input_shape)
    white_pawn_branch = build_generic_branch(inputs, num_classes, "white_pawn")
    black_pawn_branch = build_generic_branch(inputs, num_classes, "black_pawn")

    model = Model(inputs=inputs, outputs=[white_pawn_branch, black_pawn_branch])
    return model


def train_model(model, train_x, train_y):
    init_lr = 1e-4
    epochs = 50
    batch_size = 1

    opt = Adam()
    #opt = Adam(lr=init_lr, decay=init_lr / epochs) # TODO
    #loss has to be a regression loss?
    #TODO names
    model.compile(optimizer=opt, loss={
                  'white_pawn': 'binary_crossentropy',
                  'black_pawn': 'binary_crossentropy', },
                  metrics=['accuracy'])
    model.summary()

    #train_X, valid_X, train_label, valid_label = train_test_split(train_x, train_y, test_size=0.2, random_state=44)
    #train_X, valid_X, train_label, valid_label = np.array([train_x[0]]), np.array([train_x[1]]), np.array([train_y[0]]), np.array([train_y[1]])
    train_X = train_x

    t1 = train_y[0]
    t2 = train_y[1] # TODO this is where it breaks with 2+ samples?
    train = model.fit(train_X, [t1, t2], batch_size=batch_size, epochs=epochs, verbose=1, validation_split=0)
    model.save("chess_model2.h5py")

    return model, train


def run_machine_learning(train_x, train_y, input_shape):
    train_X, valid_X, train_label, valid_label = train_test_split(train_x, train_y, test_size=0.2,
                                                                  random_state=44)

    batch_size = 1
    epochs = 20
    num_classes = 64  # 64 squares total?

    chess_model = Sequential()
    chess_model.add(Conv2D(32, kernel_size=(3, 3), activation='linear', input_shape=input_shape, padding='same'))
    chess_model.add(Conv2D(64, (3, 3), activation='linear', padding='same'))
    chess_model.add(LeakyReLU(alpha=0.1))
    chess_model.add(MaxPooling2D((2, 2), padding='same'))

    chess_model.add(Conv2D(128, (3, 3), activation='linear', padding='same'))
    chess_model.add(LeakyReLU(alpha=0.1))
    chess_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

    chess_model.add(Conv2D(256, (3, 3), activation='linear', padding='same'))
    chess_model.add(LeakyReLU(alpha=0.1))
    chess_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

    chess_model.add(Conv2D(512, (3, 3), activation='linear', padding='same'))
    chess_model.add(LeakyReLU(alpha=0.1))
    chess_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

    chess_model.add(Flatten())
    chess_model.add(Dense(256, activation='linear'))
    chess_model.add(LeakyReLU(alpha=0.1))

    chess_model.add(Flatten())
    chess_model.add(Dense(128, activation=relu_advanced))
    chess_model.add(LeakyReLU(alpha=0.1))
    chess_model.add(Dense(num_classes, activation=relu_advanced))

    # try poisson with other activations
    ##MeanSquaredLogarithmicError
    chess_model.compile(loss=keras.losses.MeanSquaredError(), optimizer=keras.optimizers.Adam(),
                        metrics=['accuracy'])

    chess_model.summary()
    # train!
    chess_train = chess_model.fit(x=train_X, y=train_label, batch_size=batch_size, epochs=epochs, verbose=1,
                                  validation_data=(valid_X, valid_label))
    chess_model.save("chess_model.h5py")

    return chess_model, chess_train


def display_results(model, train, test_x, test_y, names):
    test_eval = model.evaluate(test_x, test_y, verbose=0)
    print('Test loss:', test_eval[0])
    print('Test accuracy:', test_eval[1])

    for idx, name in enumerate(names):
        accuracy = train.history[name + '_accuracy']
        val_accuracy = train.history['val_' + name + '_accuracy']
        epochs = range(len(accuracy))

        subplot_idx = 120 + idx + 1 # TODO scale this with more names
        plt.subplot(subplot_idx)
        plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
        plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
        plt.title(name + ' Training and validation accuracy')
        plt.legend()

    plt.figure()

    for idx, name in enumerate(names):
        loss = train.history[name + '_loss']
        val_loss = train.history['val_' + name + '_loss']
        accuracy = train.history[name + '_accuracy']
        epochs = range(len(accuracy))

        subplot_idx = 120 + idx + 1  # TODO scale this with more names
        plt.subplot(subplot_idx)
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title(name + ' Training and validation loss')
        plt.legend()

    plt.show()


image_array, in_shape = load_images_to_np_array()
training_data = create_training_data(image_array)
training_data = training_data.reshape(-1, 100, 100, 1)
test_X = load_test()
temp_y = [[0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
[1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]

test_Y = np.asarray(temp_y).astype('float32').reshape(1, 2, 64)
train_Y1, train_Y2 = load_image_labels()
#display_training_data(training_data)

chess_model = assemble_full_model(in_shape, 64)  # 64 squares, num of classes
ml_model, ml_train = train_model(chess_model, training_data, [train_Y1, train_Y2])

#ml_model, ml_train = run_machine_learning(training_data, train_Y, in_shape)
#display_results(ml_model, ml_train, test_X, test_Y, ["white_pawn", "black_pawn"])

predicted_classes = ml_model.predict(test_X)
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
print(predicted_classes)
