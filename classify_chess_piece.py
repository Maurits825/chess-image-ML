from utils import image_loader
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import pickle
import cv2
import os
from PIL import Image


def classify_chess_piece(model_name, label_binary, file_name, width, height):

    np_data = image_loader.load_image(file_name, width, height)
    image = cv2.imread(file_name)
    output = cv2.resize(image, (500, 500))

    # load the trained convolutional neural network and the multi-label binarizer
    print("[INFO] loading network...")
    model = load_model(model_name)
    mlb = pickle.loads(open(label_binary, "rb").read())

    # classify the input image then find the indexes of the two class
    # labels with the *largest* probability
    print("[INFO] classifying image...")
    proba = model.predict(np_data)[0]
    idxs = np.argsort(proba)[::-1][:2]

    # loop over the indexes of the high confidence class labels
    for (i, j) in enumerate(idxs):
        # build the label and draw the label on the image
        label = "{}: {:.2f}%".format(mlb.classes_[j], proba[j] * 100)
        cv2.putText(output, label, (10, (i * 30) + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    # show the probabilities for each of the individual labels
    for (label, p) in zip(mlb.classes_, proba):
        print("{}: {:.2f}%".format(label, p * 100))
    # show the output image
    cv2.imshow("Output", output)
    cv2.waitKey(0)


classify_chess_piece("chess_piece_cnn.h5py", "mlb.pickle", r"wp1.jpg", 100, 100)
