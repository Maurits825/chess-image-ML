from utils import image_loader
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import pickle
import cv2
import os
import glob
from PIL import Image


class ClassifyChessPiece:
    def __init__(self, model_name, label_name):
        self.model_name = model_name
        self.label_name = label_name

        self.model = None
        self.label = None

    def load_model_and_label(self):
        self.model = load_model(self.model_name)
        self.label = pickle.loads(open(self.label_name, "rb").read())

    def classify_chess_piece(self, file_name, width, height):

        np_data = image_loader.load_image(file_name, width, height)
        image = cv2.imread(file_name)
        output = cv2.resize(image, (500, 500))

        # classify the input image then find the indexes of the two class
        # labels with the *largest* probability
        print("[INFO] classifying image...")
        proba = self.model.predict(np_data)[0]
        idxs = np.argsort(proba)[::-1][:4]

        # loop over the indexes of the high confidence class labels
        for (i, j) in enumerate(idxs):
            # build the label and draw the label on the image
            label = "{}: {:.2f}%".format(self.label.classes_[j], proba[j] * 100)
            cv2.putText(output, label, (10, (i * 30) + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        # show the probabilities for each of the individual labels
        for (label, p) in zip(self.label.classes_, proba):
            print("{}: {:.2f}%".format(label, p * 100))
        # show the output image
        cv2.imshow("Output", output)
        cv2.waitKey(0)


image_dir = r"test images/"
images = glob.glob(image_dir + "/*")
classifyChessPiece = ClassifyChessPiece("chess_piece_cnn.h5py", "mlb.pickle")
classifyChessPiece.load_model_and_label()

for img in images:
    classifyChessPiece.classify_chess_piece(img, 100, 100)

#TODO put consol cmds, import click from AFKScape
