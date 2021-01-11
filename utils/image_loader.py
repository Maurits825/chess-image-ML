import numpy as np
from PIL import Image
import os
import random


def load_images(file_names, img_width, img_height, total_samples, shuffle=True):
    total_images = len(file_names)
    if total_samples > total_images:
        raise ValueError

    if shuffle:
        random.shuffle(file_names)

    image_data = np.zeros([total_samples, img_height, img_width, 1])
    labels = []
    for idx, img in enumerate(file_names):
        if idx >= total_samples:
            break

        image_data[idx] = load_image(img, img_width, img_height)
        labels.append(get_label(img))

    return image_data, np.array(labels)


def load_image(file_name, img_width, img_height):
    image_full = Image.open(file_name)
    image_grayscale = image_full.convert('L')
    image_np = np.asarray(image_grayscale)
    return image_np.reshape(1, img_height, img_width, 1).astype('float32') / 255.


def get_label(file_name):
    label = file_name.split(os.path.sep)[-1].split("_")
    return label[0:2]
