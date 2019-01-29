import os

from os import listdir

import numpy as np
import string
import tqdm

from PIL import Image, ImageDraw, ImageFont
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer
from skimage.transform import rotate, SimilarityTransform, warp
from skimage.util import random_noise

# Adds a random rotation to the image using a random angle from -60 to 60 degrees
# It runs the operation with probability equal to rate
def random_rotation(image, angle_range=(-60, 60), rate=0.6):
    if np.random.rand() < rate:
        angle = np.random.randint(*angle_range)
        image = rotate(image, angle, cval=1)
    return image

# Adds random noise (white pixels to the image)
# It runs the operation with probability equal to rate
def random_noise_prob(image, rate=0.6):
    if np.random.rand() < rate:
        image = random_noise(image=image, mode='s&p', salt_vs_pepper=1)
    return image

# Adds a random translation on the x and y axis.
# It runs the operation with probability equal to rate
def random_translation(image, rate=0.6):
    if np.random.rand() < rate:
        x = np.random.uniform(high=20, low=-20)
        y = np.random.uniform(high=20, low=-10)
        tform = SimilarityTransform(translation=(x, y))
        image = warp(image, tform, cval=1)
    return image

if __name__ == "__main__":


    # Loads the original data
    X_data = np.load("../data/X_data.npy")
    Y_fonts = np.load("../data/Y_fonts.npy")
    Y_chars = np.load("../data/Y_chars.npy")
    Y_bold = np.load("../data/Y_bold.npy")
    Y_italic = np.load("../data/Y_italic.npy")

    # Number of sample images to be generated
    nsample = 186120
    # np arrays to save the new generated images
    np_data = np.zeros(shape=(nsample, 64, 64, 1), dtype="float")
    np_fonts = np.empty(shape=(nsample), dtype=object)
    np_chars = np.empty(shape=(nsample) , dtype=str)
    np_bold = np.zeros(shape=(nsample))
    np_italic = np.zeros(shape=(nsample))
    
    # To control how many new images will be generated for each font
    # We define these numbers is a way to have a balanced dataset when it comes
    # to the number of instances for each font, but we don't consider the bold/italic case
    # this means that chars and fonts will be balanced but bold and italic won't.
    fonts_dict = {
        'AlexBrush': 180, 
        'Aller': 45, 
        'Amatic': 90, 
        'GreatVibes': 180, 
        'Lato': 60, 
        'OpenSans': 45,
        'Oswald': 45, 
        'Pacifico': 180, 
        'Quicksand': 45, 
        'Roboto': 60, 
        'blackjack': 180,    
    }
    total = 0
    # Generate new images
    for i in tqdm.tqdm(range(X_data.shape[0])):
        # Get original labels
        is_bold = Y_bold[i]
        is_italic = Y_italic[i]
        font = Y_fonts[i]
        char = Y_chars[i]
        image = X_data[i]
        count = 0
        while count < fonts_dict[font]:
            # Apply each transformation.
            new_image = random_noise_prob(image)
            new_image = random_rotation(new_image)
            # Always apply translation to guarantee diversity on data
            new_image = random_translation(new_image, rate=1)
            # Save new image and new labels on arrays
            np_data[total] = new_image
            np_fonts[total] = font
            np_chars[total] = char
            np_bold[total] = is_bold
            np_italic[total] = is_italic
            count += 1
            total += 1
    # Save augmented data
    np.save("../data/X_data_augmented.npy", np_data)
    np.save("../data/Y_bold_augmented.npy", np_bold)
    np.save("../data/Y_italic_augmented.npy", np_italic)
    np.save("../data/Y_fonts_augmented.npy", np_fonts)
    np.save("../data/Y_chars_augmented.npy", np_chars)

