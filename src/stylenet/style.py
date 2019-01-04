#
# style.py
# Artistic Style Transfer Metrics and Losses
# as defined in Gatys et. al
#

import numpy as np
import tensorflow as tf
import keras.backend as K

from PIL import Image
from keras.models import Model
from keras.layers import InputLayer
from keras.applications.vgg16 import VGG16

# Style transfer settings
IMAGE_SHAPE = (512, 512, 3)
INPUT_SHAPE = (None,) + IMAGE_SHAPE

# Loss computation weights
CONTENT_WEIGHT = 0.05
STYLE_WEIGHT = 50
TOTAL_VARIATION_WEIGHT = 0

# Feature extraction using CNN model
CONTENT_LAYER = 'block2_conv2'
STYLE_LAYERS = ['block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3',
                'block5_conv3']
TOTAL_VARIATION_LAYER = "block1_conv1"

## Data Preprocessing
# Crop the given image to a square frame of x by x
# where x is the length of the shorter side of the image
def crop_center(image):
    # Compute new dimentions for image
    # Crop a centered square from the image
    target_dim = min(image.size)
    len_x, len_y = image.size

    begin_y = (len_y // 2) - (target_dim // 2)
    end_y  = (len_y // 2) + (target_dim // 2)
    
    begin_x = (len_x // 2) - (target_dim // 2)
    end_x  = (len_x // 2) + (target_dim // 2)
    
    # Perform crop for computed dimentions
    image = image.crop((begin_x, begin_y, end_x, end_y))
    return image

# Preprocesses the image by the given Pillow image
# swaps the image channels to BGR and subtracts the RGB mean value
def preprocess_image(image):
    # Center crop so we can resize without distortion
    # Resize image to standardise input
    image = crop_center(image)
    image = image.resize(IMAGE_SHAPE[:-1])
    img_mat = np.array(image, dtype="float32")
    
    # Subtract mean value
    img_mat[:, :, 0] -= 103.939
    img_mat[:, :, 1] -= 116.779
    img_mat[:, :, 2] -= 123.68

    # Swap RGB to BGR
    img_mat = img_mat[:, :, ::-1]
    return img_mat

# Reverses the preprocessing done on the given matrix
# swaps the image channels to RGB and adds the RGB mean value
# Clips image values to valid range and converts matrix to Pillow image
# Returns processed image
def deprocess_image(img_mat):
    img_mat = np.copy(img_mat)
    img_mat = np.reshape(img_mat, IMAGE_SHAPE)
    # Swap BGR to RGB
    img_mat = img_mat[:, :, ::-1]

    # Add mean value
    img_mat[:, :, 0] += 103.939
    img_mat[:, :, 1] += 116.779
    img_mat[:, :, 2] += 123.68

    # Convert to image
    img_mat = np.clip(img_mat, 0, 255).astype('uint8')
    image = Image.fromarray(img_mat)

    return image

