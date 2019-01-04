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

## Utilities
# Get a dictionary of the layers and corresponding tensors of the NN
def get_layers(model):
    layers = dict([(layer.name, layer.output) for layer in model.layers if layer.name != "input_1"])
    return layers

# Compute the gram matrix for the given tensor
# Gram matrix computes the correlations between each feature in x
# Return the computed gram matrix
def compute_gram_mat(tensor):
    # Batch Flatten tensor into single vector per batch to compute gram matrix
    tensor = K.permute_dimensions(tensor, (2, 0, 1))
    features_mat  = K.batch_flatten(tensor)
    # Compute gram matrix of features
    # G = F * F'
    gram_mat = K.dot(features_mat, K.transpose(features_mat))

    return gram_mat

# Extract tensor at the given position from the layer specified by layer_name
# from the given model
# Returns the specified tensor
def extract_tensor(model, i_position, layer_name):
    layers = get_layers(model)
    layer = layers[layer_name]
    tensor = layer[i_position, :, :, :]
    return tensor

## Loss Functions
# Builds the tensor for content loss for the content and style tensors indexes
# using features extracted from the given model
# Defines how different each image is different for each other, with a higher 
# content loss meaning larger difference in content
def build_content_loss(content_idx, pastiche_idx, model):
    # Extract content and pastiche features from model
    content = extract_tensor(model, content_idx, CONTENT_LAYER)
    pastiche = extract_tensor(model, pastiche_idx, CONTENT_LAYER)

    # Lc = sum((Fc - Fp)^2
    return K.sum(K.square(content - pastiche))

# Build the tensor that will find the style loss for the style 
# and pastiche tensor indexes, using features extracted from the given model
# Defines how images differ in style, a higher style lost meaning 
# that the images differ more in style.
def build_style_loss(style_idx, pastiche_idx, model):
    # Tabulate style loss for all style layers
    style_loss = K.variable(0.0, name="style_loss")

    for layer_name in STYLE_LAYERS:
        # Extract features from layer
        style = extract_tensor(model, style_idx, layer_name)
        pastiche = extract_tensor(model, pastiche_idx, layer_name)
    
        # Compute gram matrixes
        style_gram = compute_gram_mat(style)
        pastiche_gram = compute_gram_mat(pastiche)

        # Compute style loss for layer
        # Ls = sum((Pl - Gl)^2) / (4 * Nl^2 * Ml ^ 2)
        N, M = 3, IMAGE_SHAPE[0] * IMAGE_SHAPE[1]
        layer_style_loss = K.sum(K.square(pastiche_gram - style_gram)) / \
            (4 * (N ** 2) * (M ** 2))

        style_loss = style_loss + layer_style_loss
    
    return style_loss

# Build the computational graph that will find the total variation loss for 
# given pastiche tensor index using features extracted from the given model
# This loss regularises the generated image, removing unwanted dnoise
def build_total_variation_loss(pastiche_idx, model):
    # Extract features from model
    pastiche = extract_tensor(model, pastiche_idx, TOTAL_VARIATION_LAYER)
    
    # Compute variation for each image axis
    height, width = IMAGE_SHAPE[0], IMAGE_SHAPE[1]
    height_variation = K.square(pastiche[:height-1, :width-1 :] - 
                                pastiche[1:, :width-1, :])
    width_variation = K.square(pastiche[:height-1, :width-1, :] - 
                               pastiche[:height-1, 1:, :])

    # V(y) = sum(V(h) - V(w))
    total_variation = K.sum(K.abs(width_variation + height_variation))
    
    return total_variation

# Build the tensor that will find the the total loss: a weighted
# sum of the total varaition, style and content losses for the given
# pastiche style content tensor
# Determines the optimisation problem in which style transfer is performed in 
# minimising this loss
def build_loss(pastiche, style, content):
    # Build input tensor
    pastiche_idx, style_idx, content_idx = 0, 1, 2
    input_op = K.stack([pastiche, style, content])
    model = VGG16(input_tensor=input_op, weights='imagenet', include_top=False)

    # Freeze model from being trained 
    for layer in model.layers:
        layer.trainable = False

    # Compute total loss
    content_loss = build_content_loss(content_idx, pastiche_idx, model)
    style_loss = build_style_loss(style_idx, pastiche_idx, model)
    total_variation_loss = build_total_variation_loss(pastiche_idx, model)
    
    # L = Wc * Lc + Ws * Ls + Wv + Lv
    loss = CONTENT_WEIGHT * content_loss + \
        STYLE_WEIGHT * style_loss + \
        TOTAL_VARIATION_WEIGHT * total_variation_loss

    return loss

if __name__ == "__main__":
    # Test build loss
    pastiche = K.placeholder(IMAGE_SHAPE)
    style = K.placeholder(IMAGE_SHAPE)
    content = K.placeholder(IMAGE_SHAPE)

    build_loss(pastiche, style, content)
