#
# stylenet.py
# Artistic Style Transfer
#

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from PIL import Image
from keras import backend as K
from keras.models import Model
from keras.applications.vgg16 import VGG16
from shutil import rmtree
from tensorflow.contrib.opt import ScipyOptimizerInterface 
from scipy.optimize import fmin_l_bfgs_b

# Model Settings
IMAGE_DIM = (128, 128)
IMAGE_SHAPE = IMAGE_DIM + (3,)
INPUT_SHAPE = (None,) + IMAGE_SHAPE
CONTENT_WEIGHT = 0.05
STYLE_WEIGHT = 50
TOTAL_VARIATION_WEIGHT = 1

CONTENT_LAYER = 'block2_conv2'
STYLE_LAYERS = ['block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3',
                'block5_conv3']

CONTENT_INDEX = 0
STYLE_INDEX = 1
PASTICHE_INDEX = 2

## Data Pre Processing
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

# Load the image at path
# Reshapes the image for use with the stylenet model by converting it to a 
# IMAGE_DIM square.
# Returns the reshaped image as np array
def load_image(path):
    image = Image.open(path)
    # Center crop so we can resize without distortion
    image = crop_center(image)
    image = image.resize(IMAGE_DIM)
    return np.array(image, dtype=np.float32)

## Data Preprocessing
# Preprocesses the image by the given image matrix img_mat
# swaps the image channels to BGR and subtracts the RGB mean value
def preprocess_image(img_mat):
    # Subtract mean value
    img_mat[:, :, 0] -= 103.939
    img_mat[:, :, 1] -= 116.779
    img_mat[:, :, 2] -= 123.68

    # Swap RGB to BGR
    img_mat = img_mat[:, :, ::-1]
    return img_mat

# Reverses the preprocessing done on the given matrix
# swaps the image channels to RGB and adds the RGB mean value
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

## Feature extraction 
# Get a dictionary of the layers and corresponding tensors of the VGG16 NN
def get_vgg_layers(model):
    return dict([(layer.name, layer.output) for layer in model.layers])

## Loss Functions
# Builds the computational for that finds the content loss for content and 
# pasitche features extracted from the given VGG layers
# Defines how different each image is different for each other, with a higher 
# content loss meaning larger difference in content
def build_content_loss(layers):
    # Extract content and pasitche features from content layer
    layer = layers[CONTENT_LAYER]
    content = layer[CONTENT_INDEX, :, :, :]
    pastiche = layer[PASTICHE_INDEX, :, :, :]

    # Lc = sum((Fc - Fp)^2
    return CONTENT_WEIGHT * K.sum(K.square(content - pastiche))

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

# Build the computational graph that will find the style loss for the style 
# and pastiche features extracted from the given VGG layers
# Defines how images differ in style, a higher style lost meaning 
# that the images differ more in style.
def build_style_loss(layers):
    # Tabulate style loss for all style layers
    # Tabulate style loss for all style layers
    style_loss = K.variable(0.0, name="style_loss")

    for layer_name in STYLE_LAYERS:
        # Extract style and pastiche features from layer
        layer = layers[layer_name]
        style = layer[STYLE_INDEX, :, :, :]
        pastiche = layer[PASTICHE_INDEX, :, :, :]

        # Compute gram matrixes
        style_gram = compute_gram_mat(style)
        pastiche_gram = compute_gram_mat(pastiche)

        # Compute style loss for layer
        # Ls = sum((Pl - Gl)^2) / (4 * Nl^2 * Ml ^ 2)
        N, M = 3, IMAGE_DIM[0] * IMAGE_DIM[1]
        layer_style_loss = K.sum(K.square(pastiche_gram - style_gram)) / \
            (4 * (N ** 2) * (M ** 2))

        style_loss = style_loss + ((STYLE_WEIGHT / len(STYLE_LAYERS)) * 
                                   layer_style_loss)
    
    return style_loss

# Build the computational graph that will find the total variation loss for 
# given pastiche features 
# This loss regularises the generated image, removing unwanted dnoise
def build_total_variation_loss(pastiche):
    height, width = IMAGE_DIM
    
    # Compute variation for each image axisw
    height_variation = K.square(pastiche[:height-1, :width-1 :] - 
                                pastiche[1:, :width-1, :])
    width_variation = K.square(pastiche[:height-1, :width-1, :] - 
                               pastiche[:height-1, 1:, :])

    # V(y) = sum(V(h) - V(w))
    total_variation = K.sum(K.abs(width_variation + height_variation))
    
    return TOTAL_VARIATION_WEIGHT * total_variation

# Build the computational graph that will find the the total loss: a weight 
# sum of the total varaition, style and content losses. Determines the 
# optimisation problem in which style transfer is performed in minimising this loss
def build_loss(pastiche_op, layers):
    # Compute total loss
    content_loss = build_content_loss(layers)
    style_loss = build_style_loss(layers)
    total_variation_loss = build_total_variation_loss(pastiche_op)
    
    # L = Wc * Lc + Ws * Ls + Wv + Lv
    loss = content_loss + style_loss + total_variation_loss

    return loss

# Defines a style transfer transfusion
class Transfusion:
    def __init__(self, pastiche_op, loss_op):
        self.pastiche = np.random.normal(0 - 128, 255 - 128, IMAGE_SHAPE)

        # Build evaluation function
        gradients_op = K.gradients(loss_op, pastiche_op)[0]
        inputs, outputs = [pastiche_op], [loss_op, gradients_op]
        self.evaluate = K.function(inputs, outputs)
        
    
    # Perform a iteration of style transfer on the pastiche
    # Returns the current loss and pastiche
    def transfer(self):
        pastiche = self.pastiche.ravel()
        pastiche, loss, info = fmin_l_bfgs_b(self.compute_loss, 
                                      pastiche,
                                      fprime=self.compute_gradients,
                                      maxfun=20)
        self.pastiche = pastiche.reshape(IMAGE_SHAPE)
        
        return loss, self.pastiche
        
    # Computes and returns the loss on the current state of the pastiche
    def compute_loss(self, pastiche):
        pastiche = np.reshape(pastiche, IMAGE_SHAPE)
        self.loss, self.gradients = self.evaluate([ pastiche ])
        return self.loss
    
    # Computes and returns the gradients on the current state of the pastiche 
    def compute_gradients(self, pastiche):
        gradients = self.gradients.ravel().astype("float64")
        return gradients
    
## Optmisation
if __name__ == "__main__":
    rmtree("pastiche", ignore_errors=True)
    os.mkdir("pastiche")
    
    # Setup data tensors
    # Load images
    content_img_mat = load_image("./data/Tuebingen_Neckarfront.jpg")
    style_img_mat = load_image("./data/stary_night.jpg")

    # Preprocess images
    content_img_mat = preprocess_image(content_img_mat)
    style_img_mat = preprocess_image(style_img_mat)
    
    # Create tensors for the images 
    content_op = K.constant(content_img_mat)
    style_op = K.constant(style_img_mat)
    
    # Create pastiche tensor
    pastiche_op = K.placeholder(IMAGE_SHAPE, dtype="float32")

    # Stack tensors into single input tensor
    stack = [None] * 3
    stack[CONTENT_INDEX] = content_op
    stack[STYLE_INDEX] = style_op
    stack[PASTICHE_INDEX] = pastiche_op
    input_op = K.stack(stack)  

    # Load VGG16 model for feature extraction
    vgg_model = VGG16(input_tensor=input_op, weights='imagenet',
                  include_top=False)
    layers = get_vgg_layers(vgg_model)

    # Build loss computational graph
    loss_op = build_loss(pastiche_op, layers)
    
    # Optimise cost to perform style transfer to produce pastiche
    transfusion = Transfusion(pastiche_op, loss_op)

    n_iterations = 10
    for i in range(n_iterations):
        start_time = time.time()
        
        loss, pastiche = transfusion.transfer()

        # Display progress
        print("loss: ", loss)
        end_time = time.time()
        print('Iteration %d completed in %ds' % (i, end_time - start_time))
        
        pastiche_img = deprocess_image(pastiche)
        pastiche_img.save("pastiche/{}.jpg".format(i))
