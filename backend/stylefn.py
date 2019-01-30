#
# stylefn.py
# Artistic Style Transfer Metrics and Losses
# as defined in Gatys et. al
#

import api
import numpy as np
import tensorflow as tf
import keras.backend as K
import matplotlib.pyplot as plt
 
from PIL import Image
from datetime import datetime
from keras.models import Model
from util import apply_settings
from keras.layers import InputLayer
from keras.applications.vgg19 import VGG19

# Style Transfer Function settings
# NOTE: the following are default settings and may be overriden
SETTINGS = {
    "image_shape": (512, 512, 3),

    # Loss computation weights
    "content_weight": 1e+6,
    "style_weight": 4e+2,
    "denoise_weight": 2e-3,

    # Layers for feature extraction
    "content_layers": ['block3_conv3'],
    "style_layers": ['block1_conv2', 'block2_conv1', 'block3_conv1', 'block4_conv1',
                  'block5_conv1'],
    "denoising_layers": [ "input_1" ]
}

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
# Resizes image to the given target shape
# swaps the image channels to BGR and subtracts the BGR mean value
# Returs the preprocessed image
IMG_BGR_MEAN = np.asarray((103.939, 116.779, 123.68))
def preprocess_image(image, shape):
    # Center crop so we can resize without distortion
    # Resize image to standardise input
    image = crop_center(image)
    image = image.resize(shape[:-1])
    img_mat = np.array(image, dtype="float32")
    
    # Swap RGB to BGR
    img_mat = img_mat[:, :, ::-1]
    # Subtract BGR mean value
    img_mat -= IMG_BGR_MEAN

    return img_mat

# Reverses the preprocessing done on the given matrix
# Reshapes the image to the given target shape
# adds the BGR mean value and swaps the image channels to RGB
# Clips image values to valid range and converts matrix to Pillow image
# Returns deprocessed image
def deprocess_image(img_mat, shape):
    img_mat = np.copy(img_mat)
    img_mat = np.reshape(img_mat, shape)

    # Add BGR mean value 
    img_mat += IMG_BGR_MEAN
    # Swap BGR to RGB
    img_mat = img_mat[:, :, ::-1]

    # Convert to image
    img_mat = np.clip(img_mat, 0, 255).astype('uint8')
    image = Image.fromarray(img_mat)

    return image

## Feature extraction
# Build and returns a model that returns features tensors given an input tensor
# that extracts features based on the given layers names.
# Extract Model will extract features from an input of given shape 
def build_extractor(layer_names, shape):
    with tf.name_scope("feature_extractor"):
        # Load VGG model
        vgg_model = VGG19(input_shape=shape, weights="imagenet", include_top=False)
        vgg_model.trainable = False

        # Build  model by extracting  feature tensors
        layers = [ vgg_model.get_layer(name) for name in layer_names ]
        feature_ops = [ layer.output for layer in layers ] 
        model = Model(inputs=[vgg_model.input], outputs=feature_ops)
        
        # Ensure that model always outputs list
        def wrapped_model(x):
            y = model(x)
            return y if type(y) is list else [y]

        return wrapped_model

# Build and return the gram matrix tensor for the given input features tensor
def build_gram_matrix(input_op):
    with tf.name_scope("gram_matrix"):
        # Flatten input feature dimention tensor, leaving channels dimention
        n_channels = input_op.shape[-1].value
        input_op = tf.reshape(input_op, (-1, n_channels))
        
        # Compute gram matrix for correlations between features
        gram_mat_op = tf.matmul(K.transpose(input_op), input_op)
        return gram_mat_op

## Loss functions
# Build and return tensor that computes content loss given the 
# pastiche and content image tensors using the content features extracted from 
# content layers specified by content_layers
# Weights content loss by the given content weight
def build_content_loss(pastiche_op, content_op, content_layers, content_weight):
    with tf.name_scope("content_loss"):
        # Shape tensors for feature extraction
        pastiche_op = tf.expand_dims(pastiche_op, axis=0)
        content_op = tf.expand_dims(content_op, axis=0)
        
        # Extract content features using content extractor
        op_shape = pastiche_op.shape.as_list()[1:]
        extractor = build_extractor(content_layers, op_shape)
        pastiche_feature_ops = extractor(pastiche_op)
        content_feature_ops = extractor(content_op)
        
        # Compute content loss
        scale_factor = op_shape[0] * op_shape[1] * op_shape[2]
        loss_op = tf.multiply(content_weight / scale_factor,
                              tf.reduce_sum(tf.squared_difference(
                                  pastiche_feature_ops, content_feature_ops)),
                              name="content_loss")

        # Track content loss with tensorboard
        loss_summary = tf.summary.scalar("content_loss", loss_op)

        return loss_op

# Build and return a tensor that computes style loss given the
# pastiche and style image tensors using the style features extracted from 
# style layers specified by style_layers
# Loss attempts to reduce the noise in the generated images
# Weights style loss by the given style weight
def build_style_loss(pastiche_op, style_op, style_layers, style_weight):
    with tf.name_scope("style_loss"):
        # Shape tensors for feature extraction
        pastiche_op = tf.expand_dims(pastiche_op, axis=0)
        style_op = tf.expand_dims(style_op, axis=0)
        
        # Extract style features using style extractor
        op_shape = pastiche_op.shape.as_list()[1:]
        extractor = build_extractor(style_layers, op_shape)
        pastiche_feature_ops = extractor(pastiche_op)
        style_feature_ops = extractor(style_op)
        
        # Build style loss tensor for each layer
        def build_layer_style_loss(layer_name, pastiche_feature_op, style_feature_op):
            with tf.name_scope("layer_style_loss"):
                # Extract style feawtures using gram matrix
                scale_factor = op_shape[0] * op_shape[1] * op_shape[2]
                pastiche_gram_op = build_gram_matrix(pastiche_feature_op) / scale_factor
                style_gram_op = build_gram_matrix(style_feature_op) / scale_factor

                # Compute style loss for layer
                layer_loss_name = layer_name + "_loss"
                layer_loss_op = tf.reduce_sum(tf.squared_difference(pastiche_gram_op,
                                                                    style_gram_op),
                                              name=layer_loss_name)
                return layer_loss_op

        layer_loss_ops = [ build_layer_style_loss(N, P, S) for N, P, S in 
                          zip(SETTINGS["style_layers"], 
                              pastiche_feature_ops, style_feature_ops) ]
    
        # Compute total style loss accross layers
        loss_op = tf.multiply(style_weight, tf.reduce_mean(layer_loss_ops), 
                              name="style_loss")
        # Track content loss with tensorboard
        loss_summary = tf.summary.scalar("style_loss", loss_op)

        return loss_op

# Build and return a tensor that computes total variation loss (noise loss)
# Loss attempts to reduce the noise in the generated images
# Weights noise loss by the given denoise weight
def build_noise_loss(pastiche_op, denoise_weight):
    with tf.name_scope("noise_loss"):
        #TODO: implement multiple layers
        # Compute variation accross image axis
        height_variation_op = tf.reduce_sum(K.abs(pastiche_op[:-1, :, :] - 
                                                   pastiche_op[1:, :, :]))
        width_variation_op = tf.reduce_sum(K.abs(pastiche_op[:, :-1, :] - 
                                                  pastiche_op[:, 1:, :]))
        
        loss_op = tf.multiply(denoise_weight,
                              tf.add(height_variation_op, width_variation_op),
                              name="total_variation_loss")
        
        # Track total variation with tensorboard
        loss_summary = tf.summary.scalar("total_variation_loss", loss_op)

        return loss_op

# Build and return tensor that total style transfer loss given the
# pastiche, content and style image tensors, as configured by the given settings 
# overrides see SETTINGS for configurable settings
def build_loss(pastiche_op, content_op, style_op, settings={}):
    # Apply setting overrides
    settings = apply_settings(settings, SETTINGS)
    
    with tf.name_scope("style_transfer_loss"):
        content_loss_op = build_content_loss(pastiche_op, content_op, 
                                             settings["content_layers"],
                                             settings["content_weight"])
        style_loss_op = build_style_loss(pastiche_op, style_op,
                                         settings["style_layers"],
                                         settings["style_weight"])
        noise_loss_op = build_noise_loss(pastiche_op, settings["denoise_weight"])

        # Total loss weight sum of content, style and noise losses
        loss_op = tf.add_n([content_loss_op, style_loss_op, noise_loss_op])

        # Track style transer loss loss with tensorboard
        loss_summary = tf.summary.scalar("style_transfer_loss", loss_op)
        
        return loss_op
