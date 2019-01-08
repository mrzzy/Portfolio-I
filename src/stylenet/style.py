#
# style.py
# Artistic Style Transfer Metrics and Losses
# as defined in Gatys et. al
#

import numpy as np
import tensorflow as tf
import keras.backend as K
import matplotlib.pyplot as plt

from PIL import Image
from datetime import datetime
from keras.models import Model
from keras.layers import InputLayer
from keras.applications.vgg16 import VGG16

# Style transfer settings
IMAGE_SHAPE = (512, 512, 3)

# Loss computation weights
CONTENT_WEIGHT = 1
STYLE_WEIGHT = 1e+4
DENOISE_WEIGHT = 4e-2

# Layers for feature extraction
CONTENT_LAYERS = ['block2_conv2']
STYLE_LAYERS = ['block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3']

DENOISING_LAYERS = [ "input_1" ]

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
    image = image.resize(IMAGE_SHAPE[:2])
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

## Feature extraction
# Build and returns a model that returns features tensors given an input tensor
# that extracts features based on the given layers names
def build_extractor(layer_names):
    with tf.name_scope("feature_extractor"):
        # Load VGG model
        vgg_model = VGG16(input_shape=IMAGE_SHAPE, weights="imagenet", include_top=False)
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
        n_features = tf.cast(tf.reduce_prod([ s.value for s in input_op.shape]),
                             dtype="float32")
        
        # Compute gram matrix for correlations between features
        gram_mat_op = tf.matmul(K.transpose(input_op), input_op)
        return gram_mat_op / n_features

## Loss functions
# Build and return tensor that computes content loss given the 
# pastiche and content image tensors
def build_content_loss(pastiche_op, content_op):
    with tf.name_scope("content_loss"):
        # Shape tensors for feature extraction
        pastiche_op = tf.expand_dims(pastiche_op, axis=0)
        content_op = tf.expand_dims(content_op, axis=0)
        
        # Extract content features using content extractor
        extractor = build_extractor(CONTENT_LAYERS)
        pastiche_feature_ops = extractor(pastiche_op)
        content_feature_ops = extractor(content_op)
        
        # Compute content loss
        loss_op = tf.reduce_mean(tf.squared_difference(pastiche_feature_ops,
                                                       content_feature_ops), 
                                name="content_loss")
    

        # Track content loss with tensorboard
        loss_summary = tf.summary.scalar("content_loss", loss_op)

        return loss_op

# Build and return a tensor that computes style loss given the
# pastiche and content image tensors
# Loss attempts to reduce the noise in the generated images
def build_style_loss(pastiche_op, style_op):
    with tf.name_scope("style_loss"):
        # Shape tensors for feature extraction
        pastiche_op = tf.expand_dims(pastiche_op, axis=0)
        style_op = tf.expand_dims(style_op, axis=0)
        
        # Extract style features using style extractor
        extractor = build_extractor(STYLE_LAYERS)
        pastiche_feature_ops = extractor(pastiche_op)
        style_feature_ops = extractor(style_op)
        
        # Build style loss tensor for each layer
        def build_layer_style_loss(layer_name, pastiche_feature_op, style_feature_op):
            with tf.name_scope("layer_style_loss"):
                # Extract style feawtures using gram matrix
                pastiche_gram_op = build_gram_matrix(pastiche_feature_op)
                style_gram_op = build_gram_matrix(style_feature_op)

                # Compute style loss for layer
                layer_loss_name = layer_name + "_loss"
                layer_loss_op = tf.reduce_sum(tf.squared_difference(pastiche_gram_op,
                                                                    style_gram_op),
                                              name=layer_loss_name)
                
                # Track content loss for layer with tensorboard
                loss_summary = tf.summary.scalar(layer_loss_name, layer_loss_op)
                
                return layer_loss_op

        layer_loss_ops = [ build_layer_style_loss(N, P, S) for N, P, S in 
                          zip(STYLE_LAYERS, pastiche_feature_ops, style_feature_ops) ]
    
        # Compute total style loss accross layers
        loss_op = tf.reduce_sum(layer_loss_ops, name="style_loss")
        # Track content loss with tensorboard
        loss_summary = tf.summary.scalar("style_loss", loss_op)

        return loss_op

# Build and return a tensor that computes total variation loss
# Loss attempts to reduce the noise in the generated images
def build_noise_loss(pastiche_op):
    with tf.name_scope("noise_loss"):
        #TODO: implement multiple layers
        # Compute variation accross image axis
        height_variation_op = tf.reduce_mean(K.abs(pastiche_op[:-1, :, :] - 
                                                   pastiche_op[1:, :, :]))
        width_variation_op = tf.reduce_mean(K.abs(pastiche_op[:, :-1, :] - 
                                                  pastiche_op[:, 1:, :]))
        
        loss_op = tf.add(height_variation_op, width_variation_op, name="total_variation_loss")
        
        # Track total variation with tensorboard
        loss_summary = tf.summary.scalar("total_variation_loss", loss_op)

        return loss_op

# Build and return tensor that total style transfer loss given the
# pastiche, content and style image tensors 
def build_loss(pastiche_op, content_op, style_op):
    with tf.name_scope("style_transfer_loss"):
        content_loss_op = build_content_loss(pastiche_op, content_op)
        style_loss_op = build_style_loss(pastiche_op, style_op)
        noise_loss_op = build_noise_loss(pastiche_op)

        # Total loss weight sum of content and style Losses
        loss_op = tf.add_n([CONTENT_WEIGHT * content_loss_op,
            + STYLE_WEIGHT * style_loss_op,
            + DENOISE_WEIGHT * noise_loss_op], name="style_transfer_loss")

        # Track style transer loss loss with tensorboard
        loss_summary = tf.summary.scalar("style_transfer_loss", loss_op)
        
        return loss_op

def reverse_tensor(pastiche_op):
    blue_op, green_op, red_op = tf.unstack(pastiche_op, axis=-1)

    # Add mean value
    red_op = red_op + 103.939
    green_op = green_op + 116.779
    blue_op = blue_op + 123.68

    return tf.stack([red_op, green_op, blue_op], axis=-1)
    
if __name__ == "__main__":
    import os
    from shutil import rmtree

    content = preprocess_image(Image.open("./data/Tuebingen_Neckarfront.jpg"))
    style = preprocess_image(Image.open("./data/stary_night.jpg"))
    pastiche = content.copy() # Generate pastiche from content

    tf.reset_default_graph()
    session = tf.Session()
    K.set_session(session)

    pastiche_op = K.variable(pastiche, name="pastiche")
    content_op = K.constant(content, name="content")
    style_op = K.constant(style, name="style")
    
    loss_op = build_loss(pastiche_op, content_op, style_op)
    
    optimizer = tf.train.AdamOptimizer(learning_rate=3e+1)
    train_op = optimizer.minimize(loss_op, var_list=[pastiche_op])

    writer = tf.summary.FileWriter("logs/{}".format(datetime.now()), session.graph)
    # Track changes to pastiche with tensorboard
    tf.summary.histogram("pasitche_changes", pastiche_op)
    tf.summary.image("pastiche", tf.expand_dims(reverse_tensor(pastiche_op), axis=0))
    summary_op = tf.summary.merge_all()

    session.run(tf.global_variables_initializer())
    n_epochs = 150
    
    # Setup pastiche directory
    if os.path.exists("pastiche"): rmtree("pastiche")
    os.mkdir("pastiche")

    for i in range(n_epochs):
        feed = {K.learning_phase(): 0}
        _, loss, pastiche, summary = session.run([train_op, loss_op, pastiche_op,
                                                  summary_op], feed_dict=feed)
        print("[{}/{}] loss: {:e}".format(i, n_epochs, loss))
        
        pastiche_image = deprocess_image(pastiche)

        # Plot current pastiche image to show progressa
        plt.imshow(np.asarray(pastiche_image))
        plt.draw()
        plt.pause(1e-9)
        
        # Save pastiche image for laster
        pastiche_image.save("pastiche/{}.jpg".format(i))

        writer.add_summary(summary, i)
