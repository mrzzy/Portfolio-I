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
from keras.models import Model
from keras.layers import InputLayer
from keras.applications.vgg16 import VGG16

# Style transfer settings
IMAGE_SHAPE = (64, 64, 3)

# Loss computation weights
CONTENT_WEIGHT = 0.025
STYLE_WEIGHT = 5.0
DENOISE_WEIGHT = 1.0

# Layers for feature extraction
CONTENT_LAYERS = ['block2_conv2']
STYLE_LAYERS = ['block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3',
                'block5_conv3']
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
        # Flatten input tensor, leaving features dimention
        n_features = input_op.shape[-1].value
        input_op = K.reshape(input_op, (-1, n_features))
        
        # Compute gram matrix
        gram_mat_op = K.dot(K.transpose(input_op), input_op)
        return gram_mat_op

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
        differences = [ P - C for P, C in zip(pastiche_feature_ops, 
                                              content_feature_ops) ]
                       
        loss = tf.reduce_sum(tf.square(differences), name="content_loss")

        return loss

# Build and return a tensor that computes total variation loss
# Loss attempts to reduce the noise in the generated images
def build_noise_loss(pastiche_op):
    with tf.name_scope("noise_loss"):
        #TODO: implement multiple layers
        # Compute variation accross image axis
        height_variation = K.sum(K.abs(pastiche_op[:-1, :, :] - pastiche_op[1:, :, :]))
        width_variation = K.sum(tf.abs(pastiche_op[:, :-1, :] - pastiche_op[:, 1:, :]))
        
        total_variation = height_variation + width_variation

        return total_variation

# Build and return tensor that total style transfer loss given the
# pastiche, content and style image tensors 
def build_loss(pastiche_op, content_op, style_op):
    with tf.name_scope("style_transfer_loss"):
        content_loss_op = build_content_loss(pastiche_op, content_op)
        style_loss_op = build_style_loss(pastiche_op, style_op)
        noise_loss_op = build_noise_loss(pastiche_op)

        # Total loss weight sum of content and style Losses
        loss = CONTENT_WEIGHT * content_loss_op \
            + STYLE_WEIGHT * style_loss_op \
            + DENOISE_WEIGHT * noise_loss_op

        return loss
    
def build_summary(loss_op):
    loss_summary = tf.summary.scalar("Style Transfer Loss", loss_op)
    summary_op = tf.summary.merge([loss_summary])
    
    return summary_op

if __name__ == "__main__":
    content = preprocess_image(Image.open("./data/Tuebingen_Neckarfront.jpg"))
    style = preprocess_image(Image.open("./data/stary_night.jpg"))
    pastiche = np.random.uniform(size=IMAGE_SHAPE) * 256.0 - 128.0
    
    K.clear_session()
    session = tf.Session()
    K.set_session(session)

    pastiche_op = K.variable(pastiche, name="pastiche")
    content_op = K.constant(content, name="content")
    style_op = K.constant(style, name="style")
    
    loss_op = build_content_loss(pastiche_op, content_op)
    
    optimizer = tf.train.AdamOptimizer(learning_rate=1e+2)
    train_op = optimizer.minimize(loss_op, var_list=[pastiche_op])

    writer = tf.summary.FileWriter("logs", session.graph)
    summary_op = build_summary(loss_op)

    session.run(tf.global_variables_initializer())
    n_epochs = 100
    for i in range(n_epochs):
        feed = {K.learning_phase(): 0}
        _, loss, pastiche, summary = session.run([train_op, loss_op, pastiche_op,
                                                  summary_op],
                                                 feed_dict=feed)
        print("[{}/{}] loss: {:e}".format(i, n_epochs, loss))
        
        pastiche_image = deprocess_image(pastiche)
        plt.imshow(np.asarray(pastiche_image))
        plt.draw()
        plt.pause(1e-9)

        writer.add_summary(summary, i)
