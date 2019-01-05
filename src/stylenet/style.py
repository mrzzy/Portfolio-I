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
IMAGE_SHAPE = (128, 128, 3)

# Loss computation weights
CONTENT_WEIGHT = 0.05
STYLE_WEIGHT = 50
TOTAL_VARIATION_WEIGHT = 0

# Layers for feature extraction
CONTENT_LAYERS = ['input_1']
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
    # Load VGG model
    vgg_model = VGG16(input_shape=IMAGE_SHAPE, weights="imagenet", include_top=False)
    vgg_model.trainable = False

    # Build  model by extracting  feature tensors
    layers = [ vgg_model.get_layer(name) for name in layer_names ]
    feature_ops = [ layer.output for layer in layers ] 
    model = Model(inputs=[vgg_model.input], outputs=feature_ops)

    return model

# Build and return the gram matrix tensor for the given input features tensor
def build_gram_matrix(input_op):
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
    # Shape tensors for feature extraction
    pastiche_op = tf.expand_dims(pastiche_op, axis=0)
    content_op = tf.expand_dims(content_op, axis=0)
    
    # Extract content features using content extractor
    extractor = build_extractor(CONTENT_LAYERS)
    pastiche_feature_op = extractor(pastiche_op)
    content_feature_op = extractor(content_op)
    
    # Compute content loss
    loss = tf.losses.mean_squared_error(pastiche_feature_op, content_feature_op)

    return loss

# Build and return tensor that computes style loss given the 
# pastiche and style image tensors
def build_style_loss(pastiche_op, style_op):
    # Shape tensors for feature extraction
    pastiche_op = tf.expand_dims(pastiche_op, axis=0)
    style_op = tf.expand_dims(style_op, axis=0)

    # Extract style features using style extractor
    extractor = build_extractor(STYLE_LAYERS)
    pastiche_feature_ops = extractor(pastiche_op)
    style_feature_ops = extractor(style_op)
    
    pastiche_gram_ops = [ build_gram_matrix(f) for f in pastiche_feature_ops ]
    style_gram_ops = [ build_gram_matrix(f) for f in style_feature_ops ]
    
    # Compute style loss
    losses = []
    for pastiche_feature_op, style_feature_op in zip(pastiche_feature_ops, 
                                                     style_feature_ops):
        # Define constants
        n_batch, n_height, n_width, n_features = pastiche_feature_op.shape.as_list()
        N = n_features
        M = n_height * n_width

        # Compute gram matrix representations of style features
        pastiche_gram_op = build_gram_matrix(pastiche_feature_op)
        style_gram_op = build_gram_matrix(style_feature_op)
    
        # Compute layer contribution to loss
        loss = tf.losses.mean_squared_error(pastiche_gram_op, 
                                            style_gram_op) / (4 * (N ** 2) * (M ** 2))
        losses.append(loss)
    
    return K.sum(losses)
    

if __name__ == "__main__":
    content = preprocess_image(Image.open("./data/Tuebingen_Neckarfront.jpg"))
    style = preprocess_image(Image.open("./data/stary_night.jpg"))
    pastiche = np.random.uniform(size=IMAGE_SHAPE) * 256
    
    session = tf.Session()
    K.set_session(session)
    pastiche_op = K.variable(pastiche)
    content_op = K.constant(content)
    style_op = K.constant(style)

    #loss_op = build_content_loss(pastiche_op, content_op)
    loss_op = 1e+6 * build_style_loss(pastiche_op, style_op)

    optimizer = tf.train.AdamOptimizer(learning_rate=1e+2)
    train_op = optimizer.minimize(loss_op, var_list=[ pastiche_op ])
    
    session.run(tf.global_variables_initializer())
    
    for i in range(300):
        _, loss, pastiche = session.run([train_op, loss_op, pastiche_op])
        print(i," - loss: ", loss)
    
        img = deprocess_image(pastiche)
        img.save("pastiche/{}.jpg".format(i))
