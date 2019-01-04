#
# stylenet.py
# Stylenet
# Feed Forward Style Transfer Network based on Johnson et al.
#

import keras.backend as K

from style import build_loss
from keras.layers import Conv2D, Conv2DTranspose, Activation, Add, Input
from keras import Model

## Blocks
# Defines and returns an convolution encoder block
# specified by given filter_size, n_filter, strides and padding 
def encoder_block(n_filter, filter_size=(3, 3), strides=(1,1), padding="same"):
    def layer(x):
        z = Conv2D(n_filter, filter_size, strides=strides, padding=padding)(x)
        a = Activation("relu")(z)
        return a

    return layer

# Defines and returns an convolution residual block
# specified by given filter_size, n_filter, strides and padding 
def residual_block(n_filter, filter_size=(3,3), strides=(1, 1), padding="same"):
    def layer(x):
        # Add convolution layers
        z = Conv2D(n_filter, filter_size, strides=strides, padding=padding)(x)
        a = Activation("relu")(z)

        z = Conv2D(n_filter, filter_size, strides=strides, padding=padding)(x)
        a = Activation("relu")(z)
        
        # Residual fit: F(x) + x
        y = Add()([a, x])
        return y

    return layer

# Defines and returns an convolution decoder block
# specified by given filter_size, n_filter, strides and padding 
def decoder_block(n_filter, filter_size=(3, 3), strides=(1,1), padding="same"):
    def layer(x):
        z = Conv2DTranspose(n_filter, filter_size, strides=strides, padding="same")(x)
        a = Activation("relu")(z)
        return a

    return layer

## Model
# Build and return a feedfoward model to perform style transfer for the given
# input tensor input_op
def build_model(input_op):
    # Add decoder
    a = encoder_block(32, filter_size=(9, 9), strides=(1, 1))(input_op)
    a = encoder_block(64, filter_size=(3, 3), strides=(2, 2))(a)
    a = encoder_block(128, filter_size=(3, 3), strides=(2, 2))(a)
    
    # Add residual part of model
    a = residual_block(128)(a)
    a = residual_block(128)(a)
    a = residual_block(128)(a)
    a = residual_block(128)(a)
    a = residual_block(128)(a)
    
    # Add decoder
    a = decoder_block(128, filter_size=(3,3), strides=(1, 1))(a)
    a = decoder_block(64, filter_size=(3,3), strides=(2, 2))(a)
    a = decoder_block(32, filter_size=(3,3), strides=(2, 2))(a)
    model = Model(inputs=input_op, outputs=a)
    return model

if __name__ == "__main__":
    input_op = Input((32, 32 ,3), dtype="float32")
    model = build_model(input_op)
    model.summary()
