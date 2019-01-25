#
# styleopt.py
# Artistic Style Transfer 
# Optimisation method
# as defined in Gatys et. al
#

import os
import api
import numpy as np
import tensorflow as tf
import keras.backend as K
import stylefn

from PIL import Image
from keras.models import Model, Sequential
from shutil import rmtree
from tensorflow.contrib.opt import ScipyOptimizerInterface
from datetime import datetime


# Style transfer settings
# NOTE: the following are default settings and may be overriden
SETTINGS = {
    "image_shape": (512, 512, 3),

    # Loss computation weights
    "content_weight": 1,
    "style_weight": 1e+3,
    "denoise_weight": 1.0,

    # Layers for feature extraction
    "content_layers": ['block2_conv2'],
    "style_layers": ['block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3', 'block5_conv3'],
    "denoising_layers": [ "input_1" ],
    
    "learning_rate": 10,
    "n_epochs": 100
}


# Represents the computational graph that will perform style transfer using the 
# optimisation method
class TransfuseGraph:
    # Create a style transfer graph that caters the style and content images shapes
    # with the given style transfer settings (overrides
    def __init__(self, settings):
        self.settings = settings
        # Define tensor shapes
        self.style_shape = self.settings["image_shape"]
        self.content_shape = self.settings["image_shape"]
        self.pastiche_shape = self.settings["image_shape"]
        
        self.build()
    
    # Build style transfer graph
    def build(self):
        K.clear_session()

        # Setup content and style tensors
        self.content_op = K.placeholder(self.content_shape, name="content")
        self.style_op = K.placeholder(self.style_shape, name="style")
        
        # Setup pastiche tensor derieved from random noise
        pastiche = np.random.uniform(size=self.pastiche_shape, low=-128, high=128)
        self.pastiche_op = K.variable(pastiche, name="pastiche")

        # Build style transfer graph
        self.loss_op = stylefn.build_loss(self.pastiche_op, self.content_op, 
                                          self.style_op, self.settings)
    
        # Setup optimisation
        optimizer = tf.train.AdamOptimizer(learning_rate=self.settings["learning_rate"])
        self.train_op = optimizer.minimize(self.loss_op, var_list=[self.pastiche_op])
    

# Perform style transfer using the optimisation method on the given content image 
# using the style from the given style image. 
# Optimise for the given number epochs. If verbose is True, will output training
# progress infomation to standard output
# Applys the given style transfer settings before performing style transfer
# Returns the pastiche, the results of performing style transfer
def transfer_style(content_image, style_image, settings={}, verbose=False, 
                   tensorboard=False):

    #TODO: add progressive style transfer callback
    if verbose:
        print("[transfer_style()]: performing style transfer with settings: ", 
              stylefn.SETTINGS)
    
    # Preprocess image data
    image_shape = settings["image_shape"]
    content = stylefn.preprocess_image(content_image, image_shape)
    style = stylefn.preprocess_image(style_image, image_shape)

    # Build style transfer graph
    graph = TransfuseGraph(settings)
    session = K.get_session()
    
    # Setup tensorboard
    summary_op = tf.summary.merge_all()
    if tensorboard: 
        writer = tf.summary.FileWriter("logs/{}".format(datetime.now()), session.graph)

    # Optimise style transfer graph to perform style transfer
    session.run(tf.global_variables_initializer())
    n_epochs = settings["n_epochs"] 
    for i in range(1, n_epochs + 1):
        feed = {graph.content_op: content, graph.style_op: style}
        _, loss, pastiche = session.run([graph.train_op, graph.loss_op, 
                                         graph.pastiche_op], feed_dict=feed)

        # Display progress infomation and record data for tensorboard
        if verbose: print("[{}/{}] loss: {:e}".format(i, n_epochs, loss))
        if tensorboard: 
            summary = session.run(summary_op)
            writer.add_summary(summary, i)

    # Deprocess style transfered image
    pastiche_image = stylefn.deprocess_image(pastiche, image_shape)
    return pastiche_image

if __name__ == "__main__":
    content_image = Image.open("data/Tuebingen_Neckarfront.jpg")
    style_image = Image.open("data/stary_night.jpg")

    pastiche_image = transfer_style(content_image, style_image, settings=SETTINGS,
                                    verbose=True)
    
    pastiche_image.save("pastiche.jpg")
