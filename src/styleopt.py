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
import matplotlib.pyplot as plt
import stylefn

from PIL import Image
from keras.models import Model, Sequential
from util import apply_settings
from tensorflow.contrib.opt import ScipyOptimizerInterface
from datetime import datetime


# Style transfer settings
# NOTE: the following are default settings and may be overriden
SETTINGS = {
    "image_shape": (512, 512, 3),

    # Optimisation settings
    "learning_rate": 10,
    "n_epochs": 1000,
}

# Represents the computational graph that will perform style transfer using the 
# optimisation method
class TransfuseGraph:
    # Create a style transfer graph that caters the style and content images shapes
    # with the given style transfer settings overrides & pastiche init value
    def __init__(self, pastiche_init, settings):
        self.settings = settings
        # Define tensor shapes
        self.style_shape = self.settings["image_shape"]
        self.content_shape = self.settings["image_shape"]
        self.pastiche_shape = self.settings["image_shape"]
        
        self.build(pastiche_init)
    
    # Build style transfer graph for the given pastiche_init value
    def build(self, pastiche_init):
        K.clear_session()

        # Setup content and style tensors
        self.content_op = K.placeholder(self.content_shape, name="content")
        self.style_op = K.placeholder(self.style_shape, name="style")
        
        # Setup pastiche tensor derieved from random noise
        self.pastiche_op = K.variable(pastiche_init, name="pastiche")

        # Build style transfer graph
        self.loss_op = stylefn.build_loss(self.pastiche_op, self.content_op, 
                                          self.style_op, self.settings)
    
        # Setup optimisation
        # Adam hyperparameters borrowed from jcjohnson/neural-style
        optimizer = tf.train.AdamOptimizer(learning_rate=self.settings["learning_rate"],
                                           beta1=0.99, epsilon = 1e-1)
        self.train_op = optimizer.minimize(self.loss_op, var_list=[self.pastiche_op])

        self.session = K.get_session()

# Callback for writing tensorboard infomation given transfuse graph and current
# epoch number i_epoch and feed dict to run the graph
def callback_tensorboard(graph, feed, i_epoch):
    summary = graph.session.run(graph.summary_op, feed_dict=feed)
    writer.add_summary(summary, i)

# Callback for display progress infomation given transfuse graph and current
# epoch number i_epoch and feed dict to run the graph
def callback_progress(graph, feed, i_epoch):
    loss = graph.session.run(graph.loss_op, feed_dict=feed)
    print("[{}/{}] loss: {:e}".format(i_epoch, graph.settings["n_epochs"], loss))

    
# Callback to display current pastiche given transfuse graph and current
# epoch number i_epoch and feed dict to run the graph
def callback_pastiche(graph, feed, i_epoch):
    pastiche = graph.session.run(graph.pastiche_op, feed_dict=feed)
    pastiche_image = stylefn.deprocess_image(pastiche, graph.pastiche_shape)
    
    # Display image as a plot
    plt.imshow(np.asarray(pastiche_image))
    plt.draw()
    plt.pause(1e-6)
    plt.clf()

# Perform style transfer using the optimisation method on the given content imag
# using the style from the given style image, parameterised by settings
# Applys the given style transfer settings before performing style transfer
# Every callback_step number of epochs, will call the given callbacks
# Returns the pastiche, the results of performing style transfer
def transfer_style(content_image, style_image, settings={}, callbacks=[], callback_step=1):
    # Apply setting overrides
    settings = apply_settings(settings, SETTINGS)
    print(settings)

    # Preprocess image data
    image_shape = settings["image_shape"]
    content = stylefn.preprocess_image(content_image, image_shape)
    style = stylefn.preprocess_image(style_image, image_shape)

    # Build style transfer graph
    pastiche_init = content
    graph = TransfuseGraph(pastiche_init=pastiche_init, settings=settings)
    session = graph.session
    session.run(tf.global_variables_initializer())
    
    # Optimise style transfer graph to perform style transfer
    feed = {graph.content_op: content, graph.style_op: style}
    n_epochs = settings["n_epochs"] 
    for i_epoch in range(1, n_epochs + 1):
        # Perform training setup
        session.run(graph.train_op, feed_dict=feed)
    
        # Call callbacks
        if i_epoch % callback_step == 0:
            for callback in callbacks: callback(graph, feed, i_epoch)

    # Deprocess style transfered image
    pastiche = session.run(graph.pastiche_op, feed_dict=feed)
    pastiche_image = stylefn.deprocess_image(pastiche, image_shape)
    
    return pastiche_image

if __name__ == "__main__":
    content_image = Image.open("data/Tuebingen_Neckarfront.jpg")
    style_image = Image.open("data/stary_night.jpg")

    settings = {
        "image_shape": (32, 32, 3)
    }
    pastiche_image = transfer_style(content_image, style_image, settings=settings,
                                    callbacks=[callback_pastiche, callback_progress])
    
    pastiche_image.save("pastiche.jpg")
