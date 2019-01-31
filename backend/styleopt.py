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
    "n_epochs": 100,
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
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(
            self.loss_op, method='L-BFGS-B', options={'maxfun': 20}, 
            var_list=[self.pastiche_op])
        
        # Setup tensorboard
        self.summary_op = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter("./logs/{}-{}".format(
            self.settings, datetime.now().strftime("%H:%M:%S")))

        self.session = K.get_session()
        
    # Perform one iteration of style transfer using the inputs in feed dic
    def transfer(self, feed):
        # Perform training setup
        self.optimizer.minimize(self.session, feed_dict=feed)
        
# Callback for writing tensorboard infomation given transfuse graph and current
# epoch number i_epoch and feed dict to run the graph
def callback_tensorboard(graph, feed, i_epoch):
    summary = graph.session.run(graph.summary_op, feed_dict=feed)
    graph.writer.add_summary(summary, i_epoch)

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

    # Define limits for generated pastiche 
    min_limits = - stylefn.IMG_BGR_MEAN
    max_limits = 255.0 - stylefn.IMG_BGR_MEAN

    # Build style transfer graph
    #pastiche_init = np.random.uniform(size=image_shape) * 255.0 - 127.5
    pastiche_init = content
    graph = TransfuseGraph(pastiche_init=pastiche_init, settings=settings)
    session = graph.session
    session.run(tf.global_variables_initializer())
    
    # Optimise style transfer graph to perform style transfer
    feed = {graph.content_op: content, graph.style_op: style}
    n_epochs = settings["n_epochs"] 
    for i_epoch in range(1, n_epochs + 1):
        # Clip the pastiche to ensure values say within limits
        clipped_pastiche_op = tf.clip_by_value(graph.pastiche_op, 
                                               min_limits, max_limits)
        graph.pastiche_op.assign(clipped_pastiche_op)
        
        # Perform style transfer
        graph.transfer(feed)
    
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
        "image_shape": (32, 32, 3),
        "n_epochs": 100
    }

    pastiche_image = transfer_style(content_image, style_image, settings=settings,
                                    callbacks=[callback_pastiche, callback_progress, 
                                               callback_tensorboard],
                                    callback_step=20)
    
    pastiche_image.save("pastiche.jpg")
