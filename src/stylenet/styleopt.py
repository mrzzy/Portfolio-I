#
# styleopt.py
# Artistic Style Transfer 
# Optimisation method
# as defined in Gatys et. al
#

import os
import numpy as np
import tensorflow as tf
import keras.backend as K
import style

from PIL import Image
from keras.models import Model, Sequential
from shutil import rmtree
from tensorflow.contrib.opt import ScipyOptimizerInterface

if __name__ == "__main__":
    # Load and prerprocess data
    content_image = Image.open("./data/Tuebingen_Neckarfront.jpg")
    style_image = Image.open("./data/stary_night.jpg")
    
    content_mat = style.preprocess_image(content_image)
    style_mat = style.preprocess_image(style_image)
    pastiche_mat = np.random.uniform(size=style.IMAGE_SHAPE)
    
    content_op = K.constant(content_mat)
    style_op = K.constant(style_mat)
    pastiche_op = K.variable(pastiche_mat)
    
    # Build style transfer loss
    loss_op = style.build_loss(pastiche_op, style_op, content_op)
    
    # Perform optimisation for style transfer
    #optimizer = tf.train.AdamOptimizer(learning_rate=1e+1)
    optimizer = ScipyOptimizerInterface(loss_op, options={'maxfun': 20}, 
                                        method='L-BFGS-B', var_list=[pastiche_op])
    #train_op = optimizer.minimize(loss_op, var_list=[pastiche_op])
    
    if os.path.exists("pastiche"): rmtree("pastiche")
    os.mkdir("pastiche")
    with tf.Session() as sess:
        # Init variables
        sess.run(tf.global_variables_initializer())

        for i_epoch in range(300):
            print("epoch ", i_epoch, "...")
            #sess.run(train_op)
            optimizer.minimize(sess)
        
            print("loss: ", sess.run(loss_op))
            pastiche_mat = sess.run(pastiche_op)
            pastiche_image = style.deprocess_image(pastiche_mat)
            pastiche_image.save("pastiche/{}.jpg".format(i_epoch))
