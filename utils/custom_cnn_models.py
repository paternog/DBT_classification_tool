##########################################################################
# Author: Gianfranco Paternò (paterno@fe.infn.it)                        #
# collaboration of INFN Ferrara and INFN Napoli for the next_AIM project #
# Last modification date: March 2024                                     #
#                                                                        #
# SPDX-License-Identifier: EUPL-1.2                                      #
# Copyright 2024 Istituto Nazionale di Fisica Nucleare                   #
##########################################################################


import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Activation, LeakyReLU, Dropout, BatchNormalization, GlobalAveragePooling2D, Rescaling, SeparableConv2D, add
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras import regularizers
from functools import partial
import random

 

def DBT_DCNN(dimInput, classes,
             dln=4096, alpha=0., dropout=0.,
             Sigma=0.02, Lambda=0.01,
             Filters=[96, 128, 384, 192, 128],
             Kernels=[11, 5, 3, 3, 3],
             pool_size=[(3,3), (3,3), (3,3), (3,3)],
             strides=[(1,1), (2,2), (2,2), (2,2), (2,2)]):   

    """       
    ##########################################################################
    ### Generalized version of the CNN defined in Ricciardi et al, 2021,   ###
    ### "A deep learning classifier for digital breast tomosynthesis"      ###
    ##########################################################################   
    """
    
    # weight initialization
    init = RandomNormal(seed=random.randint(1, 100), stddev=Sigma)    
    # define the weight regularizer
    reg = regularizers.L2(l2=Lambda)
    # input layer
    shape = (dimInput[0], dimInput[1], 1)
    layer_in = Input(shape=shape)    
    # the cnn
    # note: I introduced (2,2) strides also in the first Conv2D layer to match the work
    #       done and presented to the next_AIM collaboration meeting of February 2023
    hidden = Conv2D(Filters[0], kernel_size=Kernels[0], padding='same', strides=strides[0], kernel_initializer=init, kernel_regularizer=reg)(layer_in)
    hidden = BatchNormalization(center=False, scale=False)(hidden)
    hidden = LeakyReLU(alpha)(hidden)
    hidden = MaxPooling2D(pool_size=pool_size[0], padding='same', strides=strides[1])(hidden)
    hidden = Conv2D(Filters[1], kernel_size=Kernels[1], padding='same', kernel_initializer=init, kernel_regularizer=reg)(hidden)
    hidden = BatchNormalization(center=False, scale=False)(hidden)
    hidden = LeakyReLU(alpha)(hidden)
    hidden = MaxPooling2D(pool_size=pool_size[1], padding='same', strides=strides[2])(hidden)
    hidden = Conv2D(Filters[2], kernel_size=Kernels[2], padding='same', kernel_initializer=init, kernel_regularizer=reg)(hidden)
    hidden = BatchNormalization(center=False, scale=False)(hidden)
    hidden = LeakyReLU(alpha)(hidden)
    hidden = MaxPooling2D(pool_size=pool_size[2], padding='same', strides=strides[3])(hidden)
    hidden = Conv2D(Filters[3], kernel_size=Kernels[3], padding='same', kernel_initializer=init, kernel_regularizer=reg)(hidden)
    hidden = BatchNormalization(center=False, scale=False)(hidden)
    hidden = LeakyReLU(alpha)(hidden)
    hidden = MaxPooling2D(pool_size=pool_size[3], padding='same', strides=strides[4])(hidden)
    hidden = Conv2D(Filters[4], kernel_size=Kernels[4], padding='same', kernel_initializer=init, kernel_regularizer=reg)(hidden)
    hidden = BatchNormalization(center=False, scale=False)(hidden)
    hidden = LeakyReLU(alpha)(hidden)
    hidden = Flatten()(hidden)    
    hidden = Dense(dln, kernel_regularizer=reg)(hidden)    
    hidden = Dropout(dropout)(hidden)  
    # output layer
    hidden = Dense(len(classes))(hidden)
    layer_out = Activation('softmax')(hidden)    
    # the model
    model = Model(inputs=layer_in, outputs=layer_out)    
    return model



##########################################################################
### Custom implementation of a Darknet19 model (as a set of functions) ###
##########################################################################
new_conv = partial(Conv2D, padding="same")

def _base_block(out, x):
    "(3,3), Leaky, Batch"
    x = new_conv(out, (3,3))(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization()(x) 
    return x

def _block_1(out, x):     
    #output follows:
    #out//2, out
    x = new_conv(out//2, (1,1))(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization()(x)
    x = _base_block(out,x)
    return x

def _block_2(out, x):    
    #output follows:
    #out, out//2, out
    x = _base_block(out, x)
    x = _block_1(out, x)
    return x

def Darknet19(dimInput):
    input_layer = Input((dimInput[0], dimInput[1], 1))
    x = _base_block(32, input_layer)
    x = MaxPooling2D((2,2), strides=2)(x)
    x = _base_block(64, x)
    x = MaxPooling2D((2,2), strides=2)(x)
    x = _block_2(128, x)
    x = MaxPooling2D((2,2), strides=2)(x)
    x = _block_2(256, x)
    x = MaxPooling2D((2,2), strides=2)(x)
    x = _block_2(512, x)
    x = _block_1(512, x)
    x = MaxPooling2D((2,2), strides=2)(x)
    x = _block_2(1024, x)
    x = _block_1(512, x)
    x = new_conv(1, (1,1), activation="linear")(x)
    model = Model(inputs=input_layer, outputs=x)  
    return model

def Darknet19_classifier(dimInput, classes):
    base_model = Darknet19(dimInput)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(len(classes))(x)
    output = Activation('softmax')(x)
    model = Model(inputs=base_model.inputs, outputs=output)
    return model



def CNN1(dimInput, classes,
         dln=5, alpha=0.2, dropout=0.1,
         Filters=[6, 6, 6, 12, 12, 12],
         Kernels=[3, 3, 3, 3, 3, 3]):   
    
    """
    ##########################################################################
    ### A simple CNN (inspired to the one developed for classifying        ###
    ### pathches of images with microcalc during the ML_hackathon_2)       ###
    ##########################################################################    
    """

    # weight initialization
    init = RandomNormal(stddev=0.02)
    # layers
    shape = (dimInput[0], dimInput[1], 1)
    in_layer = Input(shape=shape)
    hidden = Conv2D(Filters[0], Kernels[0], padding='same', kernel_initializer=init)(in_layer)
    hidden = Conv2D(Filters[1], Kernels[1], padding='same', kernel_initializer=init)(hidden)
    hidden = Conv2D(Filters[2], Kernels[2], padding='same', kernel_initializer=init)(hidden)
    hidden = BatchNormalization()(hidden)
    hidden = Activation(LeakyReLU(alpha))(hidden)
    hidden = MaxPooling2D((2,2))(hidden)
    hidden = Conv2D(Filters[3], Kernels[3], padding='same', kernel_initializer=init)(hidden)
    hidden = Conv2D(Filters[4], Kernels[4], padding='same', kernel_initializer=init)(hidden)
    hidden = Conv2D(Filters[5], Kernels[5], padding='same', kernel_initializer=init)(hidden)
    hidden = BatchNormalization()(hidden)
    hidden = Activation(LeakyReLU(alpha))(hidden)
    hidden = MaxPooling2D((2,2))(hidden)
    hidden = Flatten()(hidden)
    hidden = Dense(dln, activation=LeakyReLU(alpha))(hidden)
    hidden = Dropout(dropout)(hidden)
    out_layer = Dense(len(classes), activation='softmax')(hidden)
    # the model
    model = Model(inputs=in_layer, outputs=out_layer)
    return model



def CNN2(input_shape, num_classes):
    
    """
    ##########################################################################
    ### Another CNN: taken from the example "image classifier from scratch ###
    ### on the Kaggle Cats vs Dogs dataset".                               ###
    ##########################################################################
    """
    
    inputs = Input(shape=input_shape)

    # Entry block
    x = Rescaling(1.0 / 255)(inputs)
    x = Conv2D(32, 3, strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(64, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [128, 256, 512, 728]:
        x = Activation("relu")(x)
        x = SeparableConv2D(size, 3, padding="same")(x)
        x = BatchNormalization()(x)

        x = Activation("relu")(x)
        x = SeparableConv2D(size, 3, padding="same")(x)
        x = BatchNormalization()(x)

        x = MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = SeparableConv2D(1024, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = GlobalAveragePooling2D()(x)
    
    activation = "softmax"
    units = num_classes

    x = Dropout(0.5)(x)
    outputs = Dense(units, activation=activation)(x)
    return Model(inputs, outputs)



def AlexNet(input_shape, num_classes):    
    """
    ##########################################################################
    ### AlexNet (https://towardsdatascience.com/implementing-alexnet-cnn-  ###
    ### architecture-using-tensorflow-2-0-and-keras-2113e090ad98)          ###
    ##########################################################################    
    """
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=input_shape),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
        tf.keras.layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
        tf.keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])    
    return model
