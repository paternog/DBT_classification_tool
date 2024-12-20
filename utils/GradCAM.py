##########################################################################
# Author: Gianfranco Paternò (paterno@fe.infn.it)                        #
# collaboration of INFN Ferrara and INFN Napoli for the next_AIM project #
# Last modification date: March 2024                                     #
#                                                                        #
# SPDX-License-Identifier: EUPL-1.2                                      #
# Copyright 2024 Istituto Nazionale di Fisica Nucleare                   #
##########################################################################


# Create a superimposed GradCAM visualization for a set of images


import numpy as np
import pandas as pd
import os
from glob import glob
import platform
import random
import cv2
import matplotlib.pyplot as plt
import seaborn as sn
from skimage.io import imread
from skimage.transform import resize
from PIL import Image
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras.models import Model
from tqdm import tqdm



def rgb2gray(rgb):
    """
    Function to convert an image (3D numpy array into a RGB image)
    """
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])



def VizGradCAM(model, image, interpolant=0.5, plot_results=True, self_norm=True, norm_fact=1.0):
    """VizGradCAM - Displays GradCAM based on Keras / TensorFlow models
    using the gradients from the last convolutional layer. This function
    should work with all Keras Application listed here:
    https://keras.io/api/applications/
    
    Parameters:
    model (keras.model): Compiled Model with Weights Loaded
    image: Image to Perform Inference On
    plot_results (boolean): True - Function Plots using PLT
                            False - Returns Heatmap Array
    Returns:
    Heatmap Tensor
    
    gpaterno: normalization of Heatmap slightly modified.
    note: It works on a single image.
    """
    
    # Sanity Check
    assert (
        interpolant > 0 and interpolant < 1
    ), "Heatmap Interpolation Must Be Between 0 - 1"

    # Get the last Conv2D layer in the net
    last_conv_layer = next(
        x for x in model.layers[::-1] if isinstance(x, K.layers.Conv2D)
    )
    
    target_layer = model.get_layer(last_conv_layer.name)

    original_img = image
    img = np.expand_dims(original_img, axis=0)
    prediction = model.predict(img, verbose=0)

    # Obtain Prediction Index
    prediction_idx = np.argmax(prediction)

    # Compute Gradient of Top Predicted Class
    with tf.GradientTape() as tape:
        gradient_model = Model([model.inputs], [target_layer.output, model.output])
        conv2d_out, prediction = gradient_model(img)
        # Obtain the Prediction Loss
        loss = prediction[:, prediction_idx]

    # Gradient() computes the gradient using operations recorded
    # in context of this tape
    gradients = tape.gradient(loss, conv2d_out)

    # Obtain the Output from Shape [1 x H x W x CHANNEL] -> [H x W x CHANNEL]
    output = conv2d_out[0]

    # Obtain Depthwise Mean
    weights = tf.reduce_mean(gradients[0], axis=(0, 1))

    # Create a 7x7 Map for Aggregation
    activation_map = np.zeros(output.shape[0:2], dtype=np.float32)

    # Multiply Weights with Every Layer
    for idx, weight in enumerate(weights):
        activation_map += weight * output[:, :, idx]

    # Resize to Size of Image
    activation_map = cv2.resize(
        activation_map.numpy(), (original_img.shape[1], original_img.shape[0])
    )

    # Ensure No Negative Numbers
    activation_map = np.maximum(activation_map, 0)

    # Convert Class Activation Map to 0 - 255
    if self_norm:
        activation_map = (activation_map - activation_map.min()) / (activation_map.max() - activation_map.min())
    else:
        activation_map = activation_map / norm_fact 
    activation_map_8bit = np.uint8(255 * activation_map)

    # Convert to Heatmap
    heatmap = cv2.applyColorMap(activation_map_8bit, cv2.COLORMAP_JET)
    cvt_heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Superimpose Heatmap on Image Data
    original_img = np.uint8( (original_img - original_img.min()) / (original_img.max() - original_img.min()) * 255)    

    plt.rcParams["figure.dpi"] = 100 # Enlarge Plot
    
    if plot_results == True:
        plt.imshow( np.uint8(original_img * interpolant + cvt_heatmap * (1 - interpolant)) ) 
        plt.xlabel("x (pixel)")
        plt.ylabel("y (pixel)")
    
    return cvt_heatmap, activation_map

    

def plot_and_save_GradCAM(model, img_name, GradCAMsDir, x, sn=True, nf=0.0015, 
                          saveGradcam2Text=False, close_plots=True, MPLBACKEND=''):
    """
    Wrapper function to plot and save the GradCAM for an image.
    """
    print('calculating the GradCAM of image %s' % (img_name))
    plt.figure(figsize=(10, 10))
    plt.subplot(1,2,1)
    plt.imshow(x, cmap='bone')
    plt.xlabel("x (pixel)")
    plt.ylabel("y (pixel)")
    plt.subplot(1,2,2)
    gradcam, activation_map = VizGradCAM(model,
                                         x,
                                         interpolant=0.5,
                                         self_norm=sn,
                                         norm_fact=nf)
    plt.savefig(GradCAMsDir + 'GradCAM_' + img_name + '.jpg')
    if not close_plots and not MPLBACKEND == 'Agg':
        plt.show()
    plt.close()    
    if saveGradcam2Text:
        fname = GradCAMsDir + 'GradCAM_' + img_name + '.dat'
        #np.savetxt(fname, np.uint8(rgb2gray(gradcam))) #convert GradCAM in grayscale -> it does not work well
        #np.savetxt(fname, activation_map, fmt='%d', delimiter=' ') #used if I want to export activation_map_8bit
        np.savetxt(fname, activation_map, fmt='%.8f', delimiter=' ') # normalized between 0 and 1    
    return gradcam



def GradCAM_calculation(dataset_path, dataset_structure, classes, img_type, 
                        model, x, y, img_name_list, GradCAMsDir, 
                        saveGradcam2Text=False, close_plots=False, sn=True, nf=0.0015,
                        type_choice=2, Nfigs2plot=10, random_choice=True, 
                        MPLBACKEND=''):
    
    """
    Function to plot the GradCAM for a set of images of a given dataset.
    type_choice: 0->positive, 1->predicted positive, 2->true postive, 
                 3->negative predicted as postive, 4->benign predicted as postive, 
                 5->any
    """
    
    # variables to plot   
    model_preds = model.predict(x, batch_size=1, verbose=0)
    preds2plot = np.argmax(model_preds, axis=-1)
    if type(y) == float:
        labels2plot = 3 * np.ones(len(preds2plot), dtype=int)
        type_choice = 5
    else:
        labels2plot = np.argmax(y, axis=-1)
         
    # number of classes and images in the dataset
    if type(y) == float:
        n_classes = 1
    else:
        n_classes = len(classes)
    n_images = len(model_preds)
    
    """
    # list of images in the dataset (now, it is passed as a paramter!)
    img_name_list = []
    for j in range(n_classes):
        if type(y) == float:
            fnames = glob(os.path.join(dataset_path + dataset_structure[-1] + '/*' + img_type))
        else:
            fnames = glob(dataset_path + dataset_structure[-1] + str(classes[j][0].upper() + classes[j][1:]) + '/*' + img_type)
            if len(fnames) == 0:
                fnames = glob(dataset_path + dataset_structure[-1] + str(classes[j][0].lower() + classes[j][1:]) + '/*' + img_type)
        for fname in fnames:
            #if platform.system() == 'Windows': 
            #    item = fname.split('\\')[1].split(img_type)[0]
            #else:
            #    item = fname.split(str(classes[j]) + '/')[1].split(img_type)[0]
            item = os.path.basename(fname).split('.')[0]
            img_name_list.append(item)
    #print(img_name_list)
    """
    
    # define a dictionary for labels and predictions
    if n_classes == 2:
        mydict = {'0': 'neg', '1': 'pos', '3': 'unknown'}
    else:
        mydict = {'0': 'neg', '1': 'pos', '2': 'ben', '3': 'unknown'}
    
    # list of images to plot
    plot_list = [] 
    if type_choice == 0:
        print("calculating the GradCAM for actual positive images")
        for j in range(n_images):
            if (labels2plot[j] == 1):
                plot_list.append(j)
    if type_choice == 1:
        print("calculating the GradCAM for predicted positive images")
        for j in range(n_images):
            if (preds2plot[j] == 1):
                plot_list.append(j)
    if type_choice == 2:
        print("calculating the GradCAM for actual and predicted positive images")
        for j in range(n_images):
            if (labels2plot[j] == 1 and preds2plot[j] == 1):
                plot_list.append(j)   
    if type_choice == 3:
        print("calculating the GradCAM for actual negative images predicted as positive")
        for j in range(n_images):
            if (labels2plot[j] == 0 and preds2plot[j] == 1):
                plot_list.append(j)
    if type_choice == 4:
        print("calculating the GradCAM for actual benigne images predicted as positive")
        for j in range(n_images):
            if (labels2plot[j] == 2 and preds2plot[j] == 1):
                plot_list.append(j)
    if type_choice > 4:
        print("calculating the GradCAM for all the types of images of the dataset")
        for j in range(n_images):
            plot_list.append(j)
    
    Nfigs2plot = min(Nfigs2plot, len(plot_list))
    print("potential number of figures to plot: %d" % len(plot_list))   
    print("number of figures to plot: %d" % Nfigs2plot)

    if random_choice:
        plot_list = random.choices(plot_list, k=Nfigs2plot)
        print("images (indices) randomly chosen")
    else:
        plot_list = plot_list[:Nfigs2plot]
    print("plot_list (image indices): ", plot_list)

    x2plot = x[plot_list]
    labels2plot = labels2plot[plot_list]
    preds2plot = preds2plot[plot_list]
    img_name_list = [img_name_list[ii] for ii in plot_list]
    
    # calculate the GradCAM for the images of the dataset
    gradcam_tensor = np.zeros((len(img_name_list), x.shape[1], x.shape[2], 3))
    img_name_list_full = []
    for i, item in enumerate(img_name_list):
        img_name = item + "_label_" + mydict[str(labels2plot[i])] + "_pred_" + mydict[str(preds2plot[i])]
        img_name_list_full.append(img_name)
        gradcam = plot_and_save_GradCAM(model, img_name, GradCAMsDir, x2plot[i], sn, nf, saveGradcam2Text, close_plots, MPLBACKEND)
        gradcam_tensor[i] = gradcam
    if close_plots:
        print("GradCAM plots have been closed")
    
    # return a (numpy) tensor with the calculated GradCAMs
    return gradcam_tensor, img_name_list_full



def GradCAM_calculation_all(dataset_path, dataset_structure, classes, img_type, 
                            model, x, y, img_name_list, 
                            GradCAMsDir, saveGradcam2Text=False, 
                            sn=True, nf=0.0015, MPLBACKEND=''):
    
    """
    Function to calculate (and show) the GradCAM for all the images of a given dataset.
    """
    
    # variables to plot   
    model_preds = model.predict(x, batch_size=1, verbose=0)
    preds2plot = np.argmax(model_preds, axis=-1)    
    if type(y) == float:
        labels2plot = 3 * np.ones(len(preds2plot), dtype=int)
        type_choice = 5
    else:
        labels2plot = np.argmax(y, axis=-1)
         
    # number of classes and images in the dataset
    if type(y) == float:
        n_classes = 1
    else:
        n_classes = len(classes)
    n_images = len(model_preds)
                
    # define a dictionary for labels and predictions
    if n_classes == 2:
        mydict = {'0': 'neg', '1': 'pos', '3': 'unknown'}
    else:
        mydict = {'0': 'neg', '1': 'pos', '2': 'ben', '3': 'unknown'}  
    
    # calculate the GradCAM for the images of the dataset
    gradcam_tensor = np.zeros((len(img_name_list), x.shape[1], x.shape[2], 3))
    img_name_list_full = []
    for i, item in enumerate(img_name_list):
        img_name = item + "_label_" + mydict[str(labels2plot[i])] + "_pred_" + mydict[str(preds2plot[i])]
        img_name_list_full.append(img_name)
        gradcam = plot_and_save_GradCAM(model, img_name, GradCAMsDir, x[i], sn, nf, saveGradcam2Text, False, MPLBACKEND)
        gradcam_tensor[i] = gradcam
    print("GradCAM plots have been closed")
    
    # return a (numpy) tensor with the calculated GradCAMs
    return gradcam_tensor, img_name_list_full
