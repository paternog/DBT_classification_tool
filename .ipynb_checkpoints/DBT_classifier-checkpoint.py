#!/usr/bin/env python
# coding: utf-8


##########################################################################
# Author: Gianfranco Paternò (paterno@fe.infn.it)                        #
# collaboration of INFN Ferrara and INFN Napoli for the next_AIM project #
# Last modification date: March 2024                                     #
#                                                                        #
# SPDX-License-Identifier: EUPL-1.2                                      #
# Copyright 2024 Istituto Nazionale di Fisica Nucleare                   #
##########################################################################


# DBT classifier


## Import the required libraries
#####################################################################################################################
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sn

from skimage.io import imread
from skimage.transform import resize
from PIL import Image
import PIL
import os
import shutil
from glob import glob
import platform
import subprocess
import random
from tqdm import tqdm
import pydicom
import json
import ast
import re

import tensorflow as tf
from tensorflow.keras import initializers, regularizers, constraints, optimizers, metrics
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Input, Dense, Conv2D, Conv2DTranspose, MaxPooling2D, Flatten, Activation, \
                                    Concatenate, LeakyReLU, Dropout, BatchNormalization, GlobalAveragePooling2D, \
                                    Lambda, Reshape, Layer, InputSpec, Rescaling, SeparableConv2D, add
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.utils import Sequence, to_categorical

from functools import partial
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.utils import class_weight
from livelossplot import PlotLossesKerasTF, PlotLosses
from tensorflow.keras.utils import plot_model
import graphviz

from utils import *

import tensorflow.keras.backend
tensorflow.keras.backend.clear_session()

#os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
#os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
#tf.get_logger().setLevel('ERROR')
#tf.config.run_functions_eagerly(True)
#tf.data.experimental.enable_debug_mode()
print("\nTensorflow version:", tf.__version__) 

os.environ['MPLBACKEND'] = 'Agg' 
MPLBACKEND = os.environ['MPLBACKEND'] 
print('MPLBACKEND:', MPLBACKEND, '\n')

import warnings
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
#####################################################################################################################

## Input
#####################################################################################################################
# Define the data structure
dataset_path = '/media/paterno/SSD1TB/'
dataset_structure = ['DUKE_fully_processed_sample_dataset_all6PcTest_train/', \
                     'DUKE_fully_processed_sample_dataset_all6PcTest_test/']
classes = ('negative', 'positive')
#classes = ('negative', 'positive', 'benign')

# Image features
img_type = '.tiff' 
get_img_type_from_data = False #retrieve img_type from the data (pay attention to filenames with dots)
grayLevels = 65535 #2^Nbit-1

# Input size of the images to feed the CNN with (it is a hyperparameter)
dimInput = (300, 300) #pixel

# Standardization and Shuffling of images
shuffle_imgs = True 
standardize_images = True #img = (img - mean) / std

# Filtering of images
filterImg = False #with a Gaussian filter of sigmaFilter pixels (deaulft is False)
sigmaFilter = 2 #default is 2

# Data augmentation options
augment_data = True
aug_factor = 1.
Ntrasf = 3

# Max number of images to handle (to avoid memory issues)
limit_images = True
Nmax_images = 16000

# Validation data
UseValDir = True #if it is False -> automatic dataset splitting
val_train_ratio = 0.3 #it is used in the automatic dataset splitting (hyperparameter)
random_st = random.randrange(42) #it is used in the automatic dataset splitting

# DataGenerator option
useDataGenerator = False
batch_size_DataGen = 128
Nimgs2read = 2500 #max number of images for each class of the test dataset to read if useDataGenerator

# Test data and related GradCAMS options
checkTestData = True
calculate_gradcam = True
delete_past_gradcam = True
gradcam4all = False
Ngradcams2plot = 100
imgtype_choice = 5 #5->any, 2->true positive (see the full list below)
closeGradcamPlots = True
saveGradcam2Text = False
saveGradcamTensor = False

# Select the CNN model 
models_available = ('DBT_DCNN_matlab', 'DBT_DCNN', 'Darknet19', \
                    'ResNet18', 'VGG16', 'VGG19', 'DenseNet121', \
                    'AlexNet', 'CNN1', 'CNN2')
model_type = models_available[1]
use_transfer_learning = True #only for ResNet18, VGG16, VGG19, DenseNet121
saveModel = True
modelName = 'model'
modelFormat = '.h5'
delete_past_weights = True
save_weights = False
load_best_model = True #for inference

# Default convolutional layers configuration for DBT_DCNN
Filters_DBT_DCNN = [8, 16, 32, 64, 64]
Kernels_DBT_DCNN = [9, 5, 3, 3, 3]
FLstrides = (2, 2) #if not (1, 1), The image size reducea already at the first layer (FL) -> parameters reduction

# Default convolutional layers configuration for CNN1
Filters_CNN1 = [6, 6, 6, 12, 12, 12]
Kernels_CNN1 = [3, 3, 3, 3, 3, 3]

# Shape of the (3) filters of the first conv layer if use_transfer_learning
FCL_filter_shape_TL = (16, 16)

# L2 regularization coefficient (for DBT_DCNN and if use_transfer_learning)
lambda_reg = 0.3

# Set class weights during training (by default they are uniform)
BalanceData = True #it has the priority
UseCustomClassWeights = False
custom_class_weights = [0.3, 0.7]

# Default Hyperparameters
default_model_hp = [16, 0.2, 0.5] #dln, alpha, dropout (for DBT_DCNN, CNN1 and if use_transfer_learning)
default_train_hp = ['adam', 10, 16] #optimizer, epochs, batch_size

# Hyperparameters search options [WARNING: if the GPU runs out of memory -> useDataGenerator or reduce Nmax_images]
HyperParametersSearch = False
GridSearch = True   #grid search for optimal hyperparamaters (set parameters below)
                    #if HyperParametersSearch = True and GridSearch = False -> RandomSearch
Nsearch_rnd = 5     #number of searches in case of RandomSearch
KfoldGS = 5;        #K-fold for cross-validation in GridSearch
simpleCV = True     #simple cross-validation with GridSearch and no DataGenerator

# Hyperparameters lists for search (if useDataGenerator, only the first 3 and epochs are considered)
dlns = [256, 512]
alphas = [0.1]
dropouts = [0.1, 0.3]
sel_optimizers = ['adam'] #'adam', 'rmsprop', 'sgd'
epochs = [15]
batches = [64]

# GPU options
disableGPUs = False
UseMultiGPUs = False
#####################################################################################################################

## Preliniminary operations
#####################################################################################################################
# Get the number of classes defined
Nclasses = len(classes)
try:
    if Nclasses < 2:
        raise Exception("The number of classes defined must be at least 2!")
finally:
    print("This is a classification problem of DBT slices in %d classes " % Nclasses)     

# Get img_type from the data (pay attention to filenames with dots)
if get_img_type_from_data:
    path0 = dataset_path + dataset_structure[0] + classes[0] + '/*'
    f0 = glob(os.path.join(path0))
    img_type = '.' + f0[0].split('/')[-1].split('.')[1]
    print("img_type: '%s'" % img_type)

# Set option variables so as they are compatible
if not HyperParametersSearch:
    GridSearch = False
    RandomSearch = False
    simpleCV = False
else:
    if not useDataGenerator:
        GridSearch = True
        RandomSearch = False
    else:
        RandomSearch = False
        if not GridSearch:
            RandomSearch = True
            
if useDataGenerator:
    augment_data = False
    UseMultiGPUs = False
    simpleCV = False
else:
    RandomSearch = False
    
if BalanceData:
    UseCustomClassWeights = False
    
if model_type not in ['ResNet18', 'VGG16', 'VGG19', 'DenseNet121']:
    use_transfer_learning = False

# Build the dataset path
if len(dataset_structure) > 2:
    Path_train, Path_val, Path_test = [dataset_path + data_type for data_type in dataset_structure]
    print('Path_train: %s\nPath_val: %s\nPath_test: %s' % (Path_train, Path_val, Path_test))
elif len(dataset_structure) > 1:
    Path_train, Path_val = [dataset_path + data_type for data_type in dataset_structure]
    print('Path_train: %s\nPath_val: %s' % (Path_train, Path_val))
    if checkTestData:
        Path_test = Path_val
        print('Path_test: %s' % Path_test)
else:
    Path_train = dataset_path + dataset_structure[0]
    print('Path_train: %s' % Path_train)  
    UseValDir = False
    checkTestData = False
    
# Automatic specification of the output subdirectory
fullstring = dataset_structure[0]
substrings = ['_train/', '_train-val/', '_train/CC/', '_train/CC2/'] 
remove_suffix = False
for substring in substrings:
    if substring in fullstring:
        remove_suffix = True
        break
if remove_suffix:
    outputSubDir = fullstring
    for substring in substrings:
        outputSubDir = outputSubDir.removesuffix(substring)
else:   
    outputSubDir = fullstring.removesuffix("/")
outputSubDir = outputSubDir + '_' + str(Nclasses) + '_' + model_type
if use_transfer_learning:
    outputSubDir = outputSubDir + '_TL'
outputSubDir = outputSubDir + '_' + str(dimInput[0]) + 'x' + str(dimInput[1])
if standardize_images:
    outputSubDir = outputSubDir + '_std'
if filterImg:
    outputSubDir = outputSubDir + '_filt'
if augment_data:
    outputSubDir = outputSubDir + '_aug'
if useDataGenerator:
    outputSubDir = outputSubDir + '_dataGen'
if HyperParametersSearch:
    if not GridSearch:
        outputSubDir = outputSubDir + '_RandomSearch'
    else:
        if not simpleCV:
            outputSubDir = outputSubDir + '_GridSearch'
        else:
            outputSubDir = outputSubDir + '_CV'
if BalanceData:
    outputSubDir = outputSubDir + '_BalancedW'
if UseCustomClassWeights:
    outputSubDir = outputSubDir + '_CustomW'
outputSubDir = outputSubDir + '_' + str(Nclasses) + 'classes'
outputSubDir = 'python_output/' + outputSubDir + '/'
print('outputSubDir:', outputSubDir)

# Create the working directories
outputDir = dataset_path + outputSubDir
if not os.path.exists(outputDir):
    os.makedirs(outputDir)
    print('created outputDir:', outputDir)

figuresDir = outputDir + 'figures/'
if not os.path.exists(figuresDir):
    os.makedirs(figuresDir)
    print('created figures subdir')

weightsDir = outputDir + 'weights/'
if not os.path.exists(weightsDir):
    os.makedirs(weightsDir)
    print('created weights subdir')
else:
    if delete_past_weights:
        files = os.listdir(weightsDir)
        Nfiles = len(files)
        print("\nnumber of files in %s directory to delete: %d" % (weightsDir, Nfiles))  
        for file in files:
            os.remove(weightsDir + '/' + file)
            #print('removed', file)  
    
GradCAMsDir = outputDir + 'GradCAM/'
if not os.path.exists(GradCAMsDir):
    os.makedirs(GradCAMsDir)
    print('created GradCAMs subdir')
else:
    if delete_past_gradcam:
        files = os.listdir(GradCAMsDir)
        Nfiles = len(files)
        print("\nnumber of files in %s directory to delete: %d" % (GradCAMsDir, Nfiles))  
        for file in files:
            os.remove(GradCAMsDir + '/' + file)
            #print('removed', file)  


# Print the Operating System
print("\nOperating System: %s" % platform.system()) #'Linux' or 'Windows'

# Print GPUs info
Ngpus = len(tf.config.list_physical_devices('GPU'))
print("number of GPUs Available:", Ngpus)
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True) 
    print("GPU name:", gpu.name) 
if platform.system() == 'Linux':    
    subprocess.run(["nvidia-smi"])
#subprocess.run(["nvidia-smi"])

# Disable the GPUs and use the CPU
if disableGPUs:    
    try:
        # Disable all GPUs
        tf.config.set_visible_devices([], 'GPU')
        visible_devices = tf.config.get_visible_devices()
        print("GPUs disabled")
        print(visible_devices)
        for device in visible_devices:
            assert device.device_type != 'GPU'
    except Exception as e:
        # Invalid device or cannot modify virtual devices once initialized
        print(e)   
else:    
    for gpu_id in range(Ngpus):
        initial_memory_usage = get_gpu_memory_usage(gpu_id)
        print('initial GPU:%d memory usage: %d MiB' % (gpu_id, initial_memory_usage))
        initial_free_memory = get_free_gpu_memory(gpu_id)
        print('initial GPU:%d free memory: %d MiB' % (gpu_id, initial_free_memory))
#####################################################################################################################

## Read the dataset
#####################################################################################################################
# Read the images in the dataset, store them in a numpy array and generate
# the appropriate numpy array with the labels corresponding to each instance. 
# If a DataGenerator is used, only Test data are read.

images_shuffled = False
if not useDataGenerator:
    if shuffle_imgs:
        print("\nreading (and shuffling) the Training dataset...")
        images_shuffled = True
    else:
        print("\nreading the Training dataset...")
    
    #read Training data
    x_train, y_train, img_names_train = read_images(Path_train, classes, dimInput, grayLevels, img_type, \
                                                    standardize_images, filterImg, sigmaFilter, shuffle_imgs, -1)
    n_train = x_train.shape[0]
    print('number of images in the Training dataset:', n_train)
    n_train_o = n_train
    
    # Augment Training data 
    if augment_data:    
        print('augmentation of the Training dataset by a factor = %.2f' % aug_factor)
        x_train, y_train, img_names_train = augment_images(x_train, y_train, img_names_train, aug_factor, Ntrasf)
        n_train = x_train.shape[0]
        print('number of images in the augmented Training dataset:', n_train)

    # Limit the number of images used
    if n_train > Nmax_images and limit_images: 
        print('limiting the number of Training images to:', Nmax_images)
        split_index = round(Nmax_images/2)
        x_train_temp = x_train[:split_index]
        y_train_temp = y_train[:split_index]    
        x_train_temp2 = x_train[-split_index:]
        y_train_temp2 = y_train[-split_index:]
        x_train_temp3 = [x_train_temp, x_train_temp2]
        y_train_temp3 = [y_train_temp, y_train_temp2]
        x_train = np.array(list_flatten(x_train_temp3))
        y_train = np.array(list_flatten(y_train_temp3))
        del x_train_temp, x_train_temp2, x_train_temp3
        del y_train_temp, y_train_temp2, y_train_temp3
    else:
        print('used Training dataset length:', x_train.shape[0])    
        
    # Validation data
    if UseValDir:
        # Direct import of Validation data 
        print("reading the Validation dataset...")
        X_val, Y_val, img_names_val = read_images(Path_val, classes, dimInput, grayLevels, img_type, \
                                                  standardize_images, filterImg, sigmaFilter, shuffle_imgs, -1)
        print('number of images in the loaded Validation dataset:', X_val.shape[0])
        if augment_data:    
            print('augmentation of the Validation dataset by a factor = %.2f ' % aug_factor)
            X_val, Y_val, img_names_val = augment_images(X_val, Y_val, img_names_val, aug_factor, Ntrasf)
        n_val = X_val.shape[0]
        X_train, Y_train = x_train, y_train
        val_train_ratio = n_val / n_train
        print('number of images in the augmented Validation dataset:', n_val)
        print('val_train_ratio:', round(val_train_ratio,2))
    else:
        # Automatic splitting of the dataset in Training and Validation data
        print('splitting of the Training dataset into Training and Validation (split_ratio = %.2f)' % val_train_ratio)
        indices = range(len(x_train))    
        X_train, X_val, Y_train, Y_val, i_train, i_val = train_test_split(x_train, y_train, indices,
                                                                          test_size=val_train_ratio, 
                                                                          random_state=random_st)
        img_names_val = [img_names_train[i] for i in i_val]
        img_names_train = [img_names_train[i] for i in i_train]
else:
    n_train = 0
    images_shuffled = False
    UseValDir = False
    checkTestData = True
    assert (
        checkTestData and len(dataset_structure) > 1
    ), "The considered dataset must be a-priori splittend at least in two parts!"
    print('\nusing a DataGenerator')

# Read independent Test data
if checkTestData:
    print("reading the Test dataset...")
    X_test, Y_test, img_names_test = read_images(Path_test, classes, dimInput, grayLevels, img_type, \
                                                 standardize_images, filterImg, sigmaFilter, False, Nimgs2read)

# Print the shape of the obtained arrays and their labels
if n_train > 0:
    print('Training data shape:', X_train.shape)
    print('Training labels shape:', Y_train.shape)
    print('Validataion data shape:', X_val.shape)
    print('Validataion labels shape:', Y_val.shape)
if checkTestData:
    print('Test data shape:', X_test.shape)
    print('Test labels shape:', Y_test.shape)

# Count the samples in each class
if n_train > 0:
    if checkTestData:
        labels = [Y_train, Y_val, Y_test]
        data_str = ['Training', 'Validation', 'Test']
    else:
        labels = [Y_train, Y_val]
        data_str = ['Training', 'Validation']
else:
    labels = [Y_test]
    data_str = ['Test']    

if Nclasses == 2:
    for target, j in zip(labels, range(len(data_str))):
        temp = np.array(target[:,1])     
        for i in range(Nclasses):
            print("number of labels '%s' in the %s dataset: %d" % (classes[i], data_str[j], len(temp[temp==i])))
else:
    for target, j in zip(labels, range(len(data_str))):
        Nneg = 0
        Npos = 0
        Nben = 0
        for item in target:
            if ([1, 0, 0] == item).all():
                Nneg = Nneg + 1
            elif ([0, 1, 0] == item).all():
                Npos = Npos + 1
            else:
                Nben = Nben + 1
        Ncases = [Nneg, Npos, Nben]
        for i in range(Nclasses):
            print("number of labels '%s' in the %s dataset: %d" % (classes[i], data_str[j], Ncases[i]))
            
# Calculate the memory occupation due to the read dataset
data_mem_occup_GB = 0
if 'X_train' in locals():
    X_train_mem_GB = X_train.size * X_train.itemsize / 10**9
    Y_train_mem_GB = Y_train.size * Y_train.itemsize / 10**9
    data_mem_occup_GB = X_train_mem_GB + Y_train_mem_GB
if 'X_val' in locals():
    X_val_mem_GB = X_val.size * X_val.itemsize / 10**9
    Y_val_mem_GB = X_val.size * Y_val.itemsize / 10**9
    data_mem_occup_GB += X_val_mem_GB + Y_val_mem_GB
if 'X_test' in locals():
    X_test_mem_GB = X_test.size * X_test.itemsize / 10**9
    Y_test_mem_GB = Y_test.size * Y_test.itemsize / 10**9
    data_mem_occup_GB += X_test_mem_GB + Y_test_mem_GB
print('memory occupation due to the read dataset: %.2f GB\n' % data_mem_occup_GB)


### Plot the histogram of the Training dataset classes
if 'Y_train' in locals():
    plot_histogram(Y_train, classes, figuresDir, "skyblue", "train", MPLBACKEND=MPLBACKEND)

# Plot the histogram of the Validation dataset classes
if 'Y_val' in locals():
    plot_histogram(Y_val, classes, figuresDir, "green", "val", MPLBACKEND=MPLBACKEND)

# Plot the histogram of the Test dataset classes
if 'Y_test' in locals():
    plot_histogram(Y_test, classes, figuresDir, "red", "test", MPLBACKEND=MPLBACKEND)
#####################################################################################################################

## Definition of DataGenerator objects
#####################################################################################################################
if useDataGenerator:
    # Define the list of images in the dataset (my IDs)
    training_names, training_labels = GetDataNamesAndLabels(Path_train, classes, img_type)
    validation_names, validation_labels = GetDataNamesAndLabels(Path_val, classes, img_type)
    
    # Define two DataGenerator objects for the training of the model
    training_generator = DataGenerator(training_names,
                                       training_labels, 
                                       batch_size=batch_size_DataGen,
                                       dim=dimInput,
                                       n_classes=Nclasses,
                                       grayLevels=grayLevels,
                                       standardize=standardize_images,
                                       filterImg=filterImg,
                                       sigmaFilter=sigmaFilter)
    
    validation_generator = DataGenerator(validation_names,
                                         validation_labels, 
                                         batch_size=batch_size_DataGen,
                                         dim=dimInput,
                                         n_classes=Nclasses,
                                         grayLevels=grayLevels,
                                         standardize=standardize_images,
                                         filterImg=filterImg,
                                         sigmaFilter=sigmaFilter)
#####################################################################################################################    

## Determine class weights (new part to take into account unbalced data or to favor positive cases)
#####################################################################################################################       
if BalanceData:
    # Get unique labels as a function of classes
    if Nclasses > 2:
        YL = np.array([np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])])
    else:
        YL = [np.array([1, 0]), np.array([0, 1])]

    Ny_per_class = np.zeros(Nclasses)

    # Get the number of labels for each class
    if not useDataGenerator:
        for i in range(Nclasses):
            Ny_per_class[i] = len([j for j in range(Y_train.shape[0]) if (Y_train[j] == YL[i]).all()])
    else:
        for i in range(Nclasses):
            Ny_per_class[i] = len([key for key in training_labels.keys() if (training_labels[key] == YL[i]).all()])
 
    # Compute class weights (through a custom function)
    class_weights = compute_class_weight(Ny_per_class)
    
elif UseCustomClassWeights:
    class_weights = np.array(custom_class_weights)
    
else:
    class_weights = np.ones(Nclasses)*0.5 #defualt: uniform weights

class_weights_dict = dict(enumerate(class_weights))
print('\nclass_weights_dict:', class_weights_dict)
if Nclasses == 2:
    print('Wclass1/Wclass0:', class_weights[1]/class_weights[0])  
#####################################################################################################################        
    
# Define Callbacks and Optimizers
#####################################################################################################################
# Define useful training callbacks
if save_weights:
    file_cp = weightsDir + "model-{epoch:03d}-{loss:.2f}-{val_accuracy:.2f}.weights" + modelFormat
else:
    file_cp = weightsDir + 'best_' + modelName + modelFormat
checkpoint = ModelCheckpoint(
    filepath=file_cp,
    monitor='val_accuracy', 
    verbose=0,
    save_best_only=True,
    save_weights_only=save_weights,
    #save_freq = 'epoch'
    mode='max' #auto
)

reduce_on_plateau = ReduceLROnPlateau(
    #monitor="loss",
    monitor='accuracy',
    factor=0.5,
    patience=10,
    verbose=1,
    mode="auto",
    min_delta=0.0001,
    cooldown=0,
    min_lr=0.000005)

early_stopping = EarlyStopping(
    #monitor="loss",
    monitor="val_loss",
    min_delta=0,
    patience=25,
    verbose=0,
    mode="auto",
    baseline=None,
    restore_best_weights=True)

plotlosses = PlotLossesKerasTF()

# Define explicitly Adam optimizer
adamlr = optimizers.Adam(learning_rate=0.0001, 
                         beta_1=0.9, 
                         beta_2=0.999, 
                         epsilon=1e-08, 
                         amsgrad=True)
#####################################################################################################################

## Model buiding
#####################################################################################################################
# Function to select and compile the model 
def get_compiled_model(model_type='DBT_DCNN', 
                       dimInput=(300, 300), 
                       classes=('Negative', 'Positive', 'Benign'), 
                       dln=64,
                       alpha=0.1,
                       dropout=0.2,
                       the_optimizer=adamlr):
    
    # input shape
    input_shape = [dimInput[0], dimInput[1], 1]
    
    # set variables depending on use_transfer_learning
    if use_transfer_learning:       
        input_shape[2] = 3
        use_top_layer = False
        weigths_used = 'imagenet'        
        input_tensor_used = None
        Nclasses = 1000
    else:
        use_top_layer = True
        weigths_used = None        
        Nclasses = len(classes)
        input_tensor_used = Input(shape=input_shape)
                
    # Select the model
    if model_type == 'DBT_DCNN_matlab':
        model = DBT_DCNN(dimInput, classes)
    
    elif model_type == 'DBT_DCNN':   
        model = DBT_DCNN(dimInput, classes,
                         dln, alpha, dropout,
                         Sigma=0.02, Lambda=lambda_reg,
                         #Filters=[8, 16, 32, 64, 128],
                         #Kernels=[11, 5, 3, 3, 3],
                         Filters=Filters_DBT_DCNN,
                         Kernels=Kernels_DBT_DCNN,
                         pool_size=[(3,3), (3,3), (3,3), (3,3)],
                         strides=[FLstrides, (2,2), (2,2), (2,2), (2,2)])
        
    elif model_type == 'Darknet19':   
        model = Darknet19_classifier(dimInput, classes)
        
    elif model_type == 'ResNet18':
        from classification_models.keras import Classifiers
        ResNet18, preprocess_input = Classifiers.get('resnet18')
        base_model = ResNet18(
                              include_top=use_top_layer,
                              weights=weigths_used,  
                              input_shape=input_shape,
                              input_tensor=input_tensor_used,
                              classes=Nclasses,
                              classifier_activation='softmax'
                             )
        
    elif model_type == 'VGG16':
        base_model = tf.keras.applications.vgg16.VGG16(
                                                       include_top=use_top_layer,
                                                       weights=weigths_used,
                                                       input_shape=input_shape,
                                                       input_tensor=input_tensor_used,
                                                       pooling=None,
                                                       classes=Nclasses,
                                                       classifier_activation='softmax'
                                                      )

    elif model_type == 'VGG19':
        base_model = tf.keras.applications.vgg19.VGG19(
                                                       include_top=use_top_layer,
                                                       weights=weigths_used,
                                                       input_shape=input_shape,
                                                       input_tensor=input_tensor_used,
                                                       pooling=None,
                                                       classes=Nclasses,
                                                       classifier_activation='softmax'
                                                      )

    elif model_type == 'DenseNet121':
        base_model = tf.keras.applications.DenseNet121(
                                                       include_top=use_top_layer,
                                                       weights=weigths_used,
                                                       input_shape=input_shape,
                                                       input_tensor=input_tensor_used,
                                                       pooling=None,
                                                       classes=Nclasses,
                                                       classifier_activation='softmax'
                                                      )

    elif model_type == 'AlexNet':
        model = AlexNet(dimInput+(1,), Nclasses) 
        
    elif model_type == 'CNN1':   
        model = CNN1(dimInput, classes,
                     dln, alpha, dropout,
                     #Filters=[6, 6, 6, 12, 12, 12],
                     #Kernels=[3, 3, 3, 3, 3, 3]
                     Filters=Filters_CNN1,
                     Kernels=Kernels_CNN1)
    
    elif model_type == 'CNN2':   
        model = CNN2(dimInput+(1,), Nclasses) 
        
    # complete the base model to enable transfer leraning   
    if 'base_model' in locals():
        if use_transfer_learning:                    
            # print the base model summary
            #base_model.summary()
                        
            # frezee the layers of the base model 
            base_model.trainable = False
            #for layer in base_model2.layers: #equivalent method1
                #layer.trainable = False
            #for i in list(range(len(base_model.layers))): #equivalent method2
                #base_model.layers[i].trainable = False               
                     
            #remove the last (classification) layer of the base model (for ResNet18, it is not required)
            #base_model.layers.pop() #this does not work (even if it does not give errors)!
            #base_model = Sequential(base_model.layers[:-1]) #this works depending on the CNN architecture!
            #base_model = Model(base_model.input, base_model.layers[-3].output) #this works!         
            #base_model.trainable = False
            #base_model.summary()

            # define new layers
            layer_in = Input(shape=(*dimInput, 1))
            first_conv_layer = Conv2D(3, FCL_filter_shape_TL, strides=(1,1), padding='same')  
            flatten_layer = Flatten()
            reg = regularizers.L2(l2=lambda_reg)
            #dense_layer_1 = Dense(50, activation='relu')            
            #dense_layer_2 = Dense(dln, kernel_regularizer=reg)
            dense_layer_1 = Dense(dln, activation=LeakyReLU(alpha), kernel_regularizer=reg)
            dense_layer_2 = Dense(dln, activation=LeakyReLU(alpha), kernel_regularizer=reg) 
            dropout_layer = Dropout(dropout)
            prediction_layer = Dense(len(classes), activation='softmax')

            # assemble the new model using the base model plus the new layers
            model = Sequential([
                layer_in,
                first_conv_layer,
                base_model,
                flatten_layer,
                dense_layer_1,
                dense_layer_2,
                dropout_layer,
                prediction_layer
            ])
        else:
            model = base_model
            del base_model
            
    # Compile the model
    model.compile(loss='categorical_crossentropy', 
                  optimizer=the_optimizer, 
                  metrics=['accuracy'])
    
    # Return the compiled model
    return model
#####################################################################################################################

## Search for optimal hyperparameters
#####################################################################################################################
## Grid Search without a DataGenerator
#%%time
if GridSearch and not useDataGenerator:
    # Define an estimator, namely a Keras wrapped Classifier
    estimator_GS = KerasClassifier(model=get_compiled_model, 
                                   model_type=model_type, #custom param
                                   dimInput=dimInput, #custom param
                                   classes=classes, #custom param
                                   dln=256, #custom param
                                   alpha=0.1, #custom param
                                   dropout=0.1, #custom param
                                   epochs=15, 
                                   batch_size=32,
                                   shuffle=True,
                                   optimizer=adamlr,
                                   callbacks=[early_stopping],
                                   verbose=1)

    #print(estimator_GS.get_params().keys())
    
    # Define the grid (a dictionary) of hyperparameters
    param_grid = dict(
                      dln = dlns,
                      alpha = alphas,
                      dropout = dropouts,
                      optimizer = sel_optimizers, 
                      epochs = epochs, 
                      batch_size = batches
                     )
    
    if simpleCV:
        param_grid = {}
        
    Nsearch = len(param_grid)
    
    # Define a grid search object
    grid = GridSearchCV(estimator=estimator_GS, 
                        param_grid =param_grid,
                        scoring='accuracy',
                        cv=KfoldGS,
                        refit=False,
                        return_train_score=True,
                        verbose=4)
    
    # Train the grid
    grid_result = grid.fit(x_train, y_train)
    
    # Summarize results
    print("Best %d-fold %s = %f using %s" % (grid.cv, grid.scoring, grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%d-fold %s = %f +/- %f with: %r" % (grid.cv, grid.scoring, mean, stdev, param))
   
    # Delete variables
    del estimator_GS
    del grid_result
    
        
## Grid Search with a DataGenerator 
def my_grid_search(dlns, alphas, dropouts, epochs):
#def my_grid_search(dlns=[64], alphas=[0.1], dropouts=[0.2], epochs=[5]): #I want to pass all the arguments explicitly 
    history_page = []

    Nsearch = len(dlns)*len(alphas)*len(dropouts)*len(epochs)
    print('Grid search of optimal hyperparameters. A total of %d training will be carried out with:' % Nsearch)
    param_len_list = [len(dlns), len(alphas), len(dropouts), len(epochs)]
    #print('param_len_list:', param_len_list)
    print('%d dln values:' % len(dlns), dlns)
    print('%d alpha values:' % len(alphas), alphas)
    print('%d dropout values:' % len(dropouts), dropouts)
    print('%d Nepoch values:' % len(epochs), epochs)

    i = 0
    for dln in dlns:           
        for alpha in alphas:
            for dropout in dropouts:
                for Nepoch in epochs:
                    print('Training (configuration) number %d with: dln = %d, alpha= %.2f, dropout = %.2f, Nepoch = %d' % (i, dln, alpha, dropout, Nepoch))

                    model = get_compiled_model(model_type=model_type, 
                                               dimInput=dimInput, 
                                               classes=classes, 
                                               dln=dln,
                                               alpha=alpha,
                                               dropout=dropout) 

                    history = model.fit(training_generator,
                                        validation_data=validation_generator,
                                        epochs=Nepoch,    
                                        callbacks=[early_stopping],
                                        verbose=1)    

                    history_page.append(history)
                    i += 1

    return history_page, param_len_list

if GridSearch and useDataGenerator:
    history_page, param_len_list = my_grid_search(dlns, alphas, dropouts, epochs)


## Random Search with a DataGenerator     
def my_random_search(dlns, alphas, dropouts, epochs, Nsearch):
    param_configuration = []
    history_page = []
    print('Random search of optimal hyperparameters. A total of %d training will be carried out with:' % Nsearch)
    
    for i in range(Nsearch):
        #set hyperparametrs randomly       
        dln = random.sample(dlns, 1)[0]    
        alpha = random.sample(alphas, 1)[0]    
        dropout = random.sample(dropouts, 1)[0]    
        Nepoch = random.sample(epochs, 1)[0]  
        
        print('Training (configuration) number %d with: dln = %d, alpha= %.2f, dropout = %.2f, Nepoch = %d' % (i, dln, alpha, dropout, Nepoch))

        model = get_compiled_model(model_type=model_type, 
                                   dimInput=dimInput, 
                                   classes=classes, 
                                   dln=dln,
                                   alpha=alpha,
                                   dropout=dropout) 

        history = model.fit(training_generator,
                            validation_data=validation_generator,
                            epochs=Nepoch,    
                            callbacks=[early_stopping],
                            verbose=1)     
        
        history_page.append(history)
        i += 1
       
        param_configuration.append([dln, alpha, dropout, Nepoch])
        
    return history_page, param_configuration

if RandomSearch:
    history_page, param_configuration = my_random_search(dlns, alphas, dropouts, epochs, Nsearch_rnd)


## Analyze the results of the hyperparameters search with a DataGenerator
if HyperParametersSearch and useDataGenerator:

    Nsearch = len(history_page)

    acc_last = np.zeros((Nsearch))
    for i in range(Nsearch):        
        acc_last[i] = history_page[i].history['accuracy'][-1]   

    acc_mean = np.mean(acc_last)
    acc_std = np.std(acc_last)
    acc_best_index = np.argmax(acc_last)

    print('Best accuracy = %f +/- %f obtained with the configuration number %d, which is:' % (acc_mean, acc_std, acc_best_index))

    if GridSearch:    
        prod = np.zeros(4, dtype=int)
        for i, param_len in enumerate(param_len_list):
            sub_list = [param_len_list[j] for j in range(len(param_len_list)) if j!=i]
            prod[i] = np.array(multiply_list(sub_list))

        dlns_extended = repeat_list_elements(dlns, prod[0])
        alphas_extended = repeat_list_elements(alphas, prod[1])
        dropouts_extended = repeat_list_elements(dropouts, prod[2])
        epochs_extended = repeat_list_elements(epochs, prod[3])

        best_dln = dlns_extended[acc_best_index]
        best_alpha = alphas_extended[acc_best_index]
        best_dropout = dropouts_extended[acc_best_index]
        best_epochs = epochs_extended[acc_best_index]
    else:
        best_dln = param_configuration[acc_best_index][0]
        best_alpha = param_configuration[acc_best_index][1]
        best_dropout = param_configuration[acc_best_index][2]
        best_epochs = param_configuration[acc_best_index][3] 
#####################################################################################################################

## Set the (optimal) model hyperparameters
#####################################################################################################################
model_hp = default_model_hp

if HyperParametersSearch:
    if GridSearch and not useDataGenerator and not simpleCV:
        if 'dln' in grid_result.best_params_.keys():
            best_dln = grid_result.best_params_['dln']
        if 'alpha' in grid_result.best_params_.keys():
            best_alpha = grid_result.best_params_['alpha']
        if 'dropout' in grid_result.best_params_.keys():  
            best_dropout = grid_result.best_params_['dropout']
    
    if 'best_dln' in locals():
        print('best dln = %.d' % best_dln)
        model_hp[0] = best_dln
    if 'best_alpha' in locals():
        print('best alpha = %.2f' % best_alpha)
        model_hp[1] = best_alpha
    if 'best_dropout' in locals():
        print('best dropout = %.2f' % best_dropout)
        model_hp[2] = best_dropout

print('\nmodel hyperparameters set:', model_hp)    
#####################################################################################################################

## Set the (optimal) training hyperparameters
#####################################################################################################################
if useDataGenerator:
    default_train_hp[2] = batch_size_DataGen
train_hp = default_train_hp
train_hp[0] = adamlr

if HyperParametersSearch:
    if GridSearch and not useDataGenerator and not simpleCV:
        if 'optimizer' in grid_result.best_params_.keys():
            best_optimizer = grid_result.best_params_['optimizer']
        if 'epochs' in grid_result.best_params_.keys():
            best_epochs = grid_result.best_params_['epochs']
        if 'batch_size' in grid_result.best_params_.keys():  
            best_batch_size = grid_result.best_params_['batch_size']
    
    if 'best_optimizer' in locals():
        print('best optimizer = %s' % best_optimizer)
        train_hp[0] = best_optimizer        
    if 'best_epochs' in locals():
        print('best epochs = %d' % best_epochs)
        train_hp[1] = best_epochs
    if 'best_batch_size' in locals():
        print('best batch_size = %d' % best_batch_size)
        train_hp[2] = best_batch_size

print('training hyperparameters set:', train_hp, '\n')    
#####################################################################################################################

## Compile the model
#####################################################################################################################
# Comiple the selected (and optimized) model using a different strategy depending on the number of used GPUs
if UseMultiGPUs:
    # Create a MirroredStrategy
    strategy = tf.distribute.MirroredStrategy()
    print("Number of devices: {}".format(strategy.num_replicas_in_sync))
    # Open a strategy scope
    with strategy.scope():
        model = get_compiled_model(model_type, dimInput, classes, model_hp[0], model_hp[1], model_hp[2])
else:
    model = get_compiled_model(model_type, dimInput, classes, model_hp[0], model_hp[1], model_hp[2])
    
# Plot model summary
model.summary()

# Save the model scheme
plot_model(model, to_file=figuresDir + modelName + '.jpg', show_shapes=True, show_layer_names=True)

# Calculate the memory occupation
model_mem_occup_GB = get_model_memory_usage(train_hp[2], model)
total_mem_occup_GB = data_mem_occup_GB + model_mem_occup_GB
print('model memory occupation: %.2f GB' % model_mem_occup_GB)
print('total (data + model) memory occupation: %.2f GB' % total_mem_occup_GB)

# Calculate the GPU free memory
if not disableGPUs:
    free_memory = 0
    for gpu_id in range(Ngpus):
        free_memory += get_free_gpu_memory(gpu_id)
    print('total GPU free memory: %.2f GB' % float(float(free_memory)/1024))
    if free_memory < 200:
        print('potential issues: free GPU memory < 200 MiB')
#####################################################################################################################        

## Train the (optimized) model on the full dataset
#####################################################################################################################
#%%time

print("\nmodel training...")

# Training parameters
batch_size = train_hp[2]
trainig_epochs = train_hp[1]

# Select the strategy depending on the number of used GPUs
if UseMultiGPUs:
    # Wrap data in Dataset objects
    train_data = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
    val_data = tf.data.Dataset.from_tensor_slices((X_val, Y_val))
    
    # The batch size must now be set on the Dataset objects
    train_data = train_data.batch(batch_size)
    val_data = val_data.batch(batch_size)
    
    # Disable AutoShard
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
    train_data = train_data.with_options(options)
    val_data = val_data.with_options(options)
    
    # Train the model 
    history = model.fit(train_data,
                        validation_data=val_data,
                        epochs=trainig_epochs,
                        shuffle=True,
                        class_weight=class_weights_dict,
                        callbacks=[early_stopping, checkpoint, reduce_on_plateau],
                        verbose=1)    
else:
    # Train the model        
    if useDataGenerator:
        history = model.fit(training_generator, 
                            validation_data=validation_generator,
                            epochs=trainig_epochs,
                            class_weight=class_weights_dict,
                            callbacks=[early_stopping, checkpoint, reduce_on_plateau],                            
                            verbose=1)
    else:
        history = model.fit(X_train, Y_train,                                      
                            validation_data=(X_val, Y_val),
                            batch_size=batch_size,
                            epochs=trainig_epochs,
                            shuffle=True,
                            class_weight=class_weights_dict,
                            #steps_per_epoch=X_train.shape[0]/batch_size, #automatic        
                            callbacks=[early_stopping, checkpoint, reduce_on_plateau],
                            verbose=1)

# Save the (trained) model (at the last epoch)
model_saved = False
if saveModel:
    model_file = outputDir + modelName + modelFormat
    if os.path.exists(model_file):
        os.remove(model_file)
    if simpleCV and (model_type == 'ResNet18' or model_type == 'VGG16' or \
                     model_type == 'VGG19' or model_type == 'DenseNet121'):
        print('WARNING: trained model not saved!')  
    else:
        model.save(model_file)
        model_saved = True   
    
# Save the model training history
with open(outputDir + 'history_' + modelName + '.json', 'w') as f:
    json.dump(str(history.history), f)

    
# Calculate the used GPU memory
if not disableGPUs:
    for gpu_id in range(Ngpus):    
        latest_gpu_memory = get_gpu_memory_usage(gpu_id)
        print('GPU:%d memory usage: %d MiB' % (gpu_id, latest_gpu_memory - initial_memory_usage))
        final_free_memory = get_free_gpu_memory(gpu_id)
        print('GPU:%d free memory: %d MiB' % (gpu_id, final_free_memory))
        
        
# Plot the model training curves  
plot_training_history(history.history, figuresDir, MPLBACKEND=MPLBACKEND)


# Write the configuration file
with open(outputDir + 'config.txt', 'w') as f:
    f.write('input:\n')
    f.write('dimInput: (%s, %s)\n' % (dimInput[0], dimInput[1]))
    f.write('useDataGenerator: %s\n' % useDataGenerator)
    if (useDataGenerator):
        f.write('batch_size_DataGen: %s\n' % batch_size_DataGen)
    f.write('UseValDir: %s\n' % UseValDir)
    f.write('val_train_ratio: %s\n' % round(val_train_ratio, 2))
    f.write('augment_data: %s\n' % augment_data)
    if augment_data:
        f.write('aug_factor: %s\n' % aug_factor)
    f.write('limit_images: %s\n' % limit_images)
    if limit_images:
        f.write('Nmax_images: %s\n' % Nmax_images)
    if not useDataGenerator:
        f.write('Nimages_used: %s\n' % x_train.shape[0])
    f.write('images_shuffled: %s\n' % images_shuffled)
    f.write('standardize_images: %s\n' % standardize_images)
    f.write('filterImg: %s\n' % filterImg)
    if filterImg:
        f.write('sigmaFilter: %s\n' % sigmaFilter)
    f.write('model_type: %s\n' % model_type)
    f.write('modelName: %s\n' % modelName)
    f.write('modelFormat: %s\n' % modelFormat)
    f.write('model_saved: %s\n' % model_saved)
    f.write('load_best_model: %s\n' % load_best_model)
    f.write('\nHyperparameters:\n')
    if HyperParametersSearch:        
        f.write('GridSearch: %s\n' % GridSearch)
        f.write('RandomSearch: %s\n' % RandomSearch)
        f.write('simpleCV: %s\n' % simpleCV) 
        if simpleCV:
            f.write("%d-fold %s = %f +/- %f\n" % (grid.cv, grid.scoring, mean, stdev)) 
        else:
            f.write('Nsearch: %s\n' % Nsearch) 
    else:
        f.write('No search for optimal hyperparameters carried out\n')
    if model_type != None and ('DBT_DCNN' in model_type or use_transfer_learning):
        f.write('dln: %s\n' % model_hp[0])
        f.write('alpha: %s\n' % model_hp[1])
        f.write('dropout: %s\n' % model_hp[2]) 
    f.write('optimizer: %s\n' % train_hp[0])
    f.write('epochs: %s\n' % train_hp[1])
    f.write('batch_size: %s\n' % train_hp[2])
    f.write('\nlambda_reg: %s\n' % lambda_reg)
    f.write('BalanceData: %s\n' % BalanceData)
    f.write('UseCustomClassWeights: %s\n' % UseCustomClassWeights)
    if UseCustomClassWeights:
        if Nclasses == 3:
            f.write('custom_class_weights: [%s, %s, %s]\n' % (custom_class_weights[0], custom_class_weights[1], custom_class_weights[2]))
        else:
            f.write('custom_class_weights: [%s, %s]\n' % (custom_class_weights[0], custom_class_weights[1]))
#####################################################################################################################

## Model evaluation
#####################################################################################################################
# Load the best model obtained during training
tensorflow.keras.backend.clear_session()
if 'model' in locals():
    del(model)
model = load_model(model_file, custom_objects={"LeakyReLU": LeakyReLU})
if load_best_model:
    print('\nloading the best model obtained during training...')
    if save_weights:
        files = os.listdir(weightsDir)
        w_epoch = []
        w_loss = []
        w_val_acc = []
        for file in files:
            w_epoch.append(file.split('-')[1])
            w_loss.append(file.split('-')[2])
            val_acc = float(file.split('-')[3].split(modelFormat)[0])
            val_acc = float(file.split('-')[3].split('.weights')[0])
            w_val_acc.append(val_acc)
        val_acc_max, imax = find_list_max(w_val_acc)        
        best_model = 'model-' + w_epoch[imax] + '-' + w_loss[imax] + '-' + str(val_acc_max) + '.weights' + modelFormat
    else:
        best_model = 'best_' + modelName + modelFormat
    print('loading %s\n' % best_model)
    model.load_weights(filepath=weightsDir + best_model)
else:
    print('using the weights obtained at the last epoch')
    

# Evaluate the model on the Training data
if 'Y_train' in locals():
    if Nclasses == 2:
        preds_train, labels_train, results_train = evaluate_model_2classes(model, X_train, Y_train, classes, 
                                                                           figuresDir, set_type='train',
                                                                           MPLBACKEND=MPLBACKEND)
    else:
        preds_train, labels_train, results_train = evaluate_model(model, X_train, Y_train, classes, 
                                                                  figuresDir, set_type='train',
                                                                  MPLBACKEND=MPLBACKEND)


# Evaluate the model on the Validation data 
if 'Y_val' in locals():
    if Nclasses == 2:
        preds_val, labels_val, results_val = evaluate_model_2classes(model, X_val, Y_val, classes, 
                                                                     figuresDir, set_type='val',
                                                                     MPLBACKEND=MPLBACKEND)
    else:
        preds_val, labels_val, results_val = evaluate_model(model, X_val, Y_val, classes, 
                                                            figuresDir, set_type='val',
                                                            MPLBACKEND=MPLBACKEND)


# Evaluate the model on the Test data
if 'Y_test' in locals():
    if Nclasses == 2:
        preds_test, labels_test, results_test = evaluate_model_2classes(model, X_test, Y_test, classes, 
                                                                        figuresDir, set_type='test',
                                                                        MPLBACKEND=MPLBACKEND)
    else:
        preds_test, labels_test, results_test = evaluate_model(model, X_test, Y_test, classes,
                                                               figuresDir, set_type='test',
                                                               MPLBACKEND=MPLBACKEND)
    
    print('\n')
    
    # calculate the GradCAMs          
    if calculate_gradcam:
        print('calculating the GradCAMs...')        
        
        pure_img_names_test = [item.split('/')[-1].split('.')[0] for item in img_names_test]
        
        if gradcam4all:
            gradcam_tensor, gradcams_img_names = GradCAM_calculation_all(dataset_path, dataset_structure, classes, img_type, 
                                                                    model, X_test, Y_test, pure_img_names_test, GradCAMsDir, 
                                                                    saveGradcam2Text, sn=True, nf=0.0015, MPLBACKEND=MPLBACKEND)

        else:
            gradcam_tensor, gradcams_img_names = GradCAM_calculation(dataset_path, dataset_structure, classes, img_type, 
                                                                model, X_test, Y_test, pure_img_names_test, GradCAMsDir,
                                                                saveGradcam2Text, close_plots=closeGradcamPlots, sn=True, 
                                                                nf=0.025, type_choice=imgtype_choice, Nfigs2plot=Ngradcams2plot, 
                                                                random_choice=True, MPLBACKEND=MPLBACKEND)
            #type_choice: 0->positive, 1->predicted positive, 2->true postive, 
            #             3->negative predicted as postive, 4->benign predicted as postive, 
            #             5->any

        # save a numpy tensor with the calculated (RGB) GradCAMs
        if saveGradcamTensor:
            np.save(GradCAMsDir + 'gradcams', gradcam_tensor)
            file = open(GradCAMsDir + 'gradcams_names.txt', 'w')
            for item in gradcams_img_names:
                file.write(item + "\n")
#####################################################################################################################

