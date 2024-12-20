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


# Load a trained model and evaluate it on the Test data


## Import the required libraries
#####################################################################################################################
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from glob import glob
import platform
import subprocess
import json
import ast
import re
import time

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from keras.utils import plot_model

import sklearn
from sklearn.metrics import roc_curve, auc, cohen_kappa_score, classification_report

from utils import *

#os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
print("\nTensorflow version:", tf.__version__) 

os.environ['MPLBACKEND'] = 'Agg' 
MPLBACKEND = os.environ['MPLBACKEND'] 
print('MPLBACKEND:', MPLBACKEND, '\n')

import warnings
warnings.filterwarnings("ignore", category = np.VisibleDeprecationWarning)
warnings.filterwarnings(action = 'ignore', category = FutureWarning)
warnings.filterwarnings(action = 'ignore', category = RuntimeWarning)
#####################################################################################################################

## Input
#####################################################################################################################
# Define the data structure
#dataset_path = '...'
#classes = '' #leave it blank ('') for unknown data

dataset_path = '/media/paterno/SSD1TB/'
dataset_structure = ['DUKE_fully_processed_sample_dataset_all6PcTest_test/']
classes = ('negative', 'positive')
#classes = ('negative', 'positive', 'benign')

# Options for Patient-based Scoring
patientBasedScoring = False
FP_eval = np.arange(0, 8, 0.1)
microAverageFROC = True
includeFPnull = True
setNullTPR0 = True

# Image features
img_type = '.tiff' 
get_img_type_from_data = False #retrieve img_type from the data (pay attention to filenames with dots)
grayLevels = 65535 #2^Nbit-1

# Input size of the images to feed the CNN with (it is a hyperparameter)
dimInput = (300, 300) #pixel

# Standardization of images
standardize_images = True #img = (img - mean) / std

# Filtering of images
filterImg = False #with a Gaussian filter of sigmaFilter pixels (deaulft is False)
sigmaFilter = 2 #default is 2

# Set the output sub directory (of dataset_path)
outputSubDir = 'python_output/test/' 

# Set the model and weights path
modelAndWeightBasePath = '/media/paterno/SSD1TB/python_output/DUKE_fully_processed_sample_dataset_all6PcTest_2_VGG16_TL_300x300_std_aug_BalancedW_2classes/' 
modelName = 'model'
modelFormat = '.h5' #'.h5' or '.hdf5'
UseWeigts = False
weigthsName = 'model'
load_best_model = True

# GradCAM options
calculate_gradcam = True
gradcam4all = False
Ngradcams2plot = 100
imgtype_choice = 5 #5->any, 2->true positive (see the full list below)
closeGradcamPlots = True
saveGradcam2Text = False
saveGradcamTensor = False

# GPU Options
disableGPUs = False

# Verbosity
verboseLevel = 1
#####################################################################################################################

## Preliniminary operations
#####################################################################################################################
# Sanity check
try:
    patientBasedScoring and (classes == ('') or len(classes) > 2)
except:
    raise Exception('Patient-based analysis cannot be carried out with unlabel data or for 3 classes problems. Check classes definition!')    

# Set unlabeled_dataset
if classes == (''):
    unlabeled_dataset = True
else:
    unlabeled_dataset = False
    
# Get info about dataset_structure to automatize operations
if len(dataset_structure) == 1:
    ids = 0
else:
    ids = 1
len_part_dataset_structure = len(dataset_structure[ids].split('/'))
    
# Get img_type from the data (pay attention to filenames with dots)
if get_img_type_from_data:
    if classes == (''):
        strcls = ''
    elif len(classes) == 1:
        strcls = classes
    else:
        strcls = classes[0]
    if len_part_dataset_structure == 5:
        mod1 = '' 
    elif len_part_dataset_structure == 4:
        mod1 = '*'  
    elif len_part_dataset_structure == 3:
        mod1 = '*/*'
    else:
        mod1 = '*/*/*'
    if patientBasedScoring:
        path1 = glob(dataset_path + dataset_structure[0] + mod1)[0] + '/'
    else:
        path1 = dataset_path + dataset_structure[0]
    path0 = path1 + classes[0] + '/*'
    if len(glob(path0)) == 0:
        path0 = path1 + classes[1] + '/*'                              
    f0 = glob(os.path.join(path0))    
    img_type = '.' + f0[0].split('/')[-1].split('.')[1]
    print("img_type: '%s'" % img_type)

# Build paths
Path_test = dataset_path + dataset_structure[ids]
print("Path_test:", Path_test)

modelPath = modelAndWeightBasePath + modelName + modelFormat
print("modelPath:", modelPath)

if not UseWeigts:
    load_best_model = False

if load_best_model:
    weigthsPath = modelAndWeightBasePath + 'weights/' + 'best_' + modelName + modelFormat
else:
    weigthsPath = modelAndWeightBasePath + 'weights/' + weigthsName + modelFormat
print("weigthsPath:", weigthsPath)

# Create the working directories
figuresDir = dataset_path + outputSubDir + 'figures/'
if not os.path.exists(figuresDir):
    os.makedirs(figuresDir)

if not patientBasedScoring:
    GradCAMsDir = dataset_path + outputSubDir + 'GradCAM/'
    if not os.path.exists(GradCAMsDir):
        os.makedirs(GradCAMsDir)

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
        print("GPU disabled")
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
        
## Load the model
#####################################################################################################################
# Load the trained model
model = load_model(modelPath)
model.summary()
if UseWeigts:
    print('\nloading %s\n' % weigthsPath)
    model.load_weights(filepath=weigthsPath)  
plot_model(model, to_file=figuresDir + modelName + '.jpg', show_shapes=True, show_layer_names=True)
    
# Import the model traning history
histoty_path = modelPath.split(modelName)[0]
f = open(histoty_path + 'history_' + modelName + '.json')
history = ast.literal_eval(json.loads(f.read()))

# Plot the model training curves
plot_training_history(history, figuresDir, MPLBACKEND=MPLBACKEND)
#####################################################################################################################
        
## Read and elaborate the data
#####################################################################################################################
# Read the Test data
print("\nreading the Test dataset...")  

if patientBasedScoring:
    start = time.time()    
    
    print("Patient-based analysis")
    #NOTE: there can be patients with slices in more than one class (for Cardarelli and IFO). 
    #Moreover, for some dataset (e.g. DUKE), one patient can have slices 
    #of more than one projection in the same class. They are correcly read.
    if len_part_dataset_structure == 5:
        mod2 = ''
    elif len_part_dataset_structure == 4:
        mod2 = '*/'
    elif len_part_dataset_structure == 3:
        mod2 = '*/*/'
    else:
        mod2 = '*/*/*/'
    patients_list = glob(os.path.join(Path_test + mod2))  
    cls_list = ['negative/', 'Negative/', 'positive/', 'Positive/']
    test_cls = [Path_test + item for item in cls_list]    
    for item in test_cls:
        if item in patients_list:
            patients_list.remove(item)
    if verboseLevel > 2:
        print(patients_list)
    Npatients = len(patients_list)
    print("Npatients:", Npatients)
       
    TPR_mean = 0.
    results = {}
    
    fp_list = []
    tpr_list = []
    
    plt.figure(figsize=(10, 6))
    lwp = 2
    fs = 14
    
    Npatients_calc = 0
    for patient_path in patients_list:
        set_list = ['Cardarelli', 'DUKE', 'IFO']
        for item in set_list:
            if item in patient_path:
                patient = item + '_' + patient_path.split('/')[-2]          
        print("\nPatient: '%s'" % patient)
        if verboseLevel > 1:
            print('patient_path:', patient_path)
              
        # Read images related to the current patient
        X_test, Y_test, img_names_test = read_images(patient_path, classes, dimInput, grayLevels, img_type, \
                                                     standardize_images, filterImg, sigmaFilter)
        print('Patient data shape: ', X_test.shape)
        print('Patient labels shape: ', Y_test.shape)

        # Count the slices
        if len(Y_test[0]) > 1:
            temp = np.array(Y_test[:,1])
        else:
            temp = np.array(Y_test)
        if len(classes) == 2:   
            for i in range(len(classes)):                
                print("number of labels '%s' for the Patient '%s': %d" % \
                      (classes[i], patient, len(temp[temp==i])))

        # Calculate predictions and related Metrics
        model_preds = model.predict(X_test, batch_size=1, verbose=1)
        preds = model_preds.ravel()
        labels = np.rint(Y_test.ravel())
        preds_pos = model_preds[:,1] 
        if model_preds[0].shape[0] > 1:
            labels_int = np.argmax(Y_test, axis=-1)
        else:
            labels = np.transpose(Y_test.ravel())
            labels_int = np.rint(labels)
      
        # Patient-based ROC (FROC)
        if microAverageFROC:
            fpr, tpr, fp, cm = custom_ROC(preds, labels)
        else: #I consider the probability of positivity only!
            fpr, tpr, fp, cm = custom_ROC(preds_pos, labels_int) 
        fpr = [x for _,x in sorted(zip(fp, fpr))]
        tpr = [x for _,x in sorted(zip(fp, tpr))]
        fp = sorted(fp)
        results[patient] = [fpr, tpr, fp, cm.tolist()]
        if verboseLevel > 2:
            print(results[patient])
                
        # Plot patient FROC
        ls_mean = '-'
        label_mean = ''
        lw_mean = lwp
        if len(dataset_structure[ids].split('/')) > 2:
            if Npatients < 50 and np.array(fp).any() != 0:
                plt.plot(results[patient][2], results[patient][1], lw=lwp, label="patient '%s'" % patient)
            ls_mean = '--'
            label_mean = 'mean'
            lw_mean = lwp*1.5
    
    # Aggregate results and interpolate all FROC curves at FP_eval points
    print("\n")
    #FP_eval = np.unique(np.concatenate([results[i][2] for i in results.keys() if len([item for item in results[i][2] if item == 0]) < len(results[i][2])])) 
    #np.interp gives errors with this new FP_eval. Use the default one!
    TPR_mean = np.zeros_like(FP_eval)
    if includeFPnull:
        for i in results.keys():
            TPR_mean += np.interp(FP_eval, results[i][2], results[i][1])
            Npatients_calc += 1      
    else:
        for i in results.keys():
            if len([item for item in results[i][2] if item == 0]) < len(results[i][2]):
                TPR_mean += np.interp(FP_eval, results[i][2], results[i][1])
                Npatients_calc += 1
            else:
                print("patient", i, "not considered for the average, since fp.any()==0!")
    if setNullTPR0:
        TPR_mean[0] = 0.
            
    # mean result
    print("Npatients:", Npatients)
    print("Npatients_calc:", Npatients_calc)
    TPR_mean = TPR_mean / Npatients_calc
    
    # Write mean FROC file
    output_file = figuresDir + '../mean_test_FROC.txt'
    with open(output_file, 'w') as f:
        f.write('FP, TPR\n')
        for i in range(len(FP_eval)):
            f.write('%.2f, %.8f\n' % (FP_eval[i], TPR_mean[i]))
    print('\nmean FROC file written!')
    
    # Write results for each patient to file
    if Npatients < 120:
        output_file = figuresDir + '../results_test_patients.txt'
        with open(output_file, 'w') as f:
            f.write(json.dumps(results))
        print('Results file for patients written!')    
        
    # Plot mean result
    plt.plot(FP_eval, TPR_mean, color='black', lw=lw_mean, linestyle=ls_mean, label=label_mean)
    plt.xlabel('Mean FPs per DBT volume', fontsize=fs)
    plt.ylabel('Sensitivity (TPR)', fontsize=fs)
    plt.xticks(fontsize=fs, rotation=0)
    plt.yticks(fontsize=fs, rotation=0)
    plt.xlim([0, FP_eval.max()*1.])
    plt.ylim([0.0, 1.1])
    plt.title('Volume-based FROC', fontsize=fs)
    plt.legend(loc="lower right")
    plt.savefig(figuresDir + 'PB_FROC_test.jpg')
    if not MPLBACKEND == 'Agg':
        plt.show()
    plt.close()
    
    # End message
    end = time.time()
    print("\nCalculation ended, time elapsed: %.0f s\n" % (end - start))
else:
    print("Standard anlaysis")
    if not unlabeled_dataset:
        X_test, Y_test, img_names_test = read_images(Path_test, classes, dimInput, grayLevels, img_type, \
                                                     standardize_images, filterImg, sigmaFilter)
        print('Test data shape: ', X_test.shape)
        print('Test labels shape: ', Y_test.shape)

        # Count the samples in each class     
        if len(classes) == 2:
            temp = np.array(Y_test[:,1]) 
            for i in range(len(classes)):
                print("number of labels '%s' in the dataset: %d" % \
                      (classes[i], len(temp[temp==i])))
        else:
            temp = np.array(Y_test) 
            Nneg = 0
            Npos = 0
            Nben = 0
            for item in temp:
                if ([1, 0, 0] == item).all():
                    Nneg = Nneg + 1
                elif ([0, 1, 0] == item).all():
                    Npos = Npos + 1
                else:
                    Nben = Nben + 1
            Ncases = [Nneg, Npos, Nben]
            for i in range(len(classes)):
                print("number of labels '%s' in the Test dataset: %d" % (classes[i], Ncases[i]))
    else:
        X_test, img_names_test = read_unlabeled_images(Path_test, dimInput, grayLevels, img_type, \
                                                       standardize_images, filterImg, sigmaFilter)    
    
    # Plot the histogram of the Test dataset classes
    if not unlabeled_dataset:
        plot_histogram(Y_test, classes, figuresDir, "red", "test", MPLBACKEND=MPLBACKEND)

    ## Calculate the predictions and related Metrics
    if not unlabeled_dataset:
        if len(classes) == 2:
            preds_test, labels_test, results_test = evaluate_model_2classes(model, X_test, Y_test, classes, 
                                                                            figuresDir, MPLBACKEND=MPLBACKEND)
        else:
            preds_test, labels_test, results_test = evaluate_model(model, X_test, Y_test, classes, 
                                                                   figuresDir, MPLBACKEND=MPLBACKEND)
        print('\n')
    else:
        model_preds = model.predict(X_test, batch_size=1, verbose=1)
        preds = np.transpose(model_preds.ravel())
        preds_test = np.argmax(model_preds, axis=-1)
        Y_test = np.nan

    ## Calculate the Gradient Maps (GradCAMs)
    if calculate_gradcam:
        print('calculating the GradCAMs...')
        
        pure_img_names_test = [item.split('/')[-1].split('.')[0] for item in img_names_test]

        if gradcam4all:
            gradcam_tensor, gradcams_img_names = GradCAM_calculation_all(dataset_path, dataset_structure, classes, img_type, 
                                                                         model, X_test, Y_test, pure_img_names_test, GradCAMsDir, 
                                                                         saveGradcam2Text, sn=True, nf=0.0015, MPLBACKEND=MPLBACKEND)
        else:
            gradcam_tensor, gradcams_img_names = GradCAM_calculation(dataset_path, dataset_structure, classes, img_type, model, 
                                                                     X_test, Y_test, pure_img_names_test, GradCAMsDir, 
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

