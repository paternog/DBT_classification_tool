##########################################################################
# Author: Gianfranco Paternò (paterno@fe.infn.it)                        #
# collaboration of INFN Ferrara and INFN Napoli for the next_AIM project #
# Last modification date: March 2024                                     #
#                                                                        #
# SPDX-License-Identifier: EUPL-1.2                                      #
# Copyright 2024 Istituto Nazionale di Fisica Nucleare                   #
##########################################################################


# Prepare data for the so called "per patient analysis".


# Import the required libraries
import os, shutil
import random
import numpy as np
import json
from tqdm import tqdm


# Input
###############################################################################################
# Define the data path
json_path = '/media/paterno/SSD1TB/selected_slices/'
set_type = 'selected_slices_all6PcTest'
classes = ('negative', 'positive')
inputPath = '/media/paterno/SSD1TB/'
datasets = ['DUKE_fully_processed_sample_dataset_all6PcTest_test'] #Do not put '/' at the end!

verboseLevel = 1
testMode = False
###############################################################################################


# Read the json file with the data names
selected_slices_test = {}
for cls in classes:
    test_json_file = json_path + set_type + '_' + str(cls) + '_test_cases.json'
    print('reading json file:', test_json_file)
    with open(test_json_file) as f:
        data = f.read()
        js = json.loads(data)
    selected_slices_test[cls] = js
    if verboseLevel > 0:
        print(selected_slices_test[cls])

        
# Read files from original directories
print('\ncopying selected files from the input to the output (test) directory...')
NfileCopied = 0
for item in datasets:
    inputDir = inputPath + item + '/'
    outputDir = inputDir + 'patients/case_' + set_type[-1] +'/'
    print(outputDir)
    patients = {}   
    for cls in classes:
        dirpath = inputDir + str(cls)
        files = os.listdir(dirpath)
   
        # copy all the slices for the selected patients (read from the json file)
        for file in tqdm(files):
            if 'IFO' in file:
                patient = str(file).split('_')[0].split('IFO')[1]
            elif 'CARD' in file:
                patient = file.split('-')[0].split('CARD')[1]
            else:
                patient = file.split('-')[1].split('.')[0]
            if patient in selected_slices_test[cls]:
                origin_path = dirpath + '/' + file
                destination_path = outputDir + str(patient) + '/' + str(cls)
                if not os.path.exists(destination_path):
                    os.makedirs(destination_path)
                destination_path = destination_path + '/' + str(file)    
                if not testMode:
                    try:
                        shutil.copyfile(origin_path, destination_path)
                        if verboseLevel > 1:
                            print("file copied:", file)
                        NfileCopied += 1
                    except Exception:
                        print('file %s not copied!' % str(file))
                        pass
print('%d images copied in the output directory\n' % NfileCopied)
