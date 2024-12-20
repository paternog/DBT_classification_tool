##########################################################################
# Author: Gianfranco Paternò (paterno@fe.infn.it)                        #
# collaboration of INFN Ferrara and INFN Napoli for the next_AIM project #
# Last modification date: March 2024                                     #
#                                                                        #
# SPDX-License-Identifier: EUPL-1.2                                      #
# Copyright 2024 Istituto Nazionale di Fisica Nucleare                   #
##########################################################################


# Prepare the dataset to be analyzed with a CNN by carring out a proper splitting
# in training and test classes and a sampling of all the available slices.


# Import the required libraries
import os, shutil
import random
import numpy as np
import json
from tqdm import tqdm
import collections


# Input
###############################################################################################
# Define the data structure
dataset_path = '/media/paterno/SSD1TB/'
dataset_structure = ['DUKE_fully_processed_sample_dataset/']
#classes = ('negative', 'positive')
classes = ('negative', 'positive', 'benign')
img_type = '.tiff'

# Options for the dataset creation
NslicesPerPatient = 10
random_sel = False
ReadFromFile = False 
filepath = dataset_path + 'selected_slices/' #where to save or read the selected slices
file_index_str = '' #'' or '_i' with i=1,...,n (in the output pathname)
suffix_to_remove = '/' #in the output pathname
sel_all = True #it has the priority
Ntest_cases_per_class = 6 #randomly selected
UseAllTestSlices = False
DisjointedTestSubSets = True
NattemptMax = 1e5 #for creating the subset defined by file_index_str

# Options
verboseLevel = 1
testMode = False
###############################################################################################


# Check
if sel_all:
    random_sel = False
    ReadFromFile = False
    DisjointedTestSubSets = False
    
# Create the directory where to save or read (if ReadFromFile is True) the selected slices
if not os.path.exists(filepath):
    os.makedirs(filepath)

# Create the path of input directory
inputDir = dataset_path + dataset_structure[0]

# Create the path of train and test subdirectories
sel_mod = 'SpGridSel'
if random_sel:
    sel_mod = 'SpRndSel'
if sel_all:
    sel_mod = 'all'
    mod = sel_mod + str(Ntest_cases_per_class) + 'PcTest' + file_index_str
else:    
    mod = str(NslicesPerPatient) + sel_mod + str(Ntest_cases_per_class) + 'PcTest' + file_index_str
SubDir = dataset_structure[0].removesuffix(suffix_to_remove)
trainSubDir = SubDir + '_' + mod + '_train/'
testSubDir = SubDir + '_' + mod + '_test/'

# Create the training directories
trainDir = dataset_path + trainSubDir
trainDirNeg = trainDir + classes[0]
trainDirPos = trainDir + classes[1]
if not os.path.exists(trainDir):
    os.makedirs(trainDir)
if not os.path.exists(trainDirNeg):
    os.makedirs(trainDirNeg)
if not os.path.exists(trainDirPos):
    os.makedirs(trainDirPos) 

# Create the test directories
testDir = dataset_path + testSubDir
testDirNeg = testDir + classes[0]
testDirPos = testDir + classes[1]
if not os.path.exists(testDirNeg):
    os.makedirs(testDirNeg)
if not os.path.exists(testDirPos):
    os.makedirs(testDirPos) 

# Create Benign directories in case of a three classes dataset
if len(classes) > 2:
    trainDirBen = trainDir + classes[2]
    testDirBen = testDir + classes[2]
    if not os.path.exists(trainDirBen):
        os.makedirs(trainDirBen)  
    if not os.path.exists(testDirBen):
        os.makedirs(testDirBen)


# Identify the patients per class
patients = {}
for cls in classes:
    dirpath = inputDir + str(cls)
    files = os.listdir(dirpath)
    Nfiles = len(files)
    print("\nnumber of files in the %s directory: %d" % (str(cls), Nfiles))   
    temp = set([str(file).split('-')[1] for file in files])
    patients[cls] = sorted(list(temp))
    print("number of patients in this class: %d" % len(patients[cls]))
    if verboseLevel > 0:
        print(patients[cls],'\n')
    
# Check
if len(classes) == 2:
    for pp in patients[classes[1]]:
        if pp in patients[classes[0]]:
            print("patient %s found in both classes" % pp)
else:
    for cls in classes:
        classes_red = list(classes)
        classes_red.remove(cls)
        for pp in patients[cls]:
            for item in classes_red:
                if pp in patients[item]:
                    print("patient %s found in more than one class" % pp)

                    
# Identify the projections per patient (classes unified)
proj_per_patient = {}
for cls in classes:
    dirpath = inputDir + str(cls)
    files = os.listdir(dirpath) 
    for file in files:
        patient = str(file).split('-')[1]
        projection = str(file).split('-')[2]        
        if not patient in proj_per_patient.keys():
            proj_per_patient[patient] = [projection]
        else:
            if not projection in proj_per_patient[patient]:
                proj_per_patient[patient].append(projection)
proj_per_patient = collections.OrderedDict(sorted(proj_per_patient.items()))
if verboseLevel > 1:
    print('proj_per_patient (len = %d):' % len(proj_per_patient))
    print(proj_per_patient,'\n')
    

# Identify the slices per patient-projection combination (classes unified)
slices_per_patient_proj = {}
for cls in classes:
    dirpath = inputDir + str(cls)
    files = os.listdir(dirpath)    
    for file in files:        
        patient = str(file).split('-')[1]
        projection = str(file).split('-')[2]
        Slice = str(file).split('-')[3].split('.')[0]
        key = str(patient) + '-' + str(projection)
        if not key in slices_per_patient_proj.keys():
            slices_per_patient_proj[key] = [Slice]
        else:
            if not Slice in slices_per_patient_proj[key]:
                slices_per_patient_proj[key].append(Slice)
slices_per_patient_proj = collections.OrderedDict(sorted(slices_per_patient_proj.items()))
if verboseLevel > 1:
    print('number of different patient-projection combinations:', len(slices_per_patient_proj))
    print(slices_per_patient_proj.keys(),'\n')
if verboseLevel > 2:
    print(slices_per_patient_proj,'\n')
    
    
# Identify the the patient-projection combination for each class
patient_proj_class = {}
for cls in classes:
    temp = []
    for key in slices_per_patient_proj.keys():
        patient = key.split('-')[0]
        if patient in patients[cls]:
             temp.append(key)
    patient_proj_class[cls] = temp
    print('\nnumber of different patient-projection combinations in the %s class: %d' % (cls, len(patient_proj_class[cls])))
    if verboseLevel > 0:  
        print(patient_proj_class[cls],'\n')
    else:
        print('\n')

    
# Select (randomly or not) a subset of slices to consider
print('\nselecting a subset of files to copy from the input to the training directory...') 
json_file = filepath + 'selected_slices_' + mod + '.json'
if not ReadFromFile:
    if not random_sel:
        print('determining the files to copy with a deterministic linear decimation') 
    else:
        print('determining the files to copy with a random sampling') 
    temp_sel = []
    selected_slices = {}
    for key in slices_per_patient_proj.keys():
        temp = slices_per_patient_proj[key]
        temp.sort()
        Nslices = len(temp) 
        if verboseLevel > 1:
            print(key)
        if sel_all:
            NslicesPerPatient = Nslices
        if not random_sel:
            sel_indices = np.linspace(0, Nslices, NslicesPerPatient, dtype=int, endpoint=False)
        else:
            sel_indices = random.sample(list(range(Nslices)), NslicesPerPatient)
        if verboseLevel > 1:
            print('selected:', sel_indices)
        temp_sel = [temp[i] for i in list(sel_indices)] 
        selected_slices[key] = temp_sel
else:
    print('reading files to copy from the file:', json_file) 
    with open(json_file) as f:
        data = f.read()
        js = json.loads(data)
    selected_slices = js


# Select (randomly) Ntest_cases_per_class cases (patients) in each class 
# to use as test and move these images in the test directory
print("selecting (randomly) the test patients")
if DisjointedTestSubSets: 
    print("disjointed mode for Test subsets activated\n")
else:
    print("\n")
selected_ID = {}
file_test_cases = filepath + 'selected_test_cases_' + mod[:-2] + '_' + str(len(classes)) + 'classes' + '.json'
test_cases = []
for cls in classes:
    json_file_test = filepath + 'selected_slices_' + mod + '_' + str(cls) + '_test_cases' + '.json' 
    if not ReadFromFile:
        if not DisjointedTestSubSets: 
            selected_ID[cls] = random.sample(patients[cls], Ntest_cases_per_class)
            if os.path.exists(file_test_cases):
                os.remove(file_test_cases)    
        else:
            if not os.path.exists(file_test_cases):
                test_cases = [patient for patient in random.sample(patients[cls], Ntest_cases_per_class)] 
                selected_ID[cls] = test_cases         
            else:
                with open(file_test_cases) as f:
                    data = f.read()
                    js = json.loads(data)
                test_cases = js
                try:
                    test_cases_cls = []
                    Nsel = 0
                    Nattempt = 0
                    while Nsel < Ntest_cases_per_class:
                        patient = random.sample(patients[cls], 1)[0]
                        if patient not in test_cases:
                            test_cases.append(patient)
                            test_cases_cls.append(patient)
                            Nsel += 1
                        Nattempt +=1
                        if Nattempt > NattemptMax:
                            raise Exception("Max number of attempts (%0.e) for subset composition reached. Execution aborted!\n" % NattemptMax)
                finally:
                    print("")                 
                selected_ID[cls] = test_cases_cls
            if not os.path.exists(json_file_test):
                test_cases.sort()
                with open(file_test_cases, 'w') as f:
                    json.dump(test_cases, f)
            else:
                print('selected_test_cases.json not updated!')
    else:
        with open(json_file_test) as f:
            data = f.read()
            js = json.loads(data)
        selected_ID[cls] = js        
    if len(patients[cls]) > 0:
        selected_ID[cls] = sorted(selected_ID[cls])
        print('selected patients of', cls, 'class:', selected_ID[cls])
        
    # number of patient-projection combinations selected for test
    Npct = 0         
    for item in patient_proj_class[cls]:
        patient = patient = item.split('-')[0]
        if patient in selected_ID[cls]:
            Npct = Npct + 1
    if verboseLevel > 0:
        print('number of patient-projection combinations in the %s class selected for test: %d' % (cls, Npct))
        

# Correct selected slices in case I want to UseAllTestSlices
projections = ['LMLO', 'RMLO', 'LML', 'RML', 'LCC', 'RCC']
if UseAllTestSlices:
    for cls in selected_ID.keys():  
        for patient in selected_ID[cls]:
            for proj in projections:
                key = str(patient) + '-' + str(proj)
                if key in slices_per_patient_proj.keys():
                    selected_slices[key] = slices_per_patient_proj[key]

    
# Print/Write actual selected slices for test/training
if not ReadFromFile:
    if verboseLevel > 2:
        print('selected_slices:', selected_slices)
    with open(json_file, 'w') as f:
        json.dump(selected_slices, f)        
nf = 0
for key in selected_slices.keys():
    nf += len(selected_slices[key])
print('number of selected slices:', nf)
if verboseLevel > 2:
    print(selected_slices)


# Create the training dataset with the selected files (slices)
print('\ncopying selected files from the input to the training directory...')
NfileCopied = 0
for cls in classes:
    NfileToCopy = 0
    trainpath = trainDir + str(cls) + '/'
    if len(os.listdir(trainpath)) == 0: 
        print('%s directory' % cls)  
        dirpath = inputDir + str(cls)
        files = os.listdir(dirpath)    
        for file in tqdm(files): 
            patient = str(file).split('-')[1]
            projection = str(file).split('-')[2]
            Slice = str(file).split('-')[3].split('.')[0]
            key = str(patient) + '-' + str(projection)
            if Slice in selected_slices[key]:
                origin_path = inputDir + str(cls) + '/' + file
                destination_path = trainDir + str(cls) + '/' + file
                NfileToCopy += 1 
                if not testMode:
                    try:
                        shutil.copyfile(origin_path, destination_path)
                        if verboseLevel > 2:
                            print("file copied:", file)
                        NfileCopied += 1
                    except Exception:
                        print('file %s not copied!' % str(file))
                        pass
    else:
        print('directory %s is not empty!' % trainpath)
    print("number of files of class %s than could be copied: %d" % (str(cls), NfileToCopy))  
print('%d images copied in the training directory' % NfileCopied)    


# Move the files related to test patients in the test directory
print("\nmoving files in the test directory...")
if UseAllTestSlices:
    print("\nfiles will be redundantly arranged both for class and patient")
NfileMoved = 0
selected_for_test = {}
for cls in classes:    
    selected = []
    selected_for_test_cls = []
    for patient in selected_ID[cls]:
        for proj in proj_per_patient[patient]:
            key = str(patient) + "-" + str(proj)
            if verboseLevel > 1:
                print('patient-projection combination to move:', key)
            for item in selected_slices[key]:
                file = 'DBT-' + str(patient) + "-" + str(proj) + "-" + str(item) + img_type
                selected.append(file)
        if verboseLevel > 2:
            print(selected)
        selected_for_test_cls.append(selected)
           
    selected_for_test_cls2 = [item for sublist in selected_for_test_cls for item in sublist] 
    selected_for_test_cls2.sort()     
            
    selected_for_test[cls] = set(selected_for_test_cls2)   
    print("number of files of class %s to move: %d" % (str(cls), len(selected_for_test[cls])))
    if verboseLevel > 1:
        print(selected_for_test[cls])
        
    json_file_test = filepath + 'selected_slices_' + mod + '_' + str(cls) + '_test_cases' + '.json'
    if not ReadFromFile:
        with open(json_file_test, 'w') as f:
            json.dump(list(selected_ID[cls]), f)
    else:
        if testMode:
            json_file_test = json_file_test.removesuffix('.json') + '_' + dataset_structure[0].removesuffix(suffix_to_remove) + '.json'
            with open(json_file_test, 'w') as f:
                json.dump(list(selected_ID[cls]), f)
     
    for file in tqdm(selected_for_test[cls]):
        origin_path = str(trainDir) + str(cls) + '/' + str(file)      
        copy_path = str(testDir) + str(cls)
        if not UseAllTestSlices:
            destination_path = copy_path
        else:
            patient = file.split('-')[1].split('.')[0]
            destination_path = str(testDir) + str(patient) + '/' + str(cls) 
        if not os.path.exists(destination_path):
            os.makedirs(destination_path)
        destination_path = destination_path + '/' + str(file)
        copy_path = copy_path + '/' + str(file)
        if not testMode:
            if UseAllTestSlices:
                shutil.copyfile(origin_path, copy_path) 
            shutil.move(origin_path, destination_path)
            NfileMoved += 1
            if verboseLevel > 2:
                print("file moved:", file)
        else:
            if verboseLevel > 2:
                print("file to copy:", file)
        
print('%d images moved in the test directory' % NfileMoved)
if NfileMoved > 0:
    test_train_ratio = NfileMoved/(NfileCopied-NfileMoved)
    print('test/train ratio = %.2f\n' % test_train_ratio)
else:
    print('\n')
