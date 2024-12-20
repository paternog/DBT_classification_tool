##########################################################################
# Author: Gianfranco Paternò (paterno@fe.infn.it)                        #
# collaboration of INFN Ferrara and INFN Napoli for the next_AIM project #
# Last modification date: March 2024                                     #
#                                                                        #
# SPDX-License-Identifier: EUPL-1.2                                      #
# Copyright 2024 Istituto Nazionale di Fisica Nucleare                   #
##########################################################################


# Read and plot the analysis of results.
# It is typically used to compare the reuslts provided by different models.


# Import the required libraries
import numpy as np
import matplotlib.pyplot as plt
import os
import json

os.environ['MPLBACKEND'] = 'Agg' 
MPLBACKEND = os.environ['MPLBACKEND'] 
print('MPLBACKEND:', MPLBACKEND, '\n')


# Input
###############################################################################################
# set data to read
dataset_path = '/media/paterno/Verbatim HDD/datasets/Database_DBT/Database_unified_1/sampling_1/python_output_ResNet18_TL/'
set_list_legend = ['ResNet18', 'VGG16']
add_to_all_pre = 'results_fully_processed_'
add_to_all_post = '_TL_300x300_aug'
file_to_open = '.json'

remove_accuracy_TPR = True
do_plots = True
save_analysis = True
save_path = dataset_path
save_filename = 'analysis_' + add_to_all_pre.removesuffix('_') + add_to_all_post
###############################################################################################


# Read data (dictionaries) from files
set_list = [add_to_all_pre + item + add_to_all_post for item in set_list_legend]
data_set_mean = {} # it will be a dictionary with set names as keys 
data_set_stdev = {} # it will be a dictionary with set names as keys 
for item in set_list:
    filename = dataset_path + item + file_to_open
    with open(filename) as f:
        data = f.read()
    js = json.loads(data)
    keys = list(js.keys())
    temp0 = []
    temp1 = []
    for key in keys:
        temp0.append(js[key][0])
        temp1.append(js[key][1])
    data_set_mean[item] = temp0
    data_set_stdev[item] = temp1
    if remove_accuracy_TPR:
        if len(keys) == 9: #3 classes full
            iacc = 3 # it is 4-1, since I'll remove the 1st element of the list  
            data_set_mean[item].pop(1) #remove TPR
            data_set_stdev[item].pop(1)
        elif len(keys) == 8:
            if "TPR" in keys: #2 classes full
                iacc = None
                data_set_mean[item].pop(6) #remove TPR
                data_set_stdev[item].pop(6)
            else: #3 classes without TPR    
                iacc = 3
        else: #2 without TPR
            iacc = None
        if iacc: #remove accuracy only in the three classes case
            data_set_mean[item].pop(iacc)
            data_set_stdev[item].pop(iacc)
if remove_accuracy_TPR:
    if iacc:
        keys.pop(iacc)
    if "tpr" in keys:
        keys.remove("tpr")  
    if "TPR" in keys:
        keys.remove("TPR")  
    
    
# print data
print('keys:', keys, '\n')
print('mean_values:', data_set_mean, '\n')
print('stdev_values:', data_set_stdev, '\n')


# save data
if save_analysis:
    with open(save_path + save_filename + '.txt', 'w') as f:
        f.write('keys: ')
        f.write(json.dumps(keys))
        f.write('\nmean_values: ')
        f.write(json.dumps(data_set_mean))
        f.write('\nstdev_values: ')
        f.write(json.dumps(data_set_stdev))

        
# plot data (bar plots)
if do_plots and len(set_list) <= 4:
    y = np.linspace(0., 1., 6) # for yticks
    x = np.arange(len(keys)) # the label locations
    width = 0.20  # the bar width
    if len(set_list) == 4:
        coeff = [-2/4, 2/4, 6/4, 10/4] # used to position the bars
        bars_shift = 0.20 # shift all the bars
    elif len(set_list) == 3:
        coeff = [-2/3, 1/3, 4/3]
        bars_shift = 0.075
    elif len(set_list) == 2:
        coeff = [-1/2, 1/2]
        bars_shift = 0.
    else:
        coeff = [0.]
        bars_shift = -0.0075

    fig, ax = plt.subplots(figsize=[12, 8])
    fs = 16

    bar_groups = []
    for i, Set in enumerate(set_list):
        bar_loc = x + width*coeff[i] - bars_shift
        bar_groups.append(ax.bar(bar_loc, data_set_mean[Set], width, label=set_list_legend[i], \
        #bar_groups.append(ax.bar(bar_loc, data_set_mean[Set], width, label="", \
                                 yerr=data_set_stdev[Set], align='center', alpha=0.5, \
                                 ecolor='black', capsize=5))

    ax.set_ylabel('Score', fontsize=fs)
    ax.set_xticks(x, keys, fontsize=fs, rotation=0)
    ax.set_yticks(y, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=fs, rotation=0)
    #ax.yaxis.set_tick_params(labelsize=fs, rotation=0) #use this instead of the previous one
    #ax.yaxis.grid(True)
    ax.set_ylim([0., 1.3])
    #ax.set_title('5-fold cross-validation', fontsize=fs)
    ax.set_title('', fontsize=fs)
    ax.legend(fontsize=fs, loc='upper right')

    fig.tight_layout()
    if save_analysis:
        plt.savefig(save_path + save_filename + '.jpg')
    if not MPLBACKEND == 'Agg':
        plt.show()
    plt.close()
