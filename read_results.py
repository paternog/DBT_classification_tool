##########################################################################
# Author: Gianfranco Paternò (paterno@fe.infn.it)                        #
# collaboration of INFN Ferrara and INFN Napoli for the next_AIM project #
# Last modification date: March 2024                                     #
#                                                                        #
# SPDX-License-Identifier: EUPL-1.2                                      #
# Copyright 2024 Istituto Nazionale di Fisica Nucleare                   #
##########################################################################


# Read the results provided by trained models. 
# It is typically used to analyze the results obtained on subsets (folds) of a given datasets.


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
set_list_legend = ['1', '2', '3', '4', '5']
add_to_all_pre = 'fully_processed_'
add_to_all_post = '_ResNet18_TL_300x300_aug'
file_to_open = 'results_test.txt'

remove_TPR = True
do_plots = True
save_analysis = True
save_path = dataset_path
save_filename = 'results_' + add_to_all_pre.removesuffix('_') + add_to_all_post
###############################################################################################


# Read data (dictionaries) from files
set_list = [add_to_all_pre + item + add_to_all_post for item in set_list_legend]
data_set = {} # it will be a dictionary with set names as keys 
for item in set_list:
    filename = dataset_path + item + '/' + file_to_open
    with open(filename) as f:
        data = f.read()
    js = json.loads(data)
    keys = list(js.keys())
    temp = []
    for key in keys:
        temp.append(js[key])
    data_set[item] = temp
    if remove_TPR:
        if len(keys) == 9: #3 classes
            itpr = 1
        else: #2 classes
            itpr = 6
        data_set[item].pop(itpr)        
if remove_TPR:    
    keys.pop(itpr)  

    
# plot data (bar plots)
if do_plots and len(set_list) <= 4:
    y = np.linspace(0., 1., 6) # for yticks
    x = np.arange(len(keys)) # the label locations
    width = 0.20 # the bar width
    strtitle = 'Comparison of different datasets'
    if len(set_list) == 4:
        coeff = [-2/4, 2/4, 6/4, 10/4] # used to position the bars
        bars_shift = 0.20 # shift all the bars
    if len(set_list) == 3:
        coeff = [-2/3, 1/3, 4/3]
        bars_shift = 0.075
    elif len(set_list) == 2:
        coeff = [-1/2, 1/2]
        bars_shift = 0.
    else:
        coeff = [0.]
        bars_shift = -0.0075
        strtitle = ''

    fig, ax = plt.subplots(figsize=[12, 8])
    fs = 16

    bar_groups = []
    for i, Set in enumerate(set_list):
        bar_loc = x + width*coeff[i] - bars_shift
        bar_groups.append(ax.bar(bar_loc, data_set[Set], width, label=set_list_legend[i]))

    ax.set_ylabel('Score', fontsize=fs)
    ax.set_xticks(x, keys, fontsize=fs, rotation=0)
    ax.set_yticks(y, fontsize=fs, rotation=0)
    ax.yaxis.set_tick_params(labelsize=fs, rotation=0)
    #ax.yaxis.grid(True)
    ax.set_ylim([0., 1.3])
    ax.set_title(strtitle, fontsize=fs)
    ax.legend(fontsize=fs, loc='upper right')

    #for i, Set in enumerate(rects):
        #ax.bar_label(bar_groups[i], padding=3)

    fig.tight_layout()
    if save_analysis:
        plt.savefig(save_path + save_filename + '.jpg')
    if not MPLBACKEND == 'Agg':
        plt.show()
    plt.close()


# calculate the statistics on the metrics
results = {}
for j, key in enumerate(keys):
    temp = []
    for Set in set_list:
        temp.append(data_set[Set][j]) 
    results[key] = temp  
#print(results,'\n')

import statistics

metrics_stat = {}
for key in keys:
    mean = round(statistics.mean(results[key]), 3)
    stdev = round(statistics.stdev(results[key]), 3)
    metrics_stat[key] = [mean, stdev]
    metrics_str = '%s = %.2f +/- %.2f' % (key, mean, stdev)
    print(metrics_str)
print('\n')

if save_analysis:
    metrics_stat_file = save_path + save_filename     
    with open(metrics_stat_file + '.txt', 'w') as f:
        f.write('metrics statistics:\n')
        for key in keys:
            metrics_str = '%s = %.2f +/- %.2f' % (key, metrics_stat[key][0], metrics_stat[key][1])
            f.write(metrics_str + '\n')
    with open(metrics_stat_file + '.json', 'w') as f:
        json.dump(metrics_stat, f)
