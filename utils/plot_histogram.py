##########################################################################
# Author: Gianfranco Paternò (paterno@fe.infn.it)                        #
# collaboration of INFN Ferrara and INFN Napoli for the next_AIM project #
# Last modification date: March 2024                                     #
#                                                                        #
# SPDX-License-Identifier: EUPL-1.2                                      #
# Copyright 2024 Istituto Nazionale di Fisica Nucleare                   #
##########################################################################


import numpy as np
import matplotlib.pyplot as plt


def plot_histogram(mydata, classes, figuresDir, color, data_type, OneHotCod=True, MPLBACKEND=''):
    """
    Function to plot the histogram of the classes in a dataset
    """
    
    if OneHotCod:
        target = [float(i) for i in np.argmax(mydata, axis=-1)]
    else:
        target = [float(i) for i in mydata]
        
    if len(classes)==2:
        Nbins = 2
        xticksPos = [0.25, 0.75]
    else:
        Nbins = 3
        xticksPos = [0.33, 1, 1.66]
    
    plt.figure(figsize=(7, 7))
    plt.hist(target, bins=Nbins, color=color, edgecolor='black', linewidth=1.2)
    plt.title("Histogram of " + data_type + " data")
    plt.xticks(xticksPos, classes, rotation=0)
    plt.xlabel(r"")
    plt.ylabel("Frequency")
    plt.savefig(figuresDir + 'histogram_' + data_type + '.jpg')
    if not MPLBACKEND == 'Agg':
        plt.show()  
    plt.close()
