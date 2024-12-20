##########################################################################
# Author: Gianfranco Paternò (paterno@fe.infn.it)                        #
# collaboration of INFN Ferrara and INFN Napoli for the next_AIM project #
# Last modification date: March 2024                                     #
#                                                                        #
# SPDX-License-Identifier: EUPL-1.2                                      #
# Copyright 2024 Istituto Nazionale di Fisica Nucleare                   #
##########################################################################


import os
from glob import glob


def GetDataNamesAndLabels(path, classes, img_type):
    """
    Function that returns the data in the subfolders 
    in a given path, and the corresponding labels.
    """
    
    if 'train' in path:
        set_type = 'training'
    else:
        set_type = 'validation'
        
    data = []
    labels = {}
    n_img_classes = []
    
    for cls in classes:
        # Get all the image filenames contained in a certain folder
        img_str = path + str(cls[0].upper() + cls[1:]) + '/*' + img_type
        fnames = glob(os.path.join(img_str))    
        if len(fnames) == 0:
            img_str = path + str(cls[0].lower() + cls[1:]) + '/*' + img_type
            fnames = glob(os.path.join(img_str))    
        n_img_classes.append(len(fnames))
        print("number of labels '%s' in the %s dataset managed by the DataGenerator: %d" % (str(cls), set_type, len(fnames)))
        
        # Encode the labels
        if len(classes) == 2:
            if isinstance(cls, str):    
                if cls.casefold() == 'negative':
                    cls = [1, 0]
                elif cls.casefold() == 'positive':
                    cls = [0, 1]                     
        else:
            if isinstance(cls, str):    
                if cls.casefold() == 'negative':
                    cls = [1, 0, 0]
                elif cls.casefold() == 'positive':
                    cls = [0, 1, 0]
                else:
                    cls = [0, 0, 1]
        
        # Populate the data list and the labels dictionary
        for fname in fnames:   
            data.append(fname)
            labels[fname] = cls       
    
    #return
    return data, labels
