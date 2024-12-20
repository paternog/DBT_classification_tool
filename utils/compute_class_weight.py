##########################################################################
# Author: Gianfranco Paternò (paterno@fe.infn.it)                        #
# collaboration of INFN Ferrara and INFN Napoli for the next_AIM project #
# Last modification date: March 2024                                     #
#                                                                        #
# SPDX-License-Identifier: EUPL-1.2                                      #
# Copyright 2024 Istituto Nazionale di Fisica Nucleare                   #
##########################################################################


# Compute the class weights of a given dataset, which could be very unbalanced.

"""
y_ints = np.array([y.argmax() for y in Y_train])
class_weights = class_weight.compute_class_weight(class_weight='balanced',
                                                  classes=np.unique(y_ints),
                                                  y=y_ints)
class_weights = class_weights/sum(class_weights)
"""   

def compute_class_weight(Ny_per_class, mu=0):
    """
    It is equivalent to sklearn.utils.class_weight.compute_class_weight
    if mu, which is a paramater to tune, is = 0 (default).
    Otherwise, mu is a parameter to tune.
    The use of sklearn.utils method is shown above.
    """
    import math
    import numpy as np
    Nclasses = len(Ny_per_class)
    total = np.sum(Ny_per_class)
    class_weight = []  
    for i in range(Nclasses):
        if mu == 0:
            score = total/float(Ny_per_class[i])
        else:
            score = math.log(mu*total/float(Ny_per_class[i]))
        score = score if score > 1.0 else 1.0
        class_weight.append(score)
    return class_weight/sum(class_weight)
