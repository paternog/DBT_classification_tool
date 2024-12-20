##########################################################################
# Author: Gianfranco Paternò (paterno@fe.infn.it)                        #
# collaboration of INFN Ferrara and INFN Napoli for the next_AIM project #
# Last modification date: March 2024                                     #
#                                                                        #
# SPDX-License-Identifier: EUPL-1.2                                      #
# Copyright 2024 Istituto Nazionale di Fisica Nucleare                   #
##########################################################################


# Set of functions to evuluate the inference performance of a model



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.metrics import roc_curve, auc, cohen_kappa_score, classification_report, matthews_corrcoef, log_loss
from tensorflow.keras.models import Model
from itertools import cycle
import json

import utils.confusion_matrix_pretty_print as cmpp

# Set defualt parameters for plots
plt.rcParams["figure.figsize"] = [6.0, 6.0]
plt.rcParams["figure.autolayout"] = True
plt.rcParams.update({'font.size': 14})
#plt.rcParams["figure.dpi"] = 100



def custom_ROC(score, y):
    """
    Custom function to calculate (volume-based) ROC (FROC)
    NOTE: It works for 2 classes problem only.
    NOTE: y must be an integer, score must be a real, one-hot encoding is not allowed!
    """
    import numpy as np
    import sklearn
    # false positive rate
    fpr = []
    # true positive rate
    tpr = []
    # false positive list (for FROC)
    fp = []   
    # define the thresholds
    thresholds = np.arange(0.0, 1.01, .01)
    # get the number of positive and negative cases in the dataset
    P = sum(y)
    N = len(y) - P
    # iterate through all the thresholds and determine the fractions of 
    # true positives and false positives found at each threshold
    for thresh in thresholds:
        FP = 0
        TP = 0
        for i in range(len(score)):
            if (score[i] > thresh):
                if y[i] == 1:
                    TP = TP + 1
                if y[i] == 0:
                    FP = FP + 1
        if N > 0:
            fpr.append(FP/float(N))
        else:
            fpr.append(0)
        if P > 0:
            tpr.append(TP/float(P))
        else:
            tpr.append(0)
        fp.append(FP)
    # confusion matrix
    cm = sklearn.metrics.confusion_matrix(np.rint(score), y)
    # return
    return fpr, tpr, fp, cm
    


def evaluate_model_2classes(model, x, y, classes, figuresDir, set_type='test', OneHotCod=True, MPLBACKEND=''):
    """
    Function for the evaluation of a model with two classes (OneHotCod is optional)
    """
    
    # model prediction
    print("\ncalculating predictions on the %s dataset..." % set_type)
    model_preds = model.predict(x, batch_size=1, verbose=1)
    #print("\n")
    preds = model_preds.ravel()
    labels = y.ravel()    
    if OneHotCod:
        preds_int = np.argmax(model_preds, axis=-1)
        labels_int = np.argmax(y, axis=-1)  
    else:
        preds_int = np.rint(preds)
        labels_int = np.rint(labels)

    # Receiver Operating Characteristic (ROC)
    # Calculation of ROC
    fpr, tpr, th = roc_curve(labels, preds)    
    #fpr, tpr, th = roc_curve(np.rint(labels_int), model_preds[:,1]) #it gives slightly different results!
    roc_auc = auc(fpr, tpr)
    # Plot ROC
    #plt.figure(figsize=(7, 7))
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='AUC = %0.2f' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    fs = 14
    plt.xlabel('False Positive Rate', fontsize=fs)
    plt.ylabel('True Positive Rate', fontsize=fs)
    plt.xticks(fontsize=fs, rotation=0)
    plt.yticks(fontsize=fs, rotation=0)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title('Receiver operating characteristic', fontsize=fs)
    #plt.title('', fontsize=fs)
    plt.legend(loc="lower right")
    plt.savefig(figuresDir + 'ROC_' + set_type + '.jpg')
    if not MPLBACKEND == 'Agg':
        plt.show()
    plt.close()
    
    # Accuracy
    #print("calculating accuracy...")
    #loss, acc = model.evaluate(x, y, batch_size=1)
    #print("\n")
          
    # Confusion Matrix (CM)
    # Calculation of CM
    confusion_matrix = sklearn.metrics.confusion_matrix(labels_int, preds_int)
    norm_confusion_matrix = confusion_matrix.astype(float)
    for j in range(2):
        norm_confusion_matrix[j,:] = norm_confusion_matrix[j,:]/len(labels_int[labels_int==j])
    labels_cm = classes
    df_cm = pd.DataFrame(confusion_matrix, index=labels_cm, columns=labels_cm)
    #Plot CM
    cmap = plt.cm.Blues
    fz = 14
    figsize = [7, 7]
    show_null_values = 2
    pred_val_axis = 'x'
    cmpp.pretty_plot_confusion_matrix(df_cm, fz=fz, cmap=cmap, figsize=figsize, 
                                      show_null_values=show_null_values, pred_val_axis=pred_val_axis)
    plt.savefig(figuresDir + 'CM_' + set_type + '.jpg')
    if not MPLBACKEND == 'Agg':
        plt.show()
    plt.close()
    
    # Calculation of metrics from CM
    TN = confusion_matrix[0,0]
    TP = confusion_matrix[1,1]
    FN = confusion_matrix[1,0]
    FP = confusion_matrix[0,1]
    accuracy = (TN + TP) / (TN + TP + FN + FP)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 =  2 * (precision*recall) / (precision + recall)
    error = 1 - precision
    TPR = recall
    FPR = FP / (FP + TN)
    sensitivity = TPR
    specificity = 1 - FPR     
    print('*********************** Metrics ***********************')
    print('TN: %d, TP: %d, FN: %d, FP: %d' % (TN, TP, FN, FP))
    print('Accuracy ((TN + TP)/(TN + TP + FN + FP)): %0.2f' % accuracy)
    print('Precision (TP / (TP + FP)): %0.2f' % precision)
    print('Recall (TP / (TP + FN)): %0.2f' % recall)
    print('F1 score (harmonic mean of Precision and Recall): %0.2f' % F1)
    print('Error rate (1 - Precision): %0.2f' % error)
    print('TPR (Sensitivity = Recall): %0.2f' % TPR)
    print('FPR (1 - Specificity): %0.2f' % FPR)
    print('*******************************************************')
    
    # Save results to a text file
    results = {
               'fpr': fpr,
               'tpr': tpr,
               'th': th,
               'roc_auc': roc_auc,
               'accuracy': accuracy,
               'precision': precision,
               'recall': recall,
               'error': error,
               'F1': F1,
               'TPR': TPR,
               'FPR': FPR
              }
    keys_to_save = ('roc_auc', 'accuracy', 'precision', 'recall', 'error', 'F1', 'TPR', 'FPR')
    results_to_save = dict((k, results[k]) for k in keys_to_save)
    output_file = figuresDir + '../results_' + set_type + '.txt'
    with open(output_file, 'w') as f:
        f.write(json.dumps(results_to_save))
    
    # Return
    return preds_int, labels_int, results



def evaluate_model(model, x, y, classes, figuresDir, set_type='test', MPLBACKEND=''):
    """
    # Function for the evaluation of a model with multiple classes (OneHot encoded)
    """
    
    # model prediction
    print("\ncalculating predictions on the %s dataset..." % set_type)
    model_preds = model.predict(x, batch_size=1, verbose=1)
    #print("\n")
    preds_int = np.argmax(model_preds, axis=-1)
    labels_int = np.argmax(y, axis=-1)  
    
    # Number of classes and images in the dataset
    n_classes = len(classes)
    n_data = len(model_preds)
    
    # Accuracy
    #print("calculating accuracy...")
    #loss, acc = model.evaluate(x, y, batch_size=1)
    #print("\n")
        
    # Compute ROC curve and ROC area for each class 
    lw = 2    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y[:, i], model_preds[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y.ravel(), model_preds.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    #all_fpr = np.linspace(0., 1., 1000) #TO CHECK!
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    # Plot all ROC curves
    plt.figure(figsize=(9, 7))
    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
        color="deeppink",
        linestyle=":",
        linewidth=4,
    )
    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
        color="navy",
        linestyle=":",
        linewidth=4,
    )
    colors = cycle(["aqua", "darkorange", "cornflowerblue"])
    for i, color in zip(range(n_classes), colors):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=lw,
            label="ROC curve of class {0} (area = {1:0.2f})".format(classes[i], roc_auc[i]),
        )
    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("multiclass ROC")
    plt.legend(loc="lower right")
    plt.savefig(figuresDir + 'ROC_' + set_type + '.jpg')
    if not MPLBACKEND == 'Agg':
        plt.show()
    plt.close()
    #print("\n")
    
    # Confusion Matrix (CM)
    # Calculation of CM
    confusion_matrix = sklearn.metrics.confusion_matrix(labels_int, preds_int)
    norm_confusion_matrix = confusion_matrix.astype(float)
    for j in range(len(classes)):
        norm_confusion_matrix[j,:] = norm_confusion_matrix[j,:]/len(labels_int[labels_int==j])
    labels_cm = classes
    df_cm = pd.DataFrame(confusion_matrix, index=labels_cm, columns=labels_cm)
    #Plot CM
    cmap = plt.cm.Blues
    fz = 14
    figsize = [8, 8]
    show_null_values = 2
    pred_val_axis = 'x'
    df_cm_calc = df_cm.copy() #I create a copy since the following function add additional
                              #rows and columuns to df_cm if pred_val_axis = 'x'
    cmpp.pretty_plot_confusion_matrix(df_cm, fz=fz, cmap=cmap, figsize=figsize, 
                                      show_null_values=show_null_values, pred_val_axis=pred_val_axis)
    plt.savefig(figuresDir + 'CM_' + set_type + '.jpg')
    if not MPLBACKEND == 'Agg':
        plt.show()
    plt.close()
        
    # Calculation of metrics (One-Vs-Rest approach)
    #print("\n")
    print('***************************** Metrics *****************************')
    report = classification_report(labels_int, preds_int, target_names=classes)
    print(report)
    sum_diag_CM = 0
    for j in range(n_classes):
        sum_diag_CM += confusion_matrix[j,j]
    accuracy = sum_diag_CM / n_data
    print ("Accuracy: %.4f" % accuracy)
    K = cohen_kappa_score(labels_int, preds_int)
    print ("Cohen's Kappa: %.4f" % K)    
    print('*******************************************************************')
         
    # Calculate the Metrics manually from the CM dataframe
    # https://towardsdatascience.com/comprehensive-guide-on-multiclass-classification-metrics-af94cfb83fbd
    # look also at https://www.evidentlyai.com/classification-metrics/multi-class-metrics
    # and https://arxiv.org/pdf/2008.05756.pdf
    
    p0a = len(labels_int[labels_int==0])/n_data
    p0p = len(preds_int[preds_int==0])/n_data
    p0e = p0a*p0p
    p1a = len(labels_int[labels_int==1])/n_data
    p1p = len(preds_int[preds_int==1])/n_data
    p1e = p1a*p1p
    p2a = len(labels_int[labels_int==2])/n_data
    p2p = len(preds_int[preds_int==2])/n_data
    p2e = p2a*p2p
    p3a = len(labels_int[labels_int==3])/n_data
    p3p = len(preds_int[preds_int==3])/n_data
    p3e = p3a*p3p
    pe = p0e + p1e + p2e + p3e
    p0 = accuracy
    K2 = (p0 - pe)/(1 - pe)
    #print ("Cohen's Kappa: %.4f\n" % K2) #It is equivalent to the previous one (provided to include all pje values...)!

    precision = np.zeros(n_classes)
    recall = np.zeros(n_classes)
    F1 = np.zeros(n_classes)
    W = np.zeros(n_classes)
    FP = np.zeros(n_classes)
    FN = np.zeros(n_classes)
    TN = np.zeros(n_classes)
    TP = np.zeros(n_classes)
    FPR = np.zeros(n_classes) 
    TPR = np.zeros(n_classes) 
    for i in range(n_classes):
        TP[i] = df_cm_calc.iloc[i,i]
        W[i] = sum(df_cm_calc.iloc[i])
        precision[i] = TP[i] / sum(df_cm_calc.iloc[:,i])
        recall[i] = TP[i] / W[i]
        F1[i] = 2*precision[i]*recall[i] / (precision[i] + recall[i])
        ii = [x for j,x in enumerate(range(n_classes)) if j!=i]
        FN[i] = sum(df_cm_calc.iloc[i,ii]) #CHECKED!
        FP[i] = sum([df_cm_calc.iloc[j,i] for j in ii]) #CHECKED!
        TN[i] = sum([df_cm_calc.iloc[j,k] for j in ii for k in ii]) #CHECKED!
        FPR[i] = FP[i] / (FP[i] + TN[i])
        TPR[i] = TP[i] / (TP[i] + FN[i])

    precision_macro = precision.mean()
    precision_weighted = sum(precision[i] * W[i] / sum(W) for i in range(n_classes))
    recall_macro = recall.mean()
    recall_weighted = sum(recall[i] * W[i] / sum(W) for i in range(n_classes))
    F1_macro = F1.mean()
    F1_weighted = sum(F1[i] * W[i] / sum(W) for i in range(n_classes))
    error_weighted = 1 - precision_weighted
    FPR_weighted = sum(FPR[i] * W[i] / sum(W) for i in range(n_classes))
    TPR_weighted = sum(TPR[i] * W[i] / sum(W) for i in range(n_classes))
    #TPR_weighted = recall_weighted #equivalent to the previous one
    
    # Further Multi-class metrics
    print("matthews_corrcoef:", matthews_corrcoef(labels_int, preds_int))
    print("log_loss:", log_loss(y, model_preds))
    
    # Save results to a text file
    results = {
               'fpr': FPR_weighted,
               'tpr': TPR_weighted,
               'roc_auc': roc_auc["macro"],
               'Kappa': K,
               'accuracy': accuracy,
               'error': error_weighted,
               'precision': precision_weighted,
               'recall': recall_weighted,
               'F1': F1_weighted            
              }
    output_file = figuresDir + '../results_' + set_type + '.txt'
    import json
    with open(output_file, 'w') as f:
        f.write(json.dumps(results))
    
    # Return
    return preds_int, labels_int, results
