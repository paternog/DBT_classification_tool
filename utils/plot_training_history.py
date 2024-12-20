##########################################################################
# Author: Gianfranco Paternò (paterno@fe.infn.it)                        #
# collaboration of INFN Ferrara and INFN Napoli for the next_AIM project #
# Last modification date: March 2024                                     #
#                                                                        #
# SPDX-License-Identifier: EUPL-1.2                                      #
# Copyright 2024 Istituto Nazionale di Fisica Nucleare                   #
##########################################################################


import matplotlib.pyplot as plt


def plot_training_history(history, figuresDir, save_fig=True, MPLBACKEND=''):
    """
    Function to plot the training history of a model
    """

    # Retrieve the variables from the history dictionary
    acc = history['accuracy']
    val_acc = history['val_accuracy']
    loss = history['loss']
    val_loss = history['val_loss']
    if 'lr' in history.keys():
        lr = history['lr']
    epochs_range = range(1, len(acc)+1)

    plt.figure(figsize=(12, 12))
    fs = 16 
    
    # Plot Training and Validation accuracy     
    plt.subplot(2, 2, 1)
    plt.plot(epochs_range, acc, label='Training')
    plt.plot(epochs_range, val_acc, label='Validation')
    plt.xlabel('Epoch',fontsize=fs)
    plt.ylabel('Accuracy',fontsize=fs)
    plt.xticks(fontsize=fs, rotation=0)
    plt.yticks(fontsize=fs, rotation=0)
    #plt.ylim([0.0, 1.0])
    #plt.grid('on')
    plt.legend(fontsize=fs)

    # plot Training and Validation loss
    plt.subplot(2, 2, 2)
    plt.plot(epochs_range, loss, label='Training')
    plt.plot(epochs_range, val_loss, label='Validation')
    plt.xlabel('Epoch',fontsize=fs)
    plt.ylabel('Loss',fontsize=fs)
    plt.xticks(fontsize=fs, rotation=0)
    plt.yticks(fontsize=fs, rotation=0)
    plt.title('', fontsize=fs)
    #plt.grid('on')
    plt.legend(fontsize=fs)
    plt.xlabel('Epoch')
    plt.legend(fontsize=fs)
    
    if save_fig:
        plt.savefig(figuresDir + 'training_curves' + '.jpg')
    if not MPLBACKEND == 'Agg':
        plt.show()
    plt.close()
