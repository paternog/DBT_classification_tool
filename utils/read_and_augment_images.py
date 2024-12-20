##########################################################################
# Author: Gianfranco Paternò (paterno@fe.infn.it)                        #
# collaboration of INFN Ferrara and INFN Napoli for the next_AIM project #
# Last modification date: March 2024                                     #
#                                                                        #
# SPDX-License-Identifier: EUPL-1.2                                      #
# Copyright 2024 Istituto Nazionale di Fisica Nucleare                   #
##########################################################################


import numpy as np
from skimage.io import imread
from PIL import Image
import os
from glob import glob
import random
import pydicom
from skimage.transform import resize
from scipy import ndimage
from tqdm import tqdm



def read_images(dataset_path, classes, imgRescaleSize, grayLevels, img_type, \
                standardize=False, filterImg=False, sigmaFilter=2, shuffle=False, Nimgs2read=-1):
    
    """
    Function to read a certain image dataset by specifing the path
    and the classes/subfolders containing the labeled images.
    """
    
    tmp = []
    labels = []
    img_names_cls = []
    for cls in classes:
        # Return all the image filenames contained in a certain folder
        img_str = dataset_path + str(cls[0].upper() + cls[1:]) + '/*' + img_type
        fnames = glob(os.path.join(img_str))
        if len(fnames) == 0:
            img_str = dataset_path + str(cls[0].lower() + cls[1:]) + '/*' + img_type
            fnames = glob(os.path.join(img_str))

        # Select the figures to read for each class
        n_imgs_cls = len(fnames)
        if Nimgs2read != -1:
            fnames = random.choices(fnames, k=min(Nimgs2read, n_imgs_cls)) #select randomly Nimgs2read
        else:
            if shuffle:
                random.shuffle(fnames) #shuffle the images      
        img_names_cls.append(fnames)
    
        # Read, with a list comprehension, all the images listed before
        print('reading the images of the %s class' % cls)
        for fname in tqdm(fnames):
            img = Image.open(fname)
            if filterImg:
                img = Image.fromarray(ndimage.gaussian_filter(img, sigma=sigmaFilter))
            resized_img = img.resize(imgRescaleSize)
            if standardize:
                tmp += [(standardize_img(resized_img))]
                grayLevels = 1
            else:
                tmp += [np.array(resized_img)]
        
        # Correct the data labels
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
            
        # Populate the labels list with the label of the read image
        labels += len(fnames)*[cls]
        
    image_names = [item for sublist in img_names_cls for item in sublist]
 
    return np.array(tmp, dtype='float32')[..., np.newaxis]/grayLevels, np.array(labels), image_names



def read_images_dicom(dataset_path, classes, imgRescaleSize, grayLevels, img_type, \
                      standardize=False, filterImg=False, sigmaFilter=2, shuffle=False, OneHotCod=True):
    
    """
    Complete function that include the ability to handle dicom images
    and (optionally) One-Hot encoding.
    """
    
    tmp = []
    labels = []
    img_names_cls = []
    for cls in classes:
        # Return all the image filenames contained in a certain folder
        img_str = dataset_path + str(cls[0].upper() + cls[1:]) + '/*' + img_type
        fnames = glob(os.path.join(img_str))
        if len(fnames) == 0:
            img_str = dataset_path + str(cls[0].lower() + cls[1:]) + '/*' + img_type
            fnames = glob(os.path.join(img_str)) 
        
        if shuffle:
            random.shuffle(fnames) #shuffle the images
        img_names_cls.append(fnames)
    
        # Read, with a list comprehension, all the images listed before
        print('reading the images of %s class' % cls)
        #for fname in fnames:
        for fname in tqdm(fnames):
            if img_type == '.dcm':
                img = pydicom.dcmread(fname).pixel_array
                if filterImg:
                    img = ndimage.gaussian_filter(img, sigma=sigmaFilter)
                resized_img = Image.fromarray(resize(img, imgRescaleSize, anti_aliasing=True))
            else:
                img = Image.open(fname)
                if filterImg:
                    img = Image.fromarray(ndimage.gaussian_filter(img, sigma=sigmaFilter))
                resized_img = img.resize(imgRescaleSize)
            if standardize:
                tmp += [(standardize_img(resized_img))]
                grayLevels = 1
            else:
                tmp += [np.array(resized_img)]
        
        # Correct the data labels
        if len(classes) == 2:
            if isinstance(cls, str):    
                if cls.casefold() == 'negative':
                    cls = 0
                elif cls.casefold() == 'positive':
                    cls = 1                   
            if OneHotCod:
                if cls == 0:
                    cls = [1, 0]
                else:
                    cls = [0, 1]    
        else:
            if isinstance(cls, str):    
                if cls.casefold() == 'negative':
                    cls = [1, 0, 0]
                elif cls.casefold() == 'positive':
                    cls = [0, 1, 0]
                else:
                    cls = [0, 0, 1]     
            
        # Populate the labels list with the label of the read image
        labels += len(fnames)*[cls]
        
    image_names = [item for sublist in img_names_cls for item in sublist]
 
    return np.array(tmp, dtype='float32')[..., np.newaxis]/grayLevels, np.array(labels), image_names



def read_unlabeled_images(dataset_path, imgRescaleSize, grayLevels, img_type, \
                          standardize=False, filterImg=False, sigmaFilter=2):
    
    """
    Custom function to read the images of an unlabeled dataset.
    """
    
    tmp = []

    # Return all the image filenames contained in a certain folder
    img_str = dataset_path + '/*' + img_type
    fnames = glob(os.path.join(img_str))    

    # Read, with a list comprehension, all the images listed before
    print('reading the images of an unlabeled dataset')
    for fname in tqdm(fnames):
        img = Image.open(fname)
        if filterImg:
            img = Image.fromarray(ndimage.gaussian_filter(img, sigma=sigmaFilter))
        resized_img = img.resize(imgRescaleSize)
        if standardize:
            tmp += [(standardize_img(resized_img))]
            grayLevels = 1
        else:
            tmp += [np.array(resized_img)]

    return np.array(tmp, dtype='float32')[..., np.newaxis]/grayLevels, fnames



def augment_images(x, y, img_names, aug_factor, Ntransf):
    
    """
    Custom function for the augmentation of an image dataset.
    If aug_factor <= 1 there is a modality in which all the original images are kept and,
    for a fraction of them, data augmentation (sampling Ntransf out of 4 available) is applied.
    If aug_factor > 1, the data augmentation modality changes and the new dataset will be
    composed of all the original images plus each of them transformed int(Naug) times.
    """
    
    # Definition of Naug
    if aug_factor > 1:
        if not int(aug_factor) == aug_factor:
            Naug = max(2, int(aug_factor))
        else:
            Naug = aug_factor
    else:
        Naug = 0
    
    # Randomly select Ntransf transformations among those available
    trasformations = ["flip", "zoom", "rotation", "shift"]
    
    # Retrieve information about the dataset
    print(x.shape)
    dimImgs = (x.shape[1], x.shape[2])
    Nimages = x.shape[0]
    
    # Augment the image dataset
    if aug_factor <= 1:
        augmented_images_list = [x[i] for i in range(Nimages)]
        augmented_labels_list = [y[i] for i in range(Nimages)]
        names_list_augmented = [img_names[i] for i in range(Nimages)]
        Nimages_to_augment = round(Nimages*aug_factor) 
        images_to_augment = random.sample(range(Nimages), Nimages_to_augment)
        #print("indexes of the images to augment:", images_to_augment)
        for j in tqdm(images_to_augment):
            image = x[j,:,:,0]
            image_min = np.min(image) #useful with image standardization
            image_trasformations = random.sample(trasformations, Ntransf)                      
            if "flip" in image_trasformations:        
                image = flip(image)
            if "zoom" in image_trasformations:   
                image = zoom(image)
            if "rotation" in image_trasformations: 
                image = scipy_rotate(image, image_min)
            if "shift" in image_trasformations: 
                image = scipy_shift(image, image_min)
            image = resize(image, dimImgs, anti_aliasing=True, cval=0, order=3)     
            new_image = np.expand_dims(image, axis=-1)
            augmented_images_list.append(new_image)
            augmented_labels_list.append(y[j])
            img_type = img_names[j].split('.')[1]
            img_name = img_names[j].split('.')[0]
            img_name_aug = img_name + '_aug.' + img_type
            names_list_augmented.append(img_name_aug)           
        x_train_augmented = np.array(augmented_images_list)
        y_train_augmented = np.array(augmented_labels_list) 
    else:
        augmented_images_list = [x[i] for i in range(Nimages)]
        augmented_labels_list = [y[i] for i in range(Nimages)]
        temp_names_list = [img_names[i] for i in range(Nimages)]        
        for j in tqdm(range(Nimages)):
            image = x[j,:,:,0]  
            mylabel = y[j]
            img_type = img_names[j].split('.')[1]
            img_name = img_names[j].split('.')[0]
            image_min = np.min(image) #useful with image standardization
            for i in range(Naug):
                image_trasformations = random.sample(trasformations, Ntransf)                      
                if "flip" in image_trasformations:        
                    image = flip(image)
                if "zoom" in image_trasformations:   
                    image = zoom(image)
                if "rotation" in image_trasformations: 
                    image = scipy_rotate(image, image_min)
                if "shift" in image_trasformations: 
                    image = scipy_shift(image, image_min)
                image = resize(image, dimImgs, anti_aliasing=True, cval=0, order=3)
                new_image = np.expand_dims(image, axis=-1)
                augmented_images_list.append(new_image)
                augmented_labels_list.append(mylabel)
                img_name_aug = img_name + '_aug_' + str(i+1) + '.' + img_type
                temp_names_list.append(img_name_aug)
        # Shuffle images
        temp = list(zip(augmented_images_list, augmented_labels_list, temp_names_list))
        random.shuffle(temp)
        res1, res2, res3 = zip(*temp)
        res1, res2, res3 = list(res1), list(res2), list(res3)
        # Return an augemented numpy array of images
        x_train_augmented = np.array(res1)
        y_train_augmented = np.array(res2)
        names_list_augmented = res3
    
    # Return augmented data
    return x_train_augmented, y_train_augmented, names_list_augmented


########################################################################
# Custom functions (some based on scipy.ndimage) to transform an image #
########################################################################
def scipy_rotate(image, img_min):
    # define some rotation angles
    angles = [-30, -15, -10, 10, 15, 30]
    # pick angles at random
    angle = random.choice(angles)
    # rotate volume
    image_rot = ndimage.rotate(image, angle, cval=img_min, reshape=False)
    return image_rot
    
def scipy_shift(img, img_min):
    # define some y-displacemets (in percentage of pixels)
    list_shift = [-0.078, -0.058, -0.038, 0.038, 0.058, 0.078]
    # pick shift at random
    ty = random.choice(list_shift)*img.shape[1]
    tx = random.choice(list_shift)*img.shape[0]
    # shift volume
    img_shift = ndimage.shift(img, shift=(tx,ty), order=3, cval=img_min)
    return img_shift

def zoom(img):
    x,y = img.shape
    list_zoom = [0.05, 0.04, 0.03, 0.02]
    m = random.choice(list_zoom)
    dx = int(m*x)
    dy = int(m*y)
    xm = dx
    xM = x - dx
    ym = dy
    yM = y - dy
    if xm < 0:
        xm = 0
    if ym < 0:
        ym = 0
    if xM > img.shape[0]:
        xM = img.shape[0]
    if yM > img.shape[1]:
        yM = img.shape[1]
    xm = int(xm)
    ym = int(ym)
    xM = int(xM)
    yM = int(yM)
    img_zoom = img[xm:xM, ym:yM]
    return img_zoom

def flip(img):
    img_flip = img[:,::-1]
    return img_flip



def augment_images_Keras(x, y, img_names, Naug, img_min):
    
    """
    Custom function for image dataset augmentation based on Keras Leyers.
    """
    
    # Check 
    if not int(Naug) == Naug:
        Naug = max(2, int(Naug))
    
    # Import and set tf
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    
    # Keras image dataset augmentation function
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.1), #for some reason, it is too slow...
        tf.keras.layers.RandomTranslation(height_factor=0.2, #for some reason, it is too slow...
                                          width_factor=0.2, 
                                          fill_mode='reflect', 
                                          interpolation='bilinear',
                                          seed=None, 
                                          fill_value=img_min),
        tf.keras.layers.RandomZoom(height_factor=0.2, #for some reason, it is too slow...
                                   width_factor=None, 
                                   fill_mode='reflect', 
                                   interpolation='bilinear', 
                                   seed=None, 
                                   fill_value=0.0)
    ])
    
    # Augment image dataset
    dimImgs = (x.shape[1], x.shape[2])
    NtestImages = x.shape[0]
    augmented_images_list = []
    augmented_labels_list = []
    temp_names_list = []
    for j in tqdm(range(NtestImages)):
        image = np.array(x[j])
        image = tf.expand_dims(image, 0)
        mylabel = y[j]
        img_type = img_names[j].split('.')[1]
        img_name = img_names[j].split('.')[0]
        for i in range(Naug):
            augmented_image = data_augmentation(image) #since it is not working properly, I use custom functions
            augmented_images_list.append(augmented_image[0])
            augmented_labels_list.append(mylabel)
            img_name_aug = img_name + '_aug_' + str(i+1) + '.' + img_type
            temp_names_list.append(img_name_aug)   
            
    # Shuffle images
    temp = list(zip(augmented_images_list, augmented_labels_list, temp_names_list))
    random.shuffle(temp)
    res1, res2, res3 = zip(*temp)
    res1, res2, res3 = list(res1), list(res2), list(res3) 
    
    # Return an augemented numpy array of images
    x_train_augmented = np.array(res1)
    y_train_augmented = np.array(res2)
    names_list_augmented = res3
    return x_train_augmented, y_train_augmented, names_list_augmented



def standardize_img(img):
    """
    Custom funtion (used here) to standardize an image.
    """
    import numpy as np
    a = np.array(img).astype(np.float32)
    mean = np.mean(a[a>0])
    std = np.std(a[a>0])
    return (a - mean) / std



def standardize_img_tensor(a, axis=None):
    """
    Custom funtion (used in the main script) to standardize a tensor of images of size (N,H,W,C)
    where N is the number of images, H the image height, W the image width and C the RGB channels.
    """
    import numpy as np
    # axis param denotes axes along which mean & std reductions are to be performed
    mean = np.mean(a[a>0], axis=axis, keepdims=True)
    std = np.sqrt(((a[a>0] - mean)**2).mean(axis=axis, keepdims=True))
    return (a - mean) / std
