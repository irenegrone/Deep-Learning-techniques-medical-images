## coding: utf-8

##MSc Data Science part-time
##MSc Project - September 2020
##
##Irene Grone   
##ID 13151264
##
##
##Academic Declaration.
##
##I have read and understood the sections of plagiarism in the College Policy 
##on assessment offences and confirm that the work is my own, with the work 
##of others clearly acknowledged. 
##I give my permission to submit my report to the plagiarism testing database 
##that the College is using and test it using plagiarism detection software,
##search engines or meta-searching software.
##
##
##Resources
##
##Keras official documentation, available on line at: https://keras.io/api/ 
##Numpy official documentation, available online at: https://numpy.org
##Pandas official documentation, available online at: https://pandas.pydata.org/docs/
##pydicom official documentation, available on line at:  https://pydicom.github.io/pydicom/stable/old/pydicom_user_guide.html 
##Scikit-image official documentation, available on line at: https://scikit-image.org/docs/stable/
##Scikit-Learn official documentation, available on line at: https://scikit-learn.org/stable/modules/classes.html 
##Tensorflow official documentation, available on line at: https://www.tensorflow.org/api_docs/python/tf


##In this script:
##    - loading the datasets from storage as downloaded from TCIA database
##    - data preparation
##    - running Leave-One-Out loops for CNN models
##    - saveing the results for each model to file
##
##Before running the code please set up you working directory,
##change the file paths for dataset storage and result file names
##accordingly to you system


## standard import
import numpy as np
import pandas as pd
import os
import sys
import copy
import matplotlib.pyplot as plt

from os.path import dirname, join
from pprint import pprint

import pydicom
from skimage.transform import rotate, resize
from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, accuracy_score

import tensorflow as tf
from tensorflow import keras
from keras import layers, models, initializers, metrics

## file paths, change directory paths according to file system

PATH_STS_clinical_data = 'D:\MSc data\INFOclinical_STS.xlsx'
PATH_STS_img_Aligned_T1toPET = 'D:\MSc data\Soft-tissue-Sarcoma_Aligned_T1toPET'


def read_dicom_pixel(dir_path):
    """
    Function to read dicom files, extract 'Pixel Data' and 'Patient ID' attributes
    and  create lists of images and corresponding 'Patient ID'.
    
    'Pixel Data' array with maximum value below 11 are not included     
    
    Args:
        dir_path (string): path to overall directory of dicom files as dowloaded from TCIA database
        
    Returns:
        img (list of 2d np.arrays): list of 'Pixel Data' arrays
        
        pID (list of string): list of corresponding 'Patient ID'
    
    """
    
    tree = os.walk(dir_path)
    file_list = []
    
    # create list of file paths - the file are already sorted when dowloaded from TCIA
    for dir_a, dir_b, files in tree:
        for f in files:
            p = os.path.join(str(dir_a), str(f))
            file_list.append(p)
    
    img = []
    pID = []
    
    # read DICOM files, extract the attributes pixel_array and PatientID
    for i in range(len(file_list)):
        dicom_data = pydicom.dcmread(file_list[i])
        dicom_pixel = dicom_data.pixel_array
        if dicom_pixel.max() > 10:
            img.append(dicom_pixel)
            pID.append(dicom_data.PatientID)
            
    print("Data loaded and cleaned - lengths: ", len(img), len(pID))
    
    return img, pID


def outcome_label(patientID_list, pID_outcome):
    """
    Function that creates the corresponding outcome label for the images extracted 
    using read_dicom_pixel() from the list of patientIDs
    
    Args:
        patientID_list (list of string): this is the pID list output from read_dicom_pixel()
        
        pID_outcome (pd.DataFrame): outcome data from clinical data file (sheet=1) saved in a pd.DataFrame
    
    Returns:
        label (pd.DataFrame): Patient IDs and outcome labels for the images retrieved with read_dicom_pixel()
    
    """
    
    label = pd.DataFrame({'Patient ID': patientID_list})
    label = pd.merge(label, pID_outcome, on='Patient ID', how='left')
    
    print("Labels dataset created")
    
    return label


def extract_patches_48(img):
    """
    helper function to extract non overlapping patches of the size 48 x 48,
    except for those patches extracted from bottom and left of image
    where the patches are extracted starting from the bottom and / or left
    of the image and the overlap depends on the raw image dimensions
    
    Args:
        img (2d array): pixel data of the image
    
    Returns:
        img_patch (list): list of patches in 2d array format
        
    """
    img_patch = []
    
    len_h = img.shape[0]
    len_w = img.shape[1]

    for ii in range(0, len_h, 48):
        for jj in range(0, len_w, 48):
            if ii > len_h-48:
                if jj < len_w-48:
                    imp_b = img[-48:, jj:jj+48]
                else:
                    imp_b = img[-48:, -48:]
            elif jj > len_w-48:    
                imp_b = img[ii:ii+48, -48:]
            else:
                imp_b = img[ii:ii+48, jj:jj+48]
        
            img_patch.append(imp_b)
    
    return img_patch


def image_patches_dataset(img_list, label_df):
    """
    Function that create the datset extracting patches from 
    each image in img_list and return them as array and normalize to values between 0 and 1
    considering the maximum pixel value in the dataset,
    create the lables data accrodingly
    
    Args:
        img_list (list of arrays):  list of image arrays returned by read_dicom_pixel()
        
        label_df (pd.DataFrame): DataFrame with patient IDs and label returned by oucome_label()
        
    
    Returns:
        ID (pd.DataFrame): patient ID dataset for splitting on Leave-One-Out
    
        data (np.array): images dataset
        
        labels (np.array): labels dataset
    
    """
    
    m_val = []
    patient_ID = []
    img_patches = []
    labels = []
    
    for i in range(len(img_list)):
        img_raw = copy.deepcopy(img_list[i])    
        pat = extract_patches_48(img_raw)
        
        pID = label_df['Patient ID'][i]
        img_patches = img_patches + pat
        l = label_df['LungMets'][i]
    
        for k in range(len(pat)):
            patient_ID.append(pID)
            labels.append(l)
    
    for k in range(len(img_list)):
        a = np.amax(img_list[k])
        m_val.append(a)
    
    data = np.array(img_patches) / max(m_val)
    labels = np.array(labels, dtype=np.int8)
    
    ID = pd.DataFrame({'patient_ID': patient_ID})
    print('Patches datasets created')
                           
    return ID, data, labels


def main():

## check system and version
    print('System and version for compatibility')
    print(sys.version)
    
## Data preparation
# load clinical data
    sts_clinical_data = pd.read_excel(PATH_STS_clinical_data, sheet_name=0)

# extract row with data
    sts_clinical_data = sts_clinical_data[:51]

# clinical data sample
    print('Clinical data extract of top rows')
    print(sts_clinical_data.head())
    print('Clinical data attributes in clinical data file')
    print(sts_clinical_data.columns)

# outcome - extract label data
    sts_outcome = pd.read_excel(PATH_STS_clinical_data, sheet_name=1)

## check data - Target variable values, clinical data, label data
    print('Type of cancer at follow up visit - count')
    print(sts_clinical_data['Outcome (recurrence, mets)'].value_counts(),'\n')

# Lung metastasis outcome in the follow-up period: 0 for negative, no tumour in the lungs, 1 positive, tumour in the lungs
    print('Count of labels')
    print(sts_outcome['LungMets'].value_counts())

# list of areas of the body with verified STS and count
    print('Area of the body under investigation and count')
    print(sts_clinical_data['Site of primary STS'].value_counts())

# example of a DICOM file of an MR T1-weigthed for patient STS_002
    ds_T1 = pydicom.dcmread('D:\\MSc data\\Soft-tissue-Sarcoma_MR_T1\\STS_002\\01-03-2006-L-SPINE-56934\\501.000000-AXT1-02298\\1-01.dcm')
    print('Example of DICOM file')
    print(ds_T1)

# example of a DICOM file of an aligned_T1toPET for patient STS_002 
    ds_T1toPET = pydicom.dcmread('D:/MSc data/Soft-tissue-Sarcoma_Aligned_T1toPET/STS_002/01-28-2006-PETCT with added MR-70477/6564013.000000-AllignedT1toPETBOX-15491/1-01.dcm')

# plot of the two image types comparison
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1,2,1).imshow(ds_T1.pixel_array, cmap=plt.cm.gray)
    ax = fig.add_subplot(1,2,2).imshow(ds_T1toPET.pixel_array, cmap=plt.cm.gray)

    plt.show()

# Fused PET-MR images
# extract extract 'Pixel Data' and 'Patient ID' attributes 
    T1toPET_img, T1toPET_patient = read_dicom_pixel(PATH_STS_img_Aligned_T1toPET)

# corresponding labels dataset for Fused PET-MR images
    T1toPET_out = outcome_label(T1toPET_patient, sts_outcome)
    print('Total number of images for each label')
    print(T1toPET_out['LungMets'].value_counts())

# Fused PET-MR images - image size and number of images by Patient ID

    pat_uni = T1toPET_out['Patient ID'].unique()
    h = []
    w = []
    qty = []

    for i in pat_uni:
        h.append(T1toPET_img[(T1toPET_out['Patient ID']==i).idxmax()].shape[0]) 
        w.append(T1toPET_img[(T1toPET_out['Patient ID']==i).idxmax()].shape[1])
        qty.append((T1toPET_out['Patient ID'].where(T1toPET_out['Patient ID']==i).last_valid_index()) - (T1toPET_out['Patient ID']==i).idxmax() + 1)

    sizes = pd.DataFrame({'patient_ID': pat_uni,
                          'height': h,
                          'width': w,
                          'No. images': qty
                         })

    print(sizes)

# create image patches datasets, reshape as 4D tensor for input in Leave-One-Out CNN loop
    fused_ID, fused_dataset, fused_label = image_patches_dataset(T1toPET_img, T1toPET_out)
    fused_dataset = fused_dataset.reshape((fused_dataset.shape[0], fused_dataset.shape[1], fused_dataset.shape[2], 1))
    print('Check shape tensors: ' , fused_dataset.shape, fused_label.shape)

# unique patient IDs list to use in index search for Leave-One-Out split loop
    ids = fused_ID['patient_ID'].unique()

    input('Data preparation complete. Press Enter to see summary of fused_model_1')

# CNN model - first model proposed for fused images and summary
    fused_model_1 = models.Sequential(name='fused_model_1')
    fused_model_1.add(keras.Input(shape=[48, 48, 1], name='input'))
    fused_model_1.add(layers.Conv2D(8, (3, 3), activation='relu', padding='same', kernel_initializer='random_normal', bias_initializer='zeros', name='conv_1'))
    fused_model_1.add(layers.Conv2D(8, (3, 3), activation='relu', padding='same', kernel_initializer='random_normal', bias_initializer='zeros', name='conv_2'))
    fused_model_1.add(layers.MaxPooling2D(2, name='pooling_1'))
    fused_model_1.add(layers.Conv2D(16, (3, 3), activation='relu', padding='same', kernel_initializer='random_normal', bias_initializer='zeros', name='conv_3'))
    fused_model_1.add(layers.Conv2D(16, (3, 3), activation='relu', padding='same', kernel_initializer='random_normal', bias_initializer='zeros', name='conv_4'))
    fused_model_1.add(layers.MaxPooling2D(2, name='pooling_2'))
    fused_model_1.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='random_normal', bias_initializer='zeros', name='conv_5'))
    fused_model_1.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='random_normal', bias_initializer='zeros', name='conv_6'))
    fused_model_1.add(layers.MaxPooling2D(2, name='pooling_3'))
    fused_model_1.add(layers.Flatten(name='flatten'))
    fused_model_1.add(layers.Dense(128, activation='relu', name='FC_1', kernel_initializer='random_normal'))
    fused_model_1.add(layers.Dense(16, activation='relu', name='FC_2', kernel_initializer='random_normal'))
    fused_model_1.add(layers.Dense(1, activation='sigmoid', name='output'))

    print(fused_model_1.summary())

    input('Press Enter to see summary of fused_model_2')
    
# CNN models - second model proposed for fused images and summary
    fused_model_2 = models.Sequential(name='fused_model_2')
    fused_model_2.add(keras.Input(shape=[48, 48, 1], name='input'))
    fused_model_2.add(layers.Conv2D(8, (3, 3), activation='relu', padding='valid', kernel_initializer='random_normal', bias_initializer='zeros', name='conv_1'))
    fused_model_2.add(layers.MaxPooling2D(2, name='pooling_1'))
    fused_model_2.add(layers.Conv2D(16, (3, 3), activation='relu', padding='valid', kernel_initializer='random_normal', bias_initializer='zeros', name='conv_2'))
    fused_model_2.add(layers.MaxPooling2D(2, name='pooling_2'))
    fused_model_2.add(layers.Conv2D(32, (3, 3), activation='relu', padding='valid', kernel_initializer='random_normal', bias_initializer='zeros', name='conv_3'))
    fused_model_2.add(layers.MaxPooling2D(2, name='pooling_3'))
    fused_model_2.add(layers.Flatten(name='flatten'))
    fused_model_2.add(layers.Dense(64, activation='relu', name='FC_1', kernel_initializer='random_normal'))
    fused_model_2.add(layers.Dense(1, activation='sigmoid', name='output'))

    print(fused_model_2.summary())

## the code below this point start the models training,
## please comment/uncomment out depending on the model you want to run

####################################################################################################
## Leave-One-Out loop for fused_model_1 epoch: 50

    input('Press Enter to start Leave-One-Out loop on fused_model_1 epoch=50. This loop took approximately 1h 30 min to run on a GPU enabled machine')

    EPOCH_f1 = 50
    BATCH_f1 = 200

    true_label = []
    test_size = []
    clf_correct = []
    accuracy = []

    for pid in ids:
        id_start = (fused_ID['patient_ID']==pid).idxmax()
        id_stop = (fused_ID['patient_ID'].where(fused_ID['patient_ID']==pid).last_valid_index())

    # train and test dataset
        if id_start==0:
            fused_x_train = fused_dataset[id_stop+1:]
            fused_y_train = fused_label[id_stop+1:]
        else:
            fused_x_train = np.concatenate((fused_dataset[:id_start], fused_dataset[id_stop+1:]), axis=0)
            fused_y_train = np.concatenate((fused_label[:id_start], fused_label[id_stop+1:]), axis=0)
            
        fused_x_test = fused_dataset[id_start:id_stop+1]
        fused_y_test = fused_label[id_start:id_stop+1]

    # print start training
        print('start training for ID: ', pid, end='\t')

    # shuffle training data before 
        fused_x_train, fused_y_train = shuffle(fused_x_train, fused_y_train, random_state=31)

    # initialize the model
        fused_model_1 = models.Sequential(name='fused_model_1')
        fused_model_1.add(keras.Input(shape=[48, 48, 1], name='input'))
        fused_model_1.add(layers.Conv2D(8, (3, 3), activation='relu', padding='same', kernel_initializer='random_normal', bias_initializer='zeros', name='conv_1'))
        fused_model_1.add(layers.Conv2D(8, (3, 3), activation='relu', padding='same', kernel_initializer='random_normal', bias_initializer='zeros', name='conv_2'))
        fused_model_1.add(layers.MaxPooling2D(2, name='pooling_1'))
        fused_model_1.add(layers.Conv2D(16, (3, 3), activation='relu', padding='same', kernel_initializer='random_normal', bias_initializer='zeros', name='conv_3'))
        fused_model_1.add(layers.Conv2D(16, (3, 3), activation='relu', padding='same', kernel_initializer='random_normal', bias_initializer='zeros', name='conv_4'))
        fused_model_1.add(layers.MaxPooling2D(2, name='pooling_2'))
        fused_model_1.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='random_normal', bias_initializer='zeros', name='conv_5'))
        fused_model_1.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='random_normal', bias_initializer='zeros', name='conv_6'))
        fused_model_1.add(layers.MaxPooling2D(2, name='pooling_3'))
        fused_model_1.add(layers.Flatten(name='flatten'))
        fused_model_1.add(layers.Dense(128, activation='relu', name='FC_1', kernel_initializer='random_normal'))
        fused_model_1.add(layers.Dense(16, activation='relu', name='FC_2', kernel_initializer='random_normal'))
        fused_model_1.add(layers.Dense(1, activation='sigmoid', name='output'))
        
    # compile the model
        fused_model_1.compile(
            optimizer='rmsprop',
            loss='binary_crossentropy',
            metrics=['binary_accuracy']
            )
        
    # fit the model
        fused_model_1.fit(
            fused_x_train, 
            fused_y_train, 
            batch_size=BATCH_f1,  
            epochs=EPOCH_f1,
            verbose=0
            )
        
    # predict label
        y_pred = fused_model_1.predict(fused_x_test)
        y_pred = y_pred.astype(dtype=np.int8, copy=True)
        
    # save results and accuracy 
        correct = accuracy_score(fused_y_test, y_pred, normalize=False)
        clf_correct.append(correct)
        tot = len(fused_y_test)
        test_size.append(tot)
        acc = correct/tot
        accuracy.append(acc)
        true_label.append(fused_y_test[0])
        
        print(' finished: label ', fused_y_test[0], ' LOO test set ', tot, ', correct clf ', correct, ', acc ', acc)

    input('Pres Enter to results of LOO on fused_model_1')

# create dataframe for storing and presenting results
    model_1_rmsprop_ep50_results = pd.DataFrame({
        'Patient ID': ids,
        'True label': true_label,
        'test set size': test_size,
        'correctly classified': clf_correct,
        'accuracy': accuracy
    })

    print(model_1_rmsprop_ep50_results.head())

# save results to csv - change directory path according to your system
    f_name1 = 'model_1_rmsprop_ep50_results.csv'
    os.chdir('C:\\Users\\irene\\Desktop\\Birkbeck\\MSc_Project\\MSc project results')
    model_1_rmsprop_ep50_results.to_csv(f_name1, encoding='utf-8', index=False)
    print('Results saved in file')

    
####################################################################################################
## Leave-One-Out loop for fused_model_1 epoch: 100

    input('Press Enter to start Leave-One-Out loop on fused_model_1 epoch=100. This loop took approximately 2h 45min to run on a GPU enabled machine')
        
    EPOCH_f1_b = 100
    BATCH_f1_b = 200

    true_label_b = []
    test_size_b = []
    clf_correct_b = []
    accuracy_b = []

    for pid in ids:
        id_start = (fused_ID['patient_ID']==pid).idxmax()
        id_stop = (fused_ID['patient_ID'].where(fused_ID['patient_ID']==pid).last_valid_index())

    # train and test dataset
        if id_start==0:
            fused_x_train = fused_dataset[id_stop+1:]
            fused_y_train = fused_label[id_stop+1:]
        else:
            fused_x_train = np.concatenate((fused_dataset[:id_start], fused_dataset[id_stop+1:]), axis=0)
            fused_y_train = np.concatenate((fused_label[:id_start], fused_label[id_stop+1:]), axis=0)

        fused_x_test = fused_dataset[id_start:id_stop+1]
        fused_y_test = fused_label[id_start:id_stop+1]

    # print start training
        print('start training for ID: ', pid, end='\t')

    # shuffle training data before 
        fused_x_train, fused_y_train = shuffle(fused_x_train, fused_y_train, random_state=31)

    # initialize the model
        fused_model_1 = models.Sequential(name='fused_model_1')
        fused_model_1.add(keras.Input(shape=[48, 48, 1], name='input'))
        fused_model_1.add(layers.Conv2D(8, (3, 3), activation='relu', padding='same', kernel_initializer='random_normal', bias_initializer='zeros', name='conv_1'))
        fused_model_1.add(layers.Conv2D(8, (3, 3), activation='relu', padding='same', kernel_initializer='random_normal', bias_initializer='zeros', name='conv_2'))
        fused_model_1.add(layers.MaxPooling2D(2, name='pooling_1'))
        fused_model_1.add(layers.Conv2D(16, (3, 3), activation='relu', padding='same', kernel_initializer='random_normal', bias_initializer='zeros', name='conv_3'))
        fused_model_1.add(layers.Conv2D(16, (3, 3), activation='relu', padding='same', kernel_initializer='random_normal', bias_initializer='zeros', name='conv_4'))
        fused_model_1.add(layers.MaxPooling2D(2, name='pooling_2'))
        fused_model_1.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='random_normal', bias_initializer='zeros', name='conv_5'))
        fused_model_1.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='random_normal', bias_initializer='zeros', name='conv_6'))
        fused_model_1.add(layers.MaxPooling2D(2, name='pooling_3'))
        fused_model_1.add(layers.Flatten(name='flatten'))
        fused_model_1.add(layers.Dense(128, activation='relu', name='FC_1', kernel_initializer='random_normal'))
        fused_model_1.add(layers.Dense(16, activation='relu', name='FC_2', kernel_initializer='random_normal'))
        fused_model_1.add(layers.Dense(1, activation='sigmoid', name='output'))

    # compile the model
        fused_model_1.compile(
            optimizer='rmsprop',
            loss='binary_crossentropy',
            metrics=['binary_accuracy']
            )

    # fit the model
        fused_model_1.fit(
            fused_x_train, 
            fused_y_train, 
            batch_size=BATCH_f1_b,  
            epochs=EPOCH_f1_b,
            verbose=0
            )

    # predict label
        y_pred = fused_model_1.predict(fused_x_test)
        y_pred = y_pred.astype(dtype=np.int8, copy=True)

    # save results and accuracy 
        correct = accuracy_score(fused_y_test, y_pred, normalize=False)
        clf_correct_b.append(correct)
        tot = len(fused_y_test)
        test_size_b.append(tot)
        acc = correct/tot
        accuracy_b.append(acc)
        true_label_b.append(fused_y_test[0])

        print(' finished, results: label ', fused_y_test[0], ' LOO test set ', tot, ', correct clf ', correct, ', acc ', acc)
    
# create dataframe for storing and presenting results
    model_1_rmsprop_ep100_results = pd.DataFrame({
        'Patient ID': ids,
        'True label': true_label_b,
        'test set size': test_size_b,
        'correctly classified': clf_correct_b,
        'accuracy': accuracy_b
    })

    model_1_rmsprop_ep100_results
    
    
# save results to csv - change directory path according to your system
    f_name2 = 'model_1_rmsprop_ep100_results.csv'
    os.chdir('C:\\Users\\irene\\Desktop\\Birkbeck\\MSc_Project\\MSc project results')
    model_1_rmsprop_ep100_results.to_csv(f_name2, encoding='utf-8', index=False)
    print('Results saved in file')


####################################################################################################
## Leave-One-Out loop for fused_model_2 epoch: 100

    input('Press Enter to start Leave-One-Out loop on fused_model_2 epoch=100. This loop took approximately 1h 05min to run on a GPU enabled machine')

    EPOCH_f2 = 100
    BATCH_f2 = 200

    true_label_f2 = []
    test_size_f2 = []
    clf_correct_f2 = []
    accuracy_f2 = []

    for pid in ids:
        id_start = (fused_ID['patient_ID']==pid).idxmax()
        id_stop = (fused_ID['patient_ID'].where(fused_ID['patient_ID']==pid).last_valid_index())

    # train and test dataset
        if id_start==0:
            fused_x_train = fused_dataset[id_stop+1:]
            fused_y_train = fused_label[id_stop+1:]
        else:
            fused_x_train = np.concatenate((fused_dataset[:id_start], fused_dataset[id_stop+1:]), axis=0)
            fused_y_train = np.concatenate((fused_label[:id_start], fused_label[id_stop+1:]), axis=0)

        fused_x_test = fused_dataset[id_start:id_stop+1]
        fused_y_test = fused_label[id_start:id_stop+1]

    # print start training
        print('start training for ID: ', pid, end='\t')

    # shuffle training data before 
        fused_x_train, fused_y_train = shuffle(fused_x_train, fused_y_train, random_state=31)

    # initialize the model
        fused_model_2 = models.Sequential(name='fused_model_2')
        fused_model_2.add(keras.Input(shape=[48, 48, 1], name='input'))
        fused_model_2.add(layers.Conv2D(8, (3, 3), activation='relu', padding='valid', kernel_initializer='random_normal', bias_initializer='zeros', name='conv_1'))
        fused_model_2.add(layers.MaxPooling2D(2, name='pooling_1'))
        fused_model_2.add(layers.Conv2D(16, (3, 3), activation='relu', padding='valid', kernel_initializer='random_normal', bias_initializer='zeros', name='conv_2'))
        fused_model_2.add(layers.MaxPooling2D(2, name='pooling_2'))
        fused_model_2.add(layers.Conv2D(32, (3, 3), activation='relu', padding='valid', kernel_initializer='random_normal', bias_initializer='zeros', name='conv_3'))
        fused_model_2.add(layers.MaxPooling2D(2, name='pooling_3'))
        fused_model_2.add(layers.Flatten(name='flatten'))
        fused_model_2.add(layers.Dense(64, activation='relu', name='FC_1', kernel_initializer='random_normal'))
        fused_model_2.add(layers.Dense(1, activation='sigmoid', name='output'))

    # compile the model
        fused_model_2.compile(
            optimizer='rmsprop',
            loss='binary_crossentropy',
            metrics=['binary_accuracy']
            )

    # fit the model
        fused_model_2.fit(
            fused_x_train, 
            fused_y_train, 
            batch_size=BATCH_f2,  
            epochs=EPOCH_f2,
            verbose=0
            )

    # predict label
        y_pred = fused_model_2.predict(fused_x_test)
        y_pred = y_pred.astype(dtype=np.int8, copy=True)

    # save results and accuracy 
        correct = accuracy_score(fused_y_test, y_pred, normalize=False)
        clf_correct_f2.append(correct)
        tot = len(fused_y_test)
        test_size_f2.append(tot)
        acc = correct/tot
        accuracy_f2.append(acc)
        true_label_f2.append(fused_y_test[0])

        print(' finished: label ', fused_y_test[0], ' LOO test set ', tot, ', correct clf ', correct, ', acc ', acc)

# create dataframe for storing and presenting results
    model_2_rmsprop_ep100_results = pd.DataFrame({
        'Patient ID': ids,
        'True label': true_label_f2,
        'test set size': test_size_f2,
        'correctly classified': clf_correct_f2,
        'accuracy': accuracy_f2
    })

    model_2_rmsprop_ep100_results
    
    
# save results to csv - change directory path according to your system
    f_name3 = 'model_2_rmsprop_ep100_results.csv'
    os.chdir('C:\\Users\\irene\\Desktop\\Birkbeck\\MSc_Project\\MSc project results')
    model_2_rmsprop_ep100_results.to_csv(f_name3, encoding='utf-8', index=False)
    print('Results saved in file')

    
####################################################################################################
## Leave-One-Out loop for fused_model_2 epoch: 150

    input('Press Enter to start Leave-One-Out loop on fused_model_2 epoch=150. This loop took approximately 1h 40min to run on a GPU enabled machine')

    EPOCH_f2_b = 150
    BATCH_f2_b = 200

    true_label_f2_b = []
    test_size_f2_b = []
    clf_correct_f2_b = []
    accuracy_f2_b = []

    for pid in ids:
        id_start = (fused_ID['patient_ID']==pid).idxmax()
        id_stop = (fused_ID['patient_ID'].where(fused_ID['patient_ID']==pid).last_valid_index())

    # train and test dataset
        if id_start==0:
            fused_x_train = fused_dataset[id_stop+1:]
            fused_y_train = fused_label[id_stop+1:]
        else:
            fused_x_train = np.concatenate((fused_dataset[:id_start], fused_dataset[id_stop+1:]), axis=0)
            fused_y_train = np.concatenate((fused_label[:id_start], fused_label[id_stop+1:]), axis=0)

        fused_x_test = fused_dataset[id_start:id_stop+1]
        fused_y_test = fused_label[id_start:id_stop+1]

    # print start training
        print('start training for ID: ', pid, end='\t')

    # shuffle training data before 
        fused_x_train, fused_y_train = shuffle(fused_x_train, fused_y_train, random_state=31)

    # initialize the model
        fused_model_2 = models.Sequential(name='fused_model_2')
        fused_model_2.add(keras.Input(shape=[48, 48, 1], name='input'))
        fused_model_2.add(layers.Conv2D(8, (3, 3), activation='relu', padding='valid', kernel_initializer='random_normal', bias_initializer='zeros', name='conv_1'))
        fused_model_2.add(layers.MaxPooling2D(2, name='pooling_1'))
        fused_model_2.add(layers.Conv2D(16, (3, 3), activation='relu', padding='valid', kernel_initializer='random_normal', bias_initializer='zeros', name='conv_2'))
        fused_model_2.add(layers.MaxPooling2D(2, name='pooling_2'))
        fused_model_2.add(layers.Conv2D(32, (3, 3), activation='relu', padding='valid', kernel_initializer='random_normal', bias_initializer='zeros', name='conv_3'))
        fused_model_2.add(layers.MaxPooling2D(2, name='pooling_3'))
        fused_model_2.add(layers.Flatten(name='flatten'))
        fused_model_2.add(layers.Dense(64, activation='relu', name='FC_1', kernel_initializer='random_normal'))
        fused_model_2.add(layers.Dense(1, activation='sigmoid', name='output'))

    # compile the model
        fused_model_2.compile(
            optimizer='rmsprop',
            loss='binary_crossentropy',
            metrics=['binary_accuracy']
            )

    # fit the model
        fused_model_2.fit(
            fused_x_train, 
            fused_y_train, 
            batch_size=BATCH_f2_b,  
            epochs=EPOCH_f2_b,
            verbose=0
            )

    # predict label
        y_pred = fused_model_2.predict(fused_x_test)
        y_pred = y_pred.astype(dtype=np.int8, copy=True)

    # save results and accuracy 
        correct = accuracy_score(fused_y_test, y_pred, normalize=False)
        clf_correct_f2_b.append(correct)
        tot = len(fused_y_test)
        test_size_f2_b.append(tot)
        acc = correct/tot
        accuracy_f2_b.append(acc)
        true_label_f2_b.append(fused_y_test[0])

        print(' finished: label ', fused_y_test[0], ' LOO test set ', tot, ', correct clf ', correct, ', acc ', acc)

    
# create dataframe for storing and presenting results
    model_2_rmsprop_ep150_results = pd.DataFrame({
        'Patient ID': ids,
        'True label': true_label_f2_b,
        'test set size': test_size_f2_b,
        'correctly classified': clf_correct_f2_b,
        'accuracy': accuracy_f2_b
    })

    print(model_2_rmsprop_ep150_results)
    
# save results to csv - change directory path according to your system
    f_name4 = 'model_2_rmsprop_ep150_results.csv'
    os.chdir('C:\\Users\\irene\\Desktop\\Birkbeck\\MSc_Project\\MSc project results')
    model_2_rmsprop_ep150_results.to_csv(f_name4, encoding='utf-8', index=False)
    print('Results saved in file')
    
    
main()
