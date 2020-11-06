##coding: utf-8

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
##MAtplotlib official documentation, available on line at: https://matplotlib.org/contents.html


##This file contains the code to analyse the classification esults of the models
##as saved from the code file: MSc_project_GroneI_DS_model.py.
##Change the directory path to the folder with all the files for
##the results before running the code.


## standard import
import numpy as np
import pandas as pd
import os
from os.path import dirname, join
import sys
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import sklearn
from sklearn.metrics import roc_curve


## file paths for result files
dir_path = 'C:\\Users\\irene\\Desktop\\Birkbeck\\MSc_Project\\MSc project results'


def classified_0(true_label, test, correct_clf):
    """
    Function that return the number of images classified as negative

    Args:
        true_label (int 0 or 1): true label
        test (int): size of test set for the patient
        correct_clf (int): how many images have been correclty classified
        
    Returns:
        val (int): images classified as negative (0)
    """

    if true_label == 0:
        val = correct_clf
    else:
        val = test - correct_clf

    return val


def classified_1(true_label, test, correct_clf):
    """
    Function that return the number of images classified as positive

    Args:
        true_label (int 0 or 1): true label
        test (int): size of test set for the patient
        correct_clf (int): how many images have been correclty classified
        
    Returns:
        val (int): images classified as positive (1)
    """
    
    if true_label == 1:
        val = correct_clf
    else:
        val = test - correct_clf

    return val


def clf_threshold(thr, test, clf_as_pos):
    """
    Function that return the classification label for the
    patient as a whole depending on the threshold choosen
    for positiveness
    example thr = 0.1 the patient is considered positive if at
    least 10% of the images have been classified as positive

    Args:
        thr (float): number between 0 and 1 that indicates threshold for classification as positive
        test (int): size of test set for the patient
        clf_as_1: number of images classified as positive
        
    Returns:
        label_thr
    """

    fraction_pos = clf_as_pos / test

    if fraction_pos < thr:
        label_thr = 0
    else:
        label_thr = 1

    return label_thr


def results_analysis(f_name):

    """
    Function that read result file as dataframe, create columns: 
        clf_0: how many images have been classified as 0
        clf_1: how many images have been classified as 1
        label_thr_10: label if threshold for positive is 10% of the images for the patient
        label_thr_50: label if threshold for positive is 50% of the images for the patient
       
    Args:
        f_name (string): file path of results file
        
    Returns:
        df (pd.DataFrame): dataframe with added columns
    """

    df = pd.read_csv(f_name)
    df['clf_0'] = df.apply(lambda x: classified_0(x['True label'], x['test set size'], x['correctly classified']), axis=1)
    df['clf_1'] = df.apply(lambda x: classified_1(x['True label'], x['test set size'], x['correctly classified']), axis=1)
    df['Predicted'] = df.apply(lambda x: clf_threshold(0.5, x['test set size'], x['clf_1']), axis=1)
    df['lab_thr_10'] = df.apply(lambda x: clf_threshold(0.1, x['test set size'], x['clf_1']), axis=1)
    df['prob_positive'] = df.apply(lambda x: x['clf_1'] / x['test set size'], axis=1) 

    # ROC curve
    y_true = df['True label'].to_numpy()
    y_probs = df['prob_positive'].to_numpy()

    title = f_name.split('\\')[-1].split('.')[0]
    
    fpr, tpr, thresholds = roc_curve(y_true, y_probs, pos_label=1)
    plt.title(title)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.plot(fpr, tpr)
    plt.plot(fpr, fpr, 'g--')
    plt.show()

    return df


def clf_metrics(P, N, TP, FP, TN, FN):
    """
    Function that calculates and returns classification metrics:  
    accuracy, precision, sensitivity, specificity

    Args:
      P, N, TP, FP, TN, FN

    Returns:
        accuracy (float)
        precision (float)
        sensitivity (float)
        specificity (float) 
    """

    accuracy = (TP + TN) / (P + N)
    precision = TP / (TP + FP)
    sensitivity = TP / P
    specificity = TN / N

    return (accuracy, precision, sensitivity, specificity)

def clf_confusion(result_df, title):
    """
    Function that calculates: TP, FP, TN, FN
    and classification metrics: accuracy, precision, sensitivity, specificity

    Args:
        result_df (pd.DataFrame): output of function results_analysis()
        title (string): name of title for plot, usually result file name

    Returns:
        
    """

    # count true labels
    P = sum(pd.eval("result_df['True label'] == 1"))
    N = sum(pd.eval("result_df['True label'] == 0"))

    # calculate TP, FP, TN, FN for threshold 0.5
    TP_50 = sum(pd.eval("(result_df['True label'] == 1) & (result_df['Predicted'] == 1)"))
    FP_50 = sum(pd.eval("(result_df['True label'] == 0) & (result_df['Predicted'] == 1)"))
    TN_50 = sum(pd.eval("(result_df['True label'] == 0) & (result_df['Predicted'] == 0)"))
    FN_50 = sum(pd.eval("(result_df['True label'] == 1) & (result_df['Predicted'] == 0)"))

    # calculate TP, FP, TN, FN for threshold 0.1
    TP_10 = sum(pd.eval("(result_df['True label'] == 1) & (result_df['lab_thr_10'] == 1)"))
    FP_10 = sum(pd.eval("(result_df['True label'] == 0) & (result_df['lab_thr_10'] == 1)"))
    TN_10 = sum(pd.eval("(result_df['True label'] == 0) & (result_df['lab_thr_10'] == 0)"))
    FN_10 = sum(pd.eval("(result_df['True label'] == 1) & (result_df['lab_thr_10'] == 0)"))

    # define confusin matrices
    confusion50 = [[TN_50, FN_50],[FP_50, TP_50]]
    confusion10 = [[TN_10, FN_10],[FP_10, TP_10]]

    # metrics
    met50 = clf_metrics(P, N, TP_50, FP_50, TN_50, FN_50)
    met10 = clf_metrics(P, N, TP_10, FP_10, TN_10, FN_10)

    print('accuracy, precision, sensitivity, specificity - for threshold 0.5 and 0.1')
    print(met50, met10, end='\n')
    
    # plot confusin matrices
    f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

    h1 = sns.heatmap(confusion50, ax=ax1, annot=True, cbar=False, cmap='Blues')
    h2 = sns.heatmap(confusion10, ax=ax2, annot=True, cbar=False, cmap='Blues')

    h1.set_title(' Threshold 0.5', fontsize=20)
    h1.set_xlabel('Actual', fontsize=15)
    h1.set_ylabel('Predicted', fontsize=15)

    h2.set_title(' Threshold 0.1', fontsize=20)
    h2.set_xlabel('Actual', fontsize=15)
    h2.set_ylabel('Predicted', fontsize=15)

    f.suptitle(title, y=1.0, fontsize=20)
    f.tight_layout()
    plt.show()
    
    return met50, met10


def main():

    tree = os.walk(dir_path)
    file_list = []
    f_names = []
    metrics_01 = []
    metrics_05 = []

    for dir_a, dir_b, files in tree:
        for f in files:
            p = os.path.join(str(dir_a), str(f))
            file_list.append(p)
            f_names.append(f.split('.')[0])

    clf_result_0 = results_analysis(file_list[0])
    m50, m10 = clf_confusion(clf_result_0, f_names[0])
    metrics_05.append(m50)
    metrics_01.append(m10)
    print(clf_result_0.to_markdown())
    input('Press Enter to continue to next result \n')
    
    clf_result_1 = results_analysis(file_list[1])
    m50, m10 = clf_confusion(clf_result_1, f_names[1])
    metrics_05.append(m50)
    metrics_01.append(m10)
    print(clf_result_1.to_markdown())
    input('Press Enter to continue to next result \n')

    clf_result_2 = results_analysis(file_list[2])
    m50, m10 = clf_confusion(clf_result_2, f_names[2])
    metrics_05.append(m50)
    metrics_01.append(m10)
    print(clf_result_2.to_markdown())
    input('Press Enter to continue to next result \n')

    clf_result_3 = results_analysis(file_list[3])
    m50, m10 = clf_confusion(clf_result_3, f_names[3])
    metrics_05.append(m50)
    metrics_01.append(m10)
    print(clf_result_3.to_markdown())
    input('Press Enter to continue to next result \n')

    df_metrics_05 = pd.DataFrame(metrics_05, columns=['accuracy', 'precision', 'sensitivity', 'specificity'], index=f_names)
    df_metrics_01 = pd.DataFrame(metrics_01, columns=['accuracy', 'precision', 'sensitivity', 'specificity'], index=f_names)

    print('Metrics threshold 0.5')
    print(df_metrics_05.to_markdown())
    print()
    print('Metrics threshold 0.1')
    print(df_metrics_01.to_markdown())
    
    return

main()
