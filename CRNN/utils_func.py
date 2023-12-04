"""
@Author: Giulia Tanoni
@Date: April 2023
"""

import numpy as np
from sklearn.metrics import hamming_loss, precision_recall_curve, classification_report, roc_curve, auc
from matplotlib import pyplot as plt
from itertools import cycle
import scipy.signal
import random


def standardize_data(agg, mean, std):
    agg = agg - mean
    agg /= std
    return agg


def output_binarization(output, thres):
    print(np.min(output), np.max(output))

    new_output = np.where(output < thres, 0, 1)

    return new_output


def app_binarization_weak(output, thres, classes):
    new_output = np.where(output < thres, 0, 1)

    return new_output


def app_binarization_strong(output, thres, classes):
    new_output = np.where(output <= thres, 0, 1)

    return new_output


def thres_analysis(Y_test, new_output, classes):
    precision = dict()
    recall = dict()
    thres_list_strong = []

    for i in range(classes):

        if Y_test[0, i] == -2:
            thres_list_strong.append(0)
            continue
        precision[i], recall[i], thresh = precision_recall_curve(Y_test[:, i], new_output[:, i])
        """
        plt.title('Pres-Recall-THRES curve')
        plt.plot(precision[i], recall[i])
        plt.show()
        plt.close()
        """
        f1 = (2 * precision[i] * recall[i]) / (precision[i] + recall[i])
        opt_thres_f1 = np.argmax(f1)
        optimal_threshold_f1 = thresh[opt_thres_f1]
        print("Threshold for F1-SCORE value is:", optimal_threshold_f1)
        if (optimal_threshold_f1 >= 0.955 and i != 0) or (optimal_threshold_f1 >= 0.73 and i == 2):
            optimal_threshold_f1 = 0.3

        thres_list_strong.append(optimal_threshold_f1)

    return thres_list_strong


def weak_count(Y_train_weak):
    list_counter = [0, 0, 0, 0, 0]

    for i in range(len(Y_train_weak)):
        vect = Y_train_weak[i]
        for k in range(5):
            if vect[0][k] == 1:
                list_counter[k] += 1
    print("Weak composition:", list_counter)
