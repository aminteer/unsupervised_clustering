#!/usr/bin/env python

import unittest
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import accuracy_score, confusion_matrix
import time
import pytest

def load_data():
    # Read data. Do not change the variable names (data, label)
    data = pd.read_csv('data.csv')
    label = pd.read_csv('labels.csv')
    data=data.drop('Unnamed: 0',axis=1)
    label=label.drop('Unnamed: 0',axis=1)
    return data, label

def create_histogram_by_feature(data):
    # min value for each feature
    min_df = data.min(axis=0)
    # max value for each feature
    max_df = data.max(axis=0)
    # average value for each feature
    avg_df = data.mean(axis=0)
    
    #as typical, directions are vague so we will assume it is mins, max, means accross all features
    
    # plot grid with 3 subplots
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    # plot histogram chart for min values
    axes[0].hist(min_df, bins=20)
    axes[0].set_title('Min Values')
    # plot for max values
    axes[1].hist(max_df, bins=20)
    axes[1].set_title('Max Values')
    # plotfor average values
    axes[2].hist(avg_df, bins=20)
    axes[2].set_title('Average Values')

    plt.tight_layout()
    # save to file
    fig.savefig('histogram.png')

def create_hierarchical_clustering_model(data, n_clusters=5, distance_threshold=None, linkage='average', metric='euclidean'):
    # create model
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage, distance_threshold=distance_threshold, metric=metric)
    # fit model
    model.fit(data)
    # return model
    return model

import itertools


def label_permute_compare(ytdf, yp, n=5):
    """
    ytdf: labels dataframe object
    yp: clustering label prediction output
    Returns permuted label order and accuracy. 
    Example output: (3, 4, 1, 2, 0), 0.74 
    """
    best_order = None
    best_accuracy = 0.0
    # generate all possible permutations of label order
    perms = itertools.permutations(range(n))
    # get labels into a grouped list
    yt_lables_grouped = ytdf.groupby(ytdf.columns[0]).size().index.tolist()
    # transform the labels dataframe into a numerical 1D array that matches to the index of the grouped list
    ytdf_index_numbers = np.array(ytdf.apply(lambda x: yt_lables_grouped.index(x[0]), axis=1))
    
    order_accuracy = []
    
    # iterate through each
    for order in perms:
        # permute predicted labels
        perm_labels = np.array([order[label] for label in yp])
        #assume permuted labels are the ground truth
        accuracy = accuracy_score(y_true=ytdf_index_numbers,y_pred=perm_labels)
        # update best accuracy and add to order
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_order = order
        # add result to temp list
        order_accuracy.append((order, accuracy))
    
    return best_order, best_accuracy

def get_confusion_matrix(ytdf, yp, label_reorder=None):
    """
    ytdf: labels dataframe object
    yp: clustering label prediction output
    Returns confusion matrix
    """
    # recode labels if a recode vector was provided
    if not label_reorder is None:
        yp = np.array([label_reorder[label] for label in yp])
    
    # get labels into a grouped list
    yt_lables_grouped = ytdf.groupby(ytdf.columns[0]).size().index.tolist()
    # transform the labels dataframe into a numerical 1D array that matches to the index of the grouped list
    ytdf_index_numbers = np.array(ytdf.apply(lambda x: yt_lables_grouped.index(x[0]), axis=1))
    # create confusion matrix
    cm = confusion_matrix(y_true=ytdf_index_numbers, y_pred=yp)
    return cm

def tune_distance_and_linkage_parameters(data, label):
    # use grid search to find the best distance metric and linkage parameters based on accuracy
    # create list of distance parameters
    distance_parameters = ['euclidean', 'manhattan', 'cosine', 'l1', 'l2']
    # create list of linkage type parameters
    linkage_parameters = ['ward', 'complete', 'average', 'single'] 
    
    # list to hold all results
    results = []
    # iterate through all combinations
    for distance in distance_parameters:
        for linkage in linkage_parameters:
            # create model
            # for Ward linkage, only euclidean distance is allowed so need this ugly workaround
            if linkage == 'ward' and distance != 'euclidean':
                continue
            
            model = create_hierarchical_clustering_model(data, n_clusters=5, linkage=linkage, metric=distance)
            # make predictions
            y_pred = model.labels_
            # get best label order
            order, accuracy = label_permute_compare(label, y_pred, 5)
            # update best accuracy
            result_iteration = (distance, linkage, order, accuracy)
            results.append(result_iteration)
    
    results_sorted = sorted(results, key=lambda x: x[3], reverse=True)
    
    return results_sorted
    
    

class ClusteringTest(unittest.TestCase): 
    def setUp(self):
        pass

if __name__ == "__main__":
    data, label = load_data()

    print("data first 5 records")
    print(data.head(5))
    print("label first 5 records")
    print(label.head(5))

    create_histogram_by_feature(data)

    print("chart created")
    
    model = create_hierarchical_clustering_model(data, n_clusters=5, linkage = 'average')

    print(model.labels_)
    
    #make predictions with model
    y_pred = model.fit_predict(data)
    print("predictions from model")
    print(y_pred)
    
    # get labels into a grouped list - this is ugly way to do it but necessary due to how things are setup - we'll fix later
    yt_lables_grouped = label.groupby(label.columns[0]).size().index.tolist()
    
    print("finding best label order")
    order, accuracy = label_permute_compare(label, model.labels_, 5)
    print(order, accuracy)
    
    print("getting confusion matrix")
    cm = get_confusion_matrix(label, model.labels_, order)
    print(cm)
    
    from sklearn.metrics import ConfusionMatrixDisplay
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=yt_lables_grouped)
    disp.plot()
    plt.show()
    
    #find best distance and linkage parameters
    print("\ntuning distance and linkage parameters")
    best_params = tune_distance_and_linkage_parameters(data, label)
    print(best_params[0])
    
    print("\n\nfull list of best parameters")
    print(best_params)
    
    print("Use K-Means clustering and compare accuracy and time to run vs hierarchical clustering")
    start = time.time()
    kmeans = KMeans(n_clusters=5, random_state=0).fit(data)
    end = time.time()
    km_order, km_accuracy = label_permute_compare(label, kmeans.labels_, 5)
    print("KMeans time to run: ", end - start)
    print("KMeans accuracy: ", km_accuracy)
    
    start = time.time()
    model = create_hierarchical_clustering_model(data, n_clusters=5, linkage = 'ward', metric='euclidean')
    end = time.time()
    print("Hierarchical time to run: ", end - start)
    hc_order, hc_accuracy = label_permute_compare(label, model.labels_, 5)
    print("Hierarchical accuracy: ", hc_accuracy)
    
    #create confusion matrix for Hierarchical clustering best parameters
    print("\n\ngetting confusion matrix for Hierarchical clustering best parameters")
    cm = get_confusion_matrix(label, model.labels_, hc_order)
    print(cm)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=yt_lables_grouped)
    disp.plot()
    plt.show()

    unittest.main()