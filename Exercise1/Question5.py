import pandas as pd
import random
import numpy as np
import math

def get_train_test_data(dataset, i, k_folds):
    start_index = (len(dataset)/k_folds)*i 
    end_index = start_index + (len(dataset)/k_folds)
    train_data = dataset.drop(dataset.index[start_index : end_index])
    test_data = dataset.iloc[start_index : end_index]
    print "Train data", train_data
    print "Test data", test_data

    return train_data, test_data

def euclidean_distance(p1, p2):
    return (math.sqrt(((p1[0] - p2[0])**2) + ((p1[1] - p2[1])**2)))

def k_nearest_neighbour(test_data, train_data, k):
    train_data['distance'] = 0
    for index, data in train_data.iterrows():
        distance = euclidean_distance((test_data[1], test_data[2]), (data[1], data[2]))
        train_data.loc[index, 'distance'] = distance
    
    neighbours = train_data.nsmallest(k, 'distance')[3].tolist()

    male_count = neighbours.count(1)
    female_count = neighbours.count(-1)
    
    if male_count > female_count and test_data[3] == 1:
        return 1
    if female_count > male_count and test_data[3] == -1:
        return 1
    return 0

def t_times_k_fold_CV(t, k_folds, dataset, k):
    CV_accuracy = 0.0
    for i in range(t):
        dataset = dataset.sample(frac=1).reset_index(drop=True)
        accuracy = 0.0
        for j in range(k_folds):    
            train_data, test_data = get_train_test_data(dataset, j, k_folds)
            correct = 0
            for index, data in test_data.iterrows():
                correct = correct + k_nearest_neighbour(data, train_data, k)
            accuracy = accuracy + ((correct * 100.0) / (len(dataset)/k_folds))
        accuracy = accuracy / k_folds
        CV_accuracy = CV_accuracy + accuracy
    CV_accuracy = CV_accuracy / t
    return CV_accuracy

if __name__ == '__main__':

    dataset = pd.read_csv('DWH_Training.csv',header=None)
   
    k_folds = 10
    t = 1
    k = 3

    accuracy = t_times_k_fold_CV(t, k_folds, dataset, k)
    print accuracy
