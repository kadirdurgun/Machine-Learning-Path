# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 10:26:53 2021

@author: MONSTER
"""

from sklearn.datasets import load_iris
import pandas as pd
import numpy as np

#%% loading dataset

iris = load_iris()
x = iris.data
y = iris.target

#%% normalization

x = (x-np.min(x)) / (np.max(x) - np.min(x))

#%% train-test split
from sklearn.model_selection import train_test_split

x_train , x_test , y_train , y_test = train_test_split(x , y , test_size=0.3)

#%% knn model

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)
#%% K-Fold Cross Validation   K = 10

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = knn , X = x_train , y = y_train , cv = 10)   

print("average accuracy : ",np.mean(accuracies))
print("average std : ",np.std(accuracies))

#%% Test of model with test data

print("Accuracy of test data : ",knn.score(x_test, y_test))