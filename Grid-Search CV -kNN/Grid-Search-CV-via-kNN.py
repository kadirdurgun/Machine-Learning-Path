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


#%% Grid-Search-CV
from sklearn.model_selection import GridSearchCV

grid = {"n_neighbors":np.arange(1,50)}

knn = KNeighborsClassifier()
knn_cv = GridSearchCV(knn, grid,cv = 10)

knn_cv.fit(x_train,y_train)

#%% Print hyperparameter of k value in KNN algorithm

print("tuned hyperparameter K :",knn_cv.best_params_)
print("best score with tuned hyperparameter K :",knn_cv.best_score_)


#%% knn model

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=4)
knn.fit(x_train, y_train)

print(" score of test data :",knn.score(x_test, y_test))














