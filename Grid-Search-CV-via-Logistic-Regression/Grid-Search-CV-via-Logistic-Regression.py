# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 11:27:42 2021

@author: MONSTER
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np

#%% loading dataset and reshaping for logistic regression

iris = load_iris()
x = iris.data[:100,:]
y = iris.target[:100]

#%% normalization

x = (x-np.min(x)) / (np.max(x) - np.min(x))

#%% train-test split
from sklearn.model_selection import train_test_split

x_train , x_test , y_train , y_test = train_test_split(x , y , test_size=0.3)

#%% Grid-Search CV with logistic-regression

from sklearn.linear_model import LogisticRegression

grid = {"C":np.logspace(-3,3,7),"penalty":["l1","l2"]}   # ---> l1 : losso , l2 : ridge

log_reg = LogisticRegression()
log_reg_cv = GridSearchCV(log_reg, grid , cv= 10 )
log_reg_cv.fit(x_train, y_train)

print("tuned hyperparameters  (best parameter ) : ",log_reg_cv.best_params_)
print("accuracy :",log_reg_cv.best_score_)


#%% Logistic Regression model test  with hyper parameter

log_reg_test = LogisticRegression(C=1 ,penalty="l2")
log_reg_test.fit(x_train, y_train)
print("Score of test : ",log_reg_test.score(x_test, y_test))
