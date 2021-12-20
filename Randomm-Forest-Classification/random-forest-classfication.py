# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 10:52:45 2021

@author: MONSTER
"""

#%% İmport Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%% Read data

data = pd.read_csv("data.csv")

data.drop(["id","Unnamed: 32"],axis= 1,inplace=True)

#%% Category 
M = data[data.diagnosis == "M"]
B = data[data.diagnosis == "B"]

#%% Scatter plot
plt.scatter(M.radius_mean, M.texture_mean, color ="red",label="Melignant",alpha=0.4)
plt.scatter(B.radius_mean, B.texture_mean, color ="green",label="Benign",alpha=0.4)    
plt.xlabel("Radius Mean")
plt.ylabel("Texture Mean")
plt.legend()
plt.show()

#%% Data conversion
data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]
y = data.diagnosis.values
x_data = data.drop(["diagnosis"],axis = 1 )

#%% Normalization
x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data))

#%% train-test-split

from sklearn.model_selection import train_test_split

x_train, x_test, y_train , y_test = train_test_split(x,y,test_size=0.3,random_state=42)

#%% Decision Tree model
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(x_train,y_train)

print ("Decision tree score : ",dt.score(x_test, y_test))

#%%
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=150,random_state=12)
rf.fit(x_train,y_train)

print("Random Forest score is ",rf.score(x_test,y_test))











