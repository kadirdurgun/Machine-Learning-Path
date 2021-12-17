# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 22:09:18 2021

@author: MONSTER
"""

#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#%%

data = pd.read_csv("column_2C_weka.csv")

#%%
data["class"] = [0 if each == "Abnormal" else 1 for each in data["class"]]
y = data["class"].values.reshape(-1,1)

x_data = data.drop(["class"],axis=1)

#%%

x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data))

#%%
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y , test_size=0.3,random_state=1)


#%%

from sklearn.neighbors import KNeighborsClassifier


#%%
score_list = []
for each in range(1,25):
    knn2 = KNeighborsClassifier(n_neighbors = each)
    knn2.fit(x_train,y_train)
    score_list.append(knn2.score(x_test,y_test))
plt.plot(range(1,25),score_list)
plt.xlabel("k values")
plt.ylabel("accuracy")
plt.show()
#%%

