# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 11:44:27 2021

@author: MONSTER
"""

#%% import libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%% Creating dataset

#Class 1
x1 = np.random.normal(25,7,1000)
y1 = np.random.normal(25,7,1000)

#Class 2
x2 = np.random.normal(55,5,1000)
y2 = np.random.normal(70,5,1000)

#Class 3
x3 = np.random.normal(65,4,1000)
y3 = np.random.normal(20,4,1000)

x = np.concatenate((x1,x2,x3),axis = 0)
y = np.concatenate((y1,y2,y3),axis = 0)

dictionary = {"x":x , "y":y}

data =  pd.DataFrame(dictionary)


plt.scatter(x1, y1 , color="black")
plt.scatter(x2, y2 , color="black")
plt.scatter(x3, y3 , color="black")
plt.show()

#%% Dendrogram
from scipy.cluster.hierarchy import linkage,dendrogram

merg = linkage(data,method="ward")
dendrogram(merg,leaf_rotation=90)
plt.xlabel("data points")
plt.ylabel("eucledian distance")
plt.show()

# As seen from dendogram , we can apply the cluster number as 3 !!!!

#%% Hierarchical Clustering
from sklearn.cluster import AgglomerativeClustering

hierarchical_cluster = AgglomerativeClustering(n_clusters=3,affinity="euclidean",linkage="ward")
clusters = hierarchical_cluster.fit_predict(data)

data["label"] = clusters

plt.scatter(data.x[data.label == 0],data.y[data.label == 0],color ="red")
plt.scatter(data.x[data.label == 1],data.y[data.label == 1],color ="green")
plt.scatter(data.x[data.label == 2],data.y[data.label == 2],color ="blue")
plt.show()


 