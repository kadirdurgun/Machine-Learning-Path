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

#%% k-means clustering

from sklearn.cluster import KMeans
wcss = []

for k in range(1,15):
    k_means = KMeans(n_clusters=k)
    k_means.fit(data)
    wcss.append(k_means.inertia_)

plt.plot(range(1,15),wcss)
plt.xlabel("number of cluster (k)")
plt.ylabel("wcss")
plt.show()

# As we see from graph k=3 is maximum optimization for elbow method


#%% for k=3 , clustering

k_means2 = KMeans(n_clusters=3)
clusters = k_means2.fit_predict(data)
data["label" ]= clusters

plt.scatter(data.x[data.label == 0],data.y[data.label == 0], color="red")
plt.scatter(data.x[data.label == 1],data.y[data.label == 1], color="green")
plt.scatter(data.x[data.label == 2],data.y[data.label == 2], color="blue")

# centroids are shown on graph as yellow point
plt.scatter(k_means2.cluster_centers_[:,0],k_means2.cluster_centers_[:,1],color="yellow")
plt.show()









 