# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 09:15:45 2021

@author: MONSTER
"""
#%% Loading dataset and creating of dataframe
from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()

data = iris.data
feature_names = iris.feature_names
y = iris.target

df = pd.DataFrame(data,columns=feature_names)
df["Class"]=y



#%% PCA

from sklearn.decomposition import PCA

pca = PCA(n_components=2,whiten=True) # whiten means that normalization of data

pca.fit(data)

x_pca = pca.transform(data)


print("variance ratio:" ,pca.explained_variance_ratio_)

print("sum of variance:" , sum(pca.explained_variance_ratio_))

#%% 2-D visualization


df["p1"] = x_pca[:,0]  #principal component
df["p2"] = x_pca[:,1]  #second component
color = ["red","green","blue"]

import matplotlib.pyplot as plt
for each in range(3):
    plt.scatter(df.p1[df.Class == each],df.p2[df.Class == each],color = color[each],label = iris.target_names[each])

plt.legend()
plt.xlabel("p1(principal_component)")
plt.ylabel("p2(second_component)")
plt.show()