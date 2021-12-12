# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 12:00:39 2021

@author: MONSTER
"""
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%%

df=pd.read_csv("Decision_Tree_Regression.csv",sep=";",)


#%%
x = df.iloc[:,0].values.reshape(-1,1)

y = df.iloc[:,1].values.reshape(-1,1)

#%%
from sklearn.tree import DecisionTreeRegressor
dec_tree_reg = DecisionTreeRegressor()
dec_tree_reg.fit(x, y)

new_x = np.arange(min(x),max(x),0.01).reshape(-1,1)
y_head = dec_tree_reg.predict(new_x)

#%%
plt.scatter(x,y,color="red")
plt.plot(new_x,y_head,color="green")


plt.xlabel("Level")
plt.ylabel("Value")
plt.show()

    #%%