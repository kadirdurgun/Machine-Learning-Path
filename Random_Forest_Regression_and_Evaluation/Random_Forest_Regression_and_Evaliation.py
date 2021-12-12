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

df=pd.read_csv("Random_Forest_Regression_and_Evaliation.csv",sep=";",)


#%%
x = df.iloc[:,0].values.reshape(-1,1)

y = df.iloc[:,1].values.reshape(-1,1)

#%%
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=100 ,random_state=42)
rf.fit(x,y)

y_head = rf.predict(x)

#%%

from sklearn.metrics import r2_score
print("r_2 score is : " , r2_score(y,y_head))



#%%