# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 13:19:20 2021

@author: MONSTER
"""
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics


#%%
df = pd.read_csv("50_Startups.csv")

#%%
y = df.Profit.values.reshape(-1,1)
x = df.iloc[:,[0,1,2]].values
(trainX, testX, trainY, testY) = train_test_split(x, y, test_size=0.25, random_state=0)

print("X Train shape:", trainX.shape)
print("X Test shape:", testX.shape)
print("Y Train shape:", trainY.shape)
print("Y Test shape:", testY.shape)
#%%

multiple_linear_reg =LinearRegression()
multiple_linear_reg.fit(trainX,trainY)

#%%
y_predict = multiple_linear_reg.predict(testX)


#%%

df_comparised = pd.DataFrame({'Actual': testY.flatten(), 'Predicted': y_predict.flatten()})


df_comparised.plot(kind = "bar")
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='red')
plt.ylabel("Profit")
plt.xlabel("Actual and Predicted Datas")
plt.title("Actual and Predicted Data Comparison")

print('Mean Absolute Error:', metrics.mean_absolute_error(testY, y_predict))  
print('Mean Squared Error:', metrics.mean_squared_error(testY, y_predict))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(testY, y_predict)))

#%%