# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#%%  Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#%%      Read dataset and plot
df = pd.read_csv("Salary_Data.csv")


plt.scatter(df.YearsExperience, df.Salary)
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()
#%% Linear Regression model and fitting


linear_reg = LinearRegression()

x = df.YearsExperience.values.reshape(-1,1)
y = df.Salary.values.reshape(-1,1)
linear_reg.fit(x, y)

#%%  prediction and y = b0+b1*x    finding b0 and b1
k = linear_reg.predict([[5]])
b0 = linear_reg.intercept_    # Intercept point
b1 = linear_reg.coef_         # Coefficient number of equation


#%% Plotting real values and fitted line plot

array = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]).reshape(-1,1)
plt.scatter(x, y)
plt.xlabel("Experience")
plt.ylabel("Salary")
y_head = linear_reg.predict(array)
plt.plot(array,y_head,color ="red")
plt.show()
#%%
