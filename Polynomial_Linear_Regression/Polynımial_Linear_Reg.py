# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 09:29:49 2021

@author: MONSTER
"""
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
#%%

df = pd.read_csv("Position_Salaries.csv")

#%%

x =df.Level.values.reshape(-1,1)
y=df.Salary.values.reshape(-1,1)


#%%


lr = LinearRegression()
lr2 = LinearRegression()
lr4 = LinearRegression()



#%%
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=2)

x_deg2 =poly_reg.fit_transform(x)
lr2.fit(x_deg2,y)
y_head2 =lr2.predict(x_deg2)

poly_reg_4 =PolynomialFeatures(degree=4)
x_deg4 = poly_reg_4.fit_transform(x)
lr4.fit(x_deg4,y)
y_head4 = lr4.predict(x_deg4)

lr.fit(x,y)
y_head = lr.predict(x)

plt.scatter(x,y , color="black")
plt.plot(x,y_head,color="blue",label="Linear Reg")
plt.plot(x,y_head2 ,color="green",label="2nd Degree Poly")
plt.plot(x,y_head4,color="red",label="4th Degree Poly")


plt.grid(color='0.95')
plt.legend(title=' Degree of ')
plt.title('Polynomial Reg with  ... Degree')
plt.xlabel("Level")
plt.ylabel("Salaries")
plt.show()
#%%
