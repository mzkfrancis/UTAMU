#!/usr/bin/env python
# coding: utf-8

# <h2> Model to pridict CO2Emissions in Cars</h2>

# <b2> By Amizuku Francis</b2>

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv("FuelConsumptionCo2.csv")


# In[3]:


df.head()


# In[4]:


df.tail()


# In[5]:


df.describe()


# In[7]:


df.shape


# In[13]:


cdf =df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]


# In[14]:


cdf.head()


# In[15]:


viz =cdf[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]
viz.hist()
plt.show()


# In[16]:


plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS, color ="blue")
plt.ylabel('CO2 Emission')
plt.xlabel('Engine Size')
plt.show()


# In[17]:


plt.scatter(cdf.CYLINDERS, cdf.CO2EMISSIONS, color ="blue")
plt.xlabel('CO2 Emission')
plt.ylabel('CYLINDERS')
plt.show()


# In[18]:


plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS, color ="blue")
plt.ylabel('CO2 Emission')
plt.xlabel('FUELCONSUMPTION_COMB')
plt.show()


# In[19]:


msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]


# In[21]:


from sklearn import linear_model
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit (train_x, train_y)
# The coefficients
print ('Coefficients: ', regr.coef_)
print ('Intercept: ',regr.intercept_)


# In[22]:


plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")


# In[23]:


from sklearn.metrics import r2_score

test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_hat = regr.predict(test_x)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_hat - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_hat - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y_hat , test_y) )


# In[24]:


df.corr


# In[25]:


from sklearn import linear_model
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit (train_x, train_y)
# The coefficients
print ('Coefficients: ', regr.coef_)
print ('Intercept: ',regr.intercept_)


# In[26]:


plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")


# In[28]:


from sklearn.metrics import r2_score

test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_hat = regr.predict(test_x)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_hat - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_hat - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y_hat , test_y) )


# In[ ]:




