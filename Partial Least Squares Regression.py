#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

from sklearn.preprocessing import scale 
from sklearn import model_selection
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score


# In[3]:


cotton_data = pd.read_csv('Cotton_Data.csv')
num = cotton_data._get_numeric_data()
num[num < 0] = 0
cotton_data = pd.get_dummies(cotton_data, drop_first = True)
print(cotton_data.head())
print(cotton_data.info())


# In[4]:


#First quartile
response = cotton_data['Yield']
quantile_1 = np.quantile(response, 0.25)

#Third quartile
quantile_3 = np.quantile(response, 0.75)
med = np.median(response)

#Interquantile range
iqr = quantile_3-quantile_1

#Upper and lower whiskers
#Define cut off for outliers
#1.5*iqr above third quantile and 1.5*iqr below first quantile
upper_bound = quantile_3+(1.5*iqr)
lower_bound = quantile_1-(1.5*iqr)

#Find outliers given cut off defined above
outliers = response[(response <= lower_bound) | (response >= upper_bound)]

#Display quantiles and outliers
print('First Quantile:\n{}'.format(quantile_1), '\n')
print('Third Quantile:\n{}'.format(quantile_3), '\n')
print('Interquantile range:\n{}'.format(iqr), '\n')
print('Upper bound:\n{}'.format(upper_bound), '\n')
print('Lower bound:\n{}'.format(lower_bound), '\n')
print('Outliers:\n{}'.format(outliers))


# cotton_data = cotton_data.drop([360,497,557,558,747,613,809])

# In[5]:


#scale data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(cotton_data)
scaled_data = pd.DataFrame(scaled_data, columns = cotton_data.columns)
#Separate predictor and response variables
X = scaled_data.drop('Yield',axis = 1)
y = scaled_data['Yield']


# In[6]:


#10-Fold CV
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
#Calculate MSE for 1 latent variable
mse = []
n = len(X)
score = -1*model_selection.cross_val_score(PLSRegression(n_components=1),
           np.ones((n,1)), y, cv=cv, scoring='neg_mean_squared_error').mean()    
mse.append(score)


# In[10]:


#Calculate MSE for each latent variable added to PLSR
for i in np.arange(1, 30):
    pls = PLSRegression(n_components=i)
    score = -1*model_selection.cross_val_score(pls, scale(X), y, cv=cv,
               scoring='neg_mean_squared_error').mean()
    mse.append(score)


# In[11]:


#Plot MSE vs Number of PLS components
plt.plot(mse)
plt.xlabel('Number of PLS Components')
plt.ylabel('MSE')


# In[13]:


#80/20 train/test split
X_train, X_test, y_train, y_test  = train_test_split(X, y, train_size = 0.80, random_state = 35)


# In[14]:


#Fit PLSR with 13 latent variables
pls = PLSRegression(n_components=13)
pls.fit(X_train, y_train)


# In[16]:


#Print training data scores
print('PLS Train MSE:',mean_squared_error(y_train, pls.predict(X_train)))
print('PLS Train R-squared:', r2_score(y_train, pls.predict(X_train)))


# In[20]:


#Print testing data scores
print('PLS Test MSE:',mean_squared_error(y_test, pls.predict(X_test)))
print(f"PLS Test R-squared {pls.score(X_test, y_test):.3f}")

