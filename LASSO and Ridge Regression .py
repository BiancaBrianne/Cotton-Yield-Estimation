#!/usr/bin/env python
# coding: utf-8

# In[76]:


get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import scale 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler


# In[78]:


cotton_data = pd.read_csv('Cotton_Data.csv')
num = cotton_data._get_numeric_data()
num[num < 0] = 0
cotton_data = pd.get_dummies(cotton_data, drop_first = True)
print(cotton_data.head())
print(cotton_data.info())


# In[79]:


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


# #Drop outliers
# cotton_data = cotton_data.drop([360,497,557,558,747,613,809])

# In[81]:


#scale data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(cotton_data)
scaled_data = pd.DataFrame(scaled_data, columns = cotton_data.columns)
X = scaled_data.drop('Yield',axis = 1)
y = scaled_data['Yield']


# In[82]:


# Split data into training and test sets
X_train, X_test, y_train, y_test  = train_test_split(X, y, train_size = 0.80, random_state = 35)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size = .5 , random_state = 35)


# In[83]:


print(X_train.shape)
print(X_test.shape)
print(X_val.shape)
print(X.shape)


# In[84]:


lambdas = 10**np.linspace(10,-2,100)*0.5


# In[85]:


ridge = Ridge()
coefs = []

for a in lambdas:
    ridge.set_params(alpha = a)
    ridge.fit(X_val, y_val)
    coefs.append(ridge.coef_)
    
np.shape(coefs)


# In[86]:


#Plot coefficients as lambda gets very large
ax = plt.gca()
ax.plot(lambdas, coefs)
ax.set_xscale('log')
plt.axis('tight')
plt.xlabel('Lambda')
plt.ylabel('Coefficients')


# In[87]:


#Use 10-fold CV to find appropriate lambda
ridgecv = RidgeCV(alphas = lambdas,cv = 10, scoring = 'neg_mean_squared_error', normalize = True)
ridgecv.fit(X_val, y_val)
ridgecv.alpha_


# In[1]:


#Train model with selected lambda and training set
ridge = Ridge(alpha = ridgecv.alpha_, normalize = True)
ridge.fit(X_train, y_train)


# In[89]:


#Create dataframe with ridge coefficients and print 
ridge_coefficients = pd.Series(ridge.coef_, index = X.columns)
print(ridge_coefficients)


# In[90]:


#Plot LASSO coefficients as lambdas get very large
lasso = Lasso(max_iter = 10000)
coefs = []
for a in lambdas:
    lasso.set_params(alpha=a)
    lasso.fit(scale(X_val), y_val)
    coefs.append(lasso.coef_)
    
ax = plt.gca()
ax.plot(lambdas*2, coefs)
ax.set_xscale('log')
plt.axis('tight')
plt.xlabel('Lambda')
plt.ylabel('weights')


# In[91]:


#Use 10-fold CV to find appropriate alpha using validation set
lassocv = LassoCV(alphas = None, cv = 10, max_iter = 100000, normalize = True)
lassocv.fit(X_val, y_val)
#Set lambda parameter and fit training data to model
lasso.set_params(alpha=lassocv.alpha_)
lasso.fit(X_train, y_train)


# In[92]:


#Print LASSO coefficients
lasso_coefficients = pd.Series(lasso.coef_, index=X.columns)
print(lasso_coefficients)


# In[93]:


#Print MSE and R-squared for training
print(f"LASSO r-squared {lasso.score(X_train, y_train):.3f}")
print(f"Ridge r-squared {ridge.score(X_train, y_train):.3f}")
print('LASSO MSE:',mean_squared_error(y_train, lasso.predict(X_train)))
print('Ridge MSE:',mean_squared_error(y_train, ridge.predict(X_train)))


# In[94]:


#Print MSE and R-squared for testing data
print(f"LASSO r-squared {lasso.score(X_test, y_test):.3f}")
print(f"Ridge r-squared {ridge.score(X_test, y_test):.3f}")
print('LASSO MSE:',mean_squared_error(y_test, lasso.predict(X_test)))
print('Ridge MSE:',mean_squared_error(y_test, ridge.predict(X_test)))


# In[ ]:




