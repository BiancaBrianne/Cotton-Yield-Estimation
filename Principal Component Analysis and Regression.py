#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import scale 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn import model_selection


# In[4]:


cotton_data = pd.read_csv('Cotton_Data.csv')
num = cotton_data._get_numeric_data()
num[num < 0] = 0
cotton_data = pd.get_dummies(cotton_data, drop_first = True)
print(cotton_data.head())
print(cotton_data.info())


# In[5]:


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


# In[6]:


scaler = StandardScaler()
scaled_data = scaler.fit_transform(cotton_data)
scaled_data = pd.DataFrame(scaled_data, columns = cotton_data.columns)
X = scaled_data.drop('Yield',axis = 1)
y = scaled_data['Yield']


# In[7]:


pcamodel_ = PCA(13)
pca_ = pcamodel_.fit_transform(X)


# In[8]:


plt.scatter(pca_[:, 0], pca_[:, 1])
plt.xlabel('Component 1')
plt.ylabel('Component 2')


# In[9]:


PC_values = np.arange(pcamodel_.n_components_)+1
plt.plot(PC_values, pcamodel_.explained_variance_, 'o-', linewidth=2, color='red')
plt.title('Scree Plot')
plt.xlabel('Number of Components')
plt.ylabel('Percentage of Explained Variance')
plt.savefig('Scree Plot.jpg')
plt.show()


# In[10]:


loadings_df = pcamodel_.components_
num_pc = pcamodel_.n_features_
pc_list = ["PC"+str(i) for i in list(range(1, num_pc+1))]
loadings_df = pd.DataFrame.from_dict(dict(zip(pc_list, loadings_df)))
loadings_df['variable'] = X.columns.values
loadings_df = loadings_df.set_index('variable')

with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(loadings_df)


# In[12]:


pca2 = PCA()

# Split into 80/20 training and test subsets
X_train, X_test , y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=1)

#Perform PCA 
X_train_PC = pca2.fit_transform(X_train)
n = len(X_train_PC)

# 10-fold CV
kfold = model_selection.KFold( n_splits=10, shuffle=True, random_state=1)

mse = []
pcr = LinearRegression()
pcr.fit(X_train_PC, y_train)
# Calculate MSE with only the intercept (no principal components in pcression)
score = -1*model_selection.cross_val_score(pcr, np.ones((n,1)), y_train.ravel(), cv=kfold, scoring='neg_mean_squared_error').mean()    
mse.append(score)

# Calculate MSE using CV for the 50 principle components, adding one component at the time.
for i in np.arange(1, 15):
    score = -1*model_selection.cross_val_score(pcr, X_train_PC[:,:i], y_train.ravel(), cv=kfold, scoring='neg_mean_squared_error').mean()
    mse.append(score)

plt.plot(np.array(mse), '-v')
plt.xlabel('Number of principal components in pcression')
plt.ylabel('MSE')
plt.title('Number of Principal Components vs. MSE')
plt.xlim(xmin=-1);
plt.savefig('Components_vs_MSE.jpg')


# In[16]:


#Train pcression model on training data 
pcr = LinearRegression()
pcr.fit(X_train_PC[:,:6], y_train)

# Prediction with train data
pred = pcr.predict(X_train_PC[:,:6])
print('PCR MSE train:', mean_squared_error(y_train, pred))
print('PCR R-Squared train:',pcr.score(X_train_PC[:,:6], y_train))


# In[17]:


#Train pcression model on test data
X_reduced_test = pca2.fit_transform(X_test)

#Prediction with test data
pred2 = pcr.predict(X_reduced_test[:,:6])
print('PCR MSE test:', mean_squared_error(y_test, pred2))
print('PCR R-Squared test:',pcr.score(X_reduced_test[:,:6], y_test))

