# Cotton Yield Estimation
<img align="right" src="flowering stage cotton.jpg" />

This project trains and compares four machine learning models for predicting the cotton yield in an experimental field. The training and testing data were computed using surface models created from photos taken by UAVs. This collection method provides an alternative to satellite photos which can be obscured. Models based on time series data over the first 67 weeks of the growing season and the full growing season were used to train the models. Ultimately the LASSO regression model (R-squared = 0.81) was the best predictor for the entire season and the PLSR for the partial season (R-squared = 0.70), This repository includes:

* Data Analysis
  - Bar plots
  - Box plots
  - Scatter plots
  - Principal Component Analysis
  - Heat Map Correlation Matrix
* Machine Learning Models 
  - Partial Least Squares Regression (PLSR)
  - Least Absolute Shrinkage and Selection Operator Regression (LASSO)
  - Principal Component Regression (PCR)
  - Ridge Regression
* Model Selection
  - 80/20 training and testing split
  - Hyperparameter tuning

