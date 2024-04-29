#!/usr/bin/env python
# coding: utf-8

# In[18]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import xgboost as xgb
from xgboost import XGBClassifier


# In[19]:


df = pd.read_csv('data.csv')
df


# In[20]:


df = df.drop(["id", "Unnamed: 32"], axis=1)
X = df.drop(['diagnosis'], axis=1)
y = df['diagnosis']  # Target variable
y = y.map({'B': 0, 'M': 1})

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=51)


# In[21]:


dtrain = xgb.DMatrix(x_train, label=y_train)
dtest = xgb.DMatrix(x_test, label=y_test)
# Set parameters for XGBoost
params = {
    'max_depth': 3,
    'objective': 'multi:softmax',  # for multiclass classification
    'num_class': len(np.unique(y_train)),  # number of classes in the dataset
    'eta': 0.1,
}
# Train the XGBoost model and get predictions
model = xgb.train(params, dtrain, num_boost_round=100)
predictions = model.predict(dtest)


# In[22]:


accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)


# In[23]:


scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)


# In[24]:


dtrain = xgb.DMatrix(X_train_scaled, label=y_train)
dtest = xgb.DMatrix(X_test_scaled, label=y_test)
# Set parameters for XGBoost
params = {
    'max_depth': 3,
    'objective': 'multi:softmax',  # for multiclass classification
    'num_class': len(np.unique(y_train)),  # number of classes in the dataset
    'eta': 0.1,
}
# Train the XGBoost model and get predictions
model = xgb.train(params, dtrain, num_boost_round=100)
predictions = model.predict(dtest)


# In[25]:


accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)


# In[38]:


xgb_model = XGBClassifier()
cv_scores = cross_val_score(xgb_model, X_train, y_train, cv=10)
print("Cross-validation scores:", cv_scores)
print("Mean CV accuracy:", np.mean(cv_scores))


# In[39]:


# Train the model on the full training set
xgb_model.fit(X_train, y_train)

# Evaluate the model on the test set
test_accuracy = xgb_model.score(X_test, y_test)
print("Test accuracy:", test_accuracy)


# In[40]:


from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],  # Number of trees
    'max_depth': [3, 5, 7],            # Maximum depth of a tree
    'learning_rate': [0.1, 0.01, 0.001] # Learning rate
}
xgb_model = XGBClassifier()
# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=5, scoring='accuracy')

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Get the best parameters and best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("Best parameters:", best_params)
print("Best cross-validation accuracy:", best_score)

# Train the model with the best parameters
best_xgb_model = XGBClassifier(**best_params)
best_xgb_model.fit(X_train, y_train)

# Evaluate the model on the test set
test_accuracy = best_xgb_model.score(X_test, y_test)
print("Test accuracy with best parameters:", test_accuracy)


# In[ ]:




