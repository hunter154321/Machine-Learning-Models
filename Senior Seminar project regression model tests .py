#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# In[5]:


df = pd.read_csv('data.csv')
df


# In[6]:


df = df.drop(["id", "Unnamed: 32"], axis=1)
X = df.drop(['diagnosis'], axis=1)
df.diagnosis = [1 if each == "M" else 0 for each in df.diagnosis]
y = df['diagnosis']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=51)
df


# In[7]:


model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print(f'Confusion Matrix:\n{conf_matrix}')

# Classification Report
class_report = classification_report(y_test, y_pred)
print(f'Classification Report:\n{class_report}')


# requires too many iterations and can be improved with data normalization

# In[5]:


scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

model = LogisticRegression()
model.fit(X_train_scaled, y_train)


# In[6]:


y_pred = model.predict(X_test_scaled)


# In[7]:


# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print(f'Confusion Matrix:\n{conf_matrix}')

# Classification Report
class_report = classification_report(y_test, y_pred)
print(f'Classification Report:\n{class_report}')


# In[355]:


# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d")
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


# In[356]:


sns.jointplot(x=X['concavity_worst'], y=X['concave points_worst'], kind="reg", color="#ce1414")
plt.show()


# In[357]:


#Using a different training model
model = LogisticRegression(solver='liblinear')
model.fit(X_train_scaled, y_train_val_df)
y_pred = model.predict(X_test_scaled)
# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print(f'Confusion Matrix:\n{conf_matrix}')

# Classification Report
class_report = classification_report(y_test, y_pred)
print(f'Classification Report:\n{class_report}')


# In[358]:


#  drop all "worst" columns
cols = ['radius_worst', 
        'texture_worst', 
        'perimeter_worst', 
        'area_worst', 
        'smoothness_worst', 
        'compactness_worst', 
        'concavity_worst',
        'concave points_worst', 
        'symmetry_worst', 
        'fractal_dimension_worst']


# In[359]:


X1 = X.drop(cols, axis=1)  # Features
X1.info()


# In[360]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size=0.2, random_state=51)


# In[361]:


scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

model = LogisticRegression()
model.fit(X_train_scaled, y_train)


# In[362]:


y_pred = model.predict(X_test_scaled)


# In[363]:


# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print(f'Confusion Matrix:\n{conf_matrix}')

# Classification Report
class_report = classification_report(y_test, y_pred)
print(f'Classification Report:\n{class_report}')


# In[364]:


# drop all columns related to the "perimeter" and "area" attributes
cols = ['perimeter_mean',
        'perimeter_se', 
        'area_mean', 
        'area_se']
X2 = X.drop(cols, axis=1)

# drop all columns related to the "concavity" and "concave points" attributes
cols = ['concavity_mean',
        'concavity_se', 
        'concave points_mean', 
        'concave points_se']
X2 = X2.drop(cols, axis=1)


# In[365]:


X2.info()


# In[366]:


# Split the data into training and testing sets
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y, test_size=0.2, random_state=51)


# In[380]:


scaler = MinMaxScaler()
# Scale data 0-1
X2_train_scaled = StandardScaler().fit_transform(X2_train)
X2_test_scaled = StandardScaler().fit_transform(X2_test)

model = LogisticRegression()
model.fit(X2_train_scaled, y2_train)


# In[381]:


y2_pred = model.predict(X2_test_scaled)


# In[382]:


# Accuracy
accuracy = accuracy_score(y2_test, y2_pred)
print(f'Accuracy: {accuracy}')

# Confusion Matrix
conf_matrix = confusion_matrix(y2_test, y2_pred)
print(f'Confusion Matrix:\n{conf_matrix}')

# Classification Report
class_report = classification_report(y2_test, y2_pred)
print(f'Classification Report:\n{class_report}')
X2


# In[384]:


#Using a different training model
model = LogisticRegression(solver='liblinear')
model.fit(X2_train_scaled, y2_train)
y2_pred = model.predict(X2_test_scaled)
# Accuracy
accuracy = accuracy_score(y2_test, y2_pred)
print(f'Accuracy: {accuracy}')

# Confusion Matrix
conf_matrix = confusion_matrix(y2_test, y2_pred)
print(f'Confusion Matrix:\n{conf_matrix}')

# Classification Report
class_report = classification_report(y2_test, y2_pred)
print(f'Classification Report:\n{class_report}')


# In[8]:


cv_scores = cross_val_score(model, X_train, y_train, cv=10)
print("Cross-validation scores:", cv_scores)
print("Mean CV accuracy:", np.mean(cv_scores))


# In[ ]:




