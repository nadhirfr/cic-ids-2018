#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
 
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.externals import joblib 


# In[2]:


traindata = pd.read_csv('D:\KULIAH\Semester 8\Dataset\preprocessed_train_4_new.csv')
testdata = pd.read_csv('D:\KULIAH\Semester 8\Dataset\preprocessed_test_4_new.csv')

traindata.pop("Unnamed: 0")
testdata.pop("Unnamed: 0")

traindata.head()


# In[3]:


Y = traindata.pop('Label')
X = traindata.iloc[:,0:13]
C = testdata.pop('Label')
T = testdata.iloc[:,0:13]

X.info(verbose=True)


# In[4]:


_traindata = np.array(X)
_trainlabel = np.array(Y)

_testdata = np.array(T)
_testlabel = np.array(C)

_testlabel


# In[5]:


# 5. Declare data preprocessing steps
pipeline = make_pipeline(RandomForestClassifier())

# Add a dict of estimator and estimator related parameters in this list
hyperparameters = {
                'randomforestclassifier__n_estimators': [25,50,75,100],
                'randomforestclassifier__max_features' : [None, "log2", "auto"]
                }


# In[6]:


# 7. Tune model using cross-validation pipeline
clf = GridSearchCV(pipeline, hyperparameters, cv=5,verbose=1,n_jobs=-1)
clf.fit(_traindata, _trainlabel)


# In[7]:


print(clf.best_params_)
print(clf.best_estimator_)
# print(clf.cv_results_ )


# In[8]:


print(clf.best_score_ )


# In[9]:


print (clf.refit)
 
# 9. Evaluate model pipeline on test data
pred = clf.predict(_testdata)


from sklearn.metrics import accuracy_score
print(accuracy_score(_testlabel, pred))


# In[10]:


from sklearn.metrics import confusion_matrix,classification_report

cm = confusion_matrix(_testlabel, pred)
print(classification_report(_testlabel, pred))
print(cm)


# In[11]:


# 10. Save model for future use
joblib.dump(clf, 'rf_gridcv_tanpa_scaler_100-est-log2.pkl')


# In[9]:


clf2 = joblib.load('v-rf_gridcv_robust.pkl')


# In[10]:


print(clf2.best_params_)
print(clf2.best_estimator_)

