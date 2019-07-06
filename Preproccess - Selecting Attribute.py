#!/usr/bin/env python
# coding: utf-8

# In[1]:
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import classification_report
from sklearn.svm import OneClassSVM
from sklearn.pipeline import Pipeline

# In[2]:
class dataset:
    pass
sample_data = pd.read_csv("D:\KULIAH\Semester 8\Dataset\Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv")
sample_data.to_pickle('D:\KULIAH\Semester 8\Dataset\Thursday-15-02-2018_TrafficForML_CICFlowMeter.pkl')

# In[3]:
df = pd.read_pickle('D:\KULIAH\Semester 8\Dataset\Thursday-15-02-2018_TrafficForML_CICFlowMeter.pkl')
df = df[['URG Flag Cnt','SYN Flag Cnt','RST Flag Cnt','PSH Flag Cnt','Protocol',
         'Pkt Size Avg','Flow Pkts/s','FIN Flag Cnt','ECE Flag Cnt','ACK Flag Cnt','Dst Port','Label']]
df["Flow Pkts/s"] = pd.to_numeric(df["Flow Pkts/s"], errors='coerce')
df.dropna(inplace=True)
df.info(verbose=True)

# In[5]:
dataset.train = df.groupby('Label')
                .apply(pd.DataFrame.sample, frac=0.8)
                .reset_index(level='Label', drop=True)
dataset.test = df.drop(dataset.train.index)
dataset.label = dataset.train.Label.copy()

# In[6]:
dataset.train

# In[7]:
dataset.label.unique()

# In[8]:
d1 = dataset.train.replace('Benign', 0)

# In[9]:
d2 = d1.replace('DoS attacks-GoldenEye', 1)

# In[10]:
d3 = d2.replace('DoS attacks-Slowloris', 1)

# In[11]:
d6_label = d3.Label.copy()

# In[12]:
d6_label.unique()

# In[13]:
d6_label.value_counts()

# In[14]:
dataset.test_label = dataset.test.Label.copy() #ra kanggo

# In[16]:
dataset.test_label.unique() #ra kanggo

# In[15]:
a1_label = dataset.test.Label.copy()

# In[16]:
a1_label.unique()

# In[17]:
a1 = dataset.test.replace('Benign', 0)

# In[18]:
a2 = a1.replace('DoS attacks-GoldenEye', 1)

# In[19]:
a3 = a2.replace('DoS attacks-Slowloris', 1)

# In[20]:
a5_label = a3.Label.copy()

# In[21]:
a5_label.unique()

# In[22]:
a5_label.value_counts()

# In[23]:
category_variables = ["Protocol"]
for cv in category_variables:
    d3[cv] = d3[cv].astype("category")
    a3[cv] = a3[cv].astype("category")
    
    print("Length of Categories for {} are {}".format(cv , len(d3[cv].cat.categories)))
    print("Categories for {} are {} \n".format(cv ,d3[cv].cat.categories))

# In[24]:
dummy_variables_2labels = category_variables
   
class preprocessing:
    train_labels = pd.get_dummies(d3, columns = dummy_variables_2labels, prefix=dummy_variables_2labels)
    test_labels = pd.get_dummies(a3, columns = dummy_variables_2labels, prefix=dummy_variables_2labels)

# In[27]:
preprocessing.test_labels.info(verbose=True)

# In[29]:
d3.head()

# In[28]:
preprocessing.test_labels.head()

# In[47]:
preprocessing.train_labels.to_csv("D:\KULIAH\Semester 8\Dataset\preprocessed_train_4_new.csv")
preprocessing.test_labels.to_csv("D:\KULIAH\Semester 8\Dataset\preprocessed_test_4_new.csv")
