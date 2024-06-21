#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[2]:


# loading the dataset to a Pandas DataFrame
credit_card_data = pd.read_csv(r'creditcard.csv')


# In[3]:


# first 5 rows of the dataset
credit_card_data.head()


# In[4]:


credit_card_data.tail()


# In[5]:


# dataset informations
credit_card_data.info()


# In[6]:


# checking the number of missing values in each column
credit_card_data.isnull().sum()


# In[7]:


# distribution of legit transactions & fraudulent transactions
credit_card_data['Class'].value_counts()


# In[8]:


# separating the data for analysis
legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]


# In[9]:


print(legit.shape)
print(fraud.shape)


# In[10]:


# statistical measures of the data
legit.Amount.describe()


# In[11]:


fraud.Amount.describe()


# In[12]:


# compare the values for both transactions
credit_card_data.groupby('Class').mean()


# In[13]:


legit_sample = legit.sample(n=492)


# In[14]:


new_dataset = pd.concat([legit_sample, fraud], axis=0)


# In[15]:


new_dataset.head()


# In[16]:


new_dataset.tail()


# In[17]:


new_dataset['Class'].value_counts()


# In[18]:


new_dataset.groupby('Class').mean()


# In[19]:


X = new_dataset.drop(columns='Class', axis=1)
Y = new_dataset['Class']

