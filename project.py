#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# In[2]:
# loading the dataset to a Pandas DataFrame
credit_card_data = pd.read_csv(r'dataset.csv')
# credit_card_data = pd.read_csv(r'https://drive.google.com/uc?export=download&id=1ZjXpVKzYU-R5MVjE6_gdt532ur9IJ-9R')

# Membersihkan tanda kutip dari nama kolom (jika ada)
credit_card_data.columns = credit_card_data.columns.str.replace('"', '')

# In[3]:
# first 5 rows of the dataset
print(credit_card_data.head())

# In[4]:
print(credit_card_data.tail())

# In[5]:
# dataset informations
print(credit_card_data.info())

# In[6]:
# checking the number of missing values in each column
print(credit_card_data.isnull().sum())

# In[7]:
# Check if 'Class' column exists
if 'Class' in credit_card_data.columns:
    # distribution of legit transactions & fraudulent transactions
    print(credit_card_data['Class'].value_counts())

    # separating the data for analysis
    legit = credit_card_data[credit_card_data.Class == 0]
    fraud = credit_card_data[credit_card_data.Class == 1]

    print(legit.shape)
    print(fraud.shape)

    # statistical measures of the data
    print(legit.Amount.describe())
    print(fraud.Amount.describe())

    # compare the values for both transactions
    print(credit_card_data.groupby('Class').mean())

    legit_sample = legit.sample(n=492)
    new_dataset = pd.concat([legit_sample, fraud], axis=0)

    print(new_dataset.head())
    print(new_dataset.tail())
    print(new_dataset['Class'].value_counts())
    print(new_dataset.groupby('Class').mean())

    X = new_dataset.drop(columns='Class', axis=1)
    Y = new_dataset['Class']
else:
    print("Kolom 'Class' tidak ditemukan dalam dataframe.")
    # Lakukan operasi lain yang tidak memerlukan kolom 'Class'
    # Contoh: Menampilkan statistik deskriptif dari dataframe
    print("Statistik deskriptif dari dataframe:")
    print(credit_card_data.describe())
    
    # Untuk demonstrasi, kita lanjutkan dengan operasi lain. Misalnya, kita bisa membagi data menjadi fitur dan label lain jika ada
    # Dalam hal ini, kita bisa menggunakan kolom lain sebagai target jika ada, atau kita hanya menampilkan data saja.
    
    # Jika tidak ada kolom target, kita tidak bisa melanjutkan ke proses pelatihan model
    X = credit_card_data
    Y = None  # Tidak ada kolom target yang jelas

# Langkah-langkah berikutnya tergantung pada keberadaan 'Class' atau kolom target lain
if Y is not None:
    # Membagi data menjadi data latih dan data uji
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

    # In[19]:
    # Logistic Regression Model
    model = LogisticRegression()

    # training the Logistic Regression Model with Training Data
    model.fit(X_train, Y_train)

    # In[20]:
    # accuracy on training data
    X_train_prediction = model.predict(X_train)
    training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

    print('Akurasi pada data latih : ', training_data_accuracy)

    # In[21]:
    # accuracy on test data
    X_test_prediction = model.predict(X_test)
    test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

    print('Akurasi pada data uji : ', test_data_accuracy)
else:
    print("Tidak ada kolom target yang jelas untuk pelatihan model.")
