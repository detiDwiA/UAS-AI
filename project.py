#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# loading the dataset to a Pandas DataFrame
credit_card_data = pd.read_csv('dataset.csv')

# first 5 rows of the dataset
print(credit_card_data.head())

# last 5 rows of the dataset
print(credit_card_data.tail())

# dataset informations
credit_card_data.info()

# checking the number of missing values in each column
print(credit_card_data.isnull().sum())

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

print(X)
print(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

print(X.shape, X_train.shape, X_test.shape)

model = LogisticRegression(max_iter=1000)

# training the Logistic Regression Model with Training Data
model.fit(X_train, Y_train)

# accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print('Accuracy on Training data : ', training_data_accuracy)

# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy score on Test Data : ', test_data_accuracy)

# Fungsi untuk memprediksi apakah transaksi baru adalah penipuan atau tidak
def predict_transaction(model, transaction):
    prediction = model.predict(transaction)
    if prediction == 0:
        print("Transaksi ini sah.")
    else:
        print("Transaksi ini penipuan.")

# Contoh data transaksi baru
# Gantilah nilai-nilai berikut dengan nilai fitur dari transaksi baru yang ingin Anda prediksi
new_transaction = np.array([[123.45, 1.234, -0.123, 0.456, -1.234, 2.345, 0.123, -0.456, 0.789, -1.567, 0.123, -0.345, 
                             0.567, -0.789, 1.234, -1.345, 0.456, -0.567, 0.678, -0.789, 0.890, -1.012, 0.345, -0.456, 
                             0.678, -0.567, 0.456, -0.789, 0.234, -0.345]])

# Melakukan prediksi pada transaksi baru
predict_transaction(model, new_transaction)
