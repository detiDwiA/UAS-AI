import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
credit_card_data = pd.read_csv(r'dataset.csv')

# Display dataset info in Streamlit app
st.write("### Dataset Information")
st.write(credit_card_data.info())

# Display first few rows of the dataset
st.write("### First 5 Rows of the Dataset")
st.write(credit_card_data.head())

# Display the number of missing values in each column
st.write("### Missing Values in Each Column")
st.write(credit_card_data.isnull().sum())

# Distribution of legit transactions & fraudulent transactions
st.write("### Distribution of Legit & Fraudulent Transactions")
st.write(credit_card_data['Class'].value_counts())

# Data preprocessing
legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]

legit_sample = legit.sample(n=492)
new_dataset = pd.concat([legit_sample, fraud], axis=0)

# Splitting the data into features and targets
X = new_dataset.drop(columns='Class', axis=1)
Y = new_dataset['Class']

# Splitting the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Training the model
model = LogisticRegression()
model.fit(X_train, Y_train)

# Streamlit UI
st.title("Credit Card Fraud Detection")

st.write("Enter the transaction details below to predict if it's fraudulent or not.")

# Create input fields for each feature
input_data = []
for column in X.columns:
    value = st.number_input(f"Enter {column}", format="%.2f")
    input_data.append(value)

# Convert input data to numpy array
input_data = np.array(input_data).reshape(1, -1)

# Prediction
if st.button('Predict'):
    prediction = model.predict(input_data)
    if prediction[0] == 0:
        st.success("The transaction is legitimate.")
    else:
        st.error("The transaction is fraudulent.")

# Show the accuracy of the model
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

st.write(f"Accuracy on Training data: {training_data_accuracy * 100:.2f}%")
st.write(f"Accuracy on Test data: {test_data_accuracy * 100:.2f}%")
