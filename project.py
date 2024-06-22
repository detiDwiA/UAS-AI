import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
credit_card_data = pd.read_csv(r'dataset.csv')

# Display dataset info in Streamlit app
st.write("Informasi dataset")
st.write(credit_card_data.info())

# Display first few rows of the dataset
st.write("5 data teratas")
st.write(credit_card_data.head())

# Display the number of missing values in each column
st.write("data yang tidak memiliki value di dalam kolom")
st.write(credit_card_data.isnull().sum())

# Distribution of legit transactions & fraudulent transactions
st.write("palsu atau asli transaksi")
st.write(credit_card_data['Class'].value_counts())

# Data preprocessing
legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]

legit_sample = legit.sample(n=492)
new_dataset = pd.concat([legit_sample, fraud], axis=0)

# Splitting the data into features and targets
X = new_dataset[['Amount']]  # Use only 'Amount' feature
Y = new_dataset['Class']

# Splitting the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Training the model
model = LogisticRegression()
model.fit(X_train, Y_train)

# Streamlit UI
st.title("Pendeteksi kepalsuan kartu kredit")

st.write("Masukan data yang ingin di prediksi.")

# Create input field for 'Amount' feature
amount = st.number_input("masukkan Amount", format="%.2f")

# Convert input data to numpy array
input_data = np.array([amount]).reshape(1, -1)

# Check for near-zero value for 'Amount' (considering values less than 0.01 as zero)
if st.button('Prediksi sekarang'):
    if np.isclose(amount, 0, atol=0.01):
        st.error("Amount tidak boleh bernilai mendekati 0 (kurang dari 0.01)")
    else:
        prediction = model.predict(input_data)
        if prediction[0] == 0:
            st.success("Kartu Kredit Asli")
        else:
            st.error("Kartu Kredit Palsu")

# Show the accuracy of the model
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

st.write(f"tingkat akurasi data latih: {training_data_accuracy * 100:.2f}%")
st.write(f"tingkat akurasi test data: {test_data_accuracy * 100:.2f}%")
