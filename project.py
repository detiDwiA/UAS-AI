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
st.write("Data yang tidak memiliki value di dalam kolom")
st.write(credit_card_data.isnull().sum())

# Distribution of legit transactions & fraudulent transactions
st.write("Distribusi Kelas Transaksi")
st.write(credit_card_data['Class'].value_counts())

# Data preprocessing
legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]

# Ensure there are enough samples of each class
if len(legit) < 450 or len(fraud) < 450:
    st.error("Dataset tidak memiliki cukup sampel untuk setiap kelas.")
else:
    legit_sample = legit.sample(n=450, random_state=42)
    fraud_sample = fraud.sample(n=450, random_state=42)

    # Create a new dataset with balanced classes
    new_dataset = pd.concat([legit_sample, fraud_sample], axis=0).reset_index(drop=True)
    new_dataset = new_dataset.sample(frac=1, random_state=42).reset_index(drop=True)

    # Display the number of samples in each class in the new dataset
    st.write("Distribusi Kelas Setelah Sampling")
    st.write(new_dataset['Class'].value_counts())

    # Splitting the data into features and targets
    X = new_dataset.drop(columns='Class', axis=1)
    Y = new_dataset['Class']

    # Splitting the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

    # Training the model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, Y_train)

    # Streamlit UI
    st.title("Pendeteksi Kepalsuan Kartu Kredit")

    st.write("Masukkan data yang ingin diprediksi dalam format: time,v1,v2,...,v28,amount")
    st.write("Contoh input: 0,-1.359807,-0.072781,2.536346,1.378155,-0.338321,0.462388,0.239599,0.098698,0.363787,0.090794,-0.5516,-0.6178,0.991389,-0.311169,1.468177,-0.4704,0.207971,0.025791,0.403993,0.251412,-0.018307,0.277838,-0.110473,0.066928,0.128539,-0.189115,0.133558,-0.021053,149.62")

    # Create input field for user to enter comma-separated values
    user_input = st.text_input("Masukkan data:")

    # Create Enter button
    if st.button('Enter'):
        if user_input:
            # Split the input string by commas
            input_values = user_input.split(',')

            # Ensure there are exactly 30 input values
            if len(input_values) != 30:
                st.error("Masukkan harus terdiri dari 30 nilai yang dipisahkan dengan koma.")
            else:
                # Convert input values to numpy array
                try:
                    input_data = np.array(input_values, dtype=float).reshape(1, -1)
                    prediction = model.predict(input_data)

                    if prediction[0] == 0:
                        st.success("Kartu Kredit Asli")
                    else:
                        st.error("Kartu Kredit Palsu")
                except ValueError:
                    st.error("Pastikan semua nilai input adalah angka yang valid.")

    # Show the accuracy of the model
    X_train_prediction = model.predict(X_train)
    training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
    X_test_prediction = model.predict(X_test)
    test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

    st.write(f"Tingkat akurasi data latih: {training_data_accuracy * 100:.2f}%")
    st.write(f"Tingkat akurasi test data: {test_data_accuracy * 100:.2f}%")
