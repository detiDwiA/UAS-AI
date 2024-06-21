import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
credit_card_data = pd.read_csv('dataset.csv')

# Check if the dataset is loaded correctly
print(credit_card_data.head())
print(credit_card_data.info())

# Check if 'Class' column exists
if 'Class' in credit_card_data.columns:
    # Distribution of legit transactions & fraudulent transactions
    print(credit_card_data['Class'].value_counts())

    # Separate the data for analysis
    legit = credit_card_data[credit_card_data.Class == 0]
    fraud = credit_card_data[credit_card_data.Class == 1]

    print(legit.shape)
    print(fraud.shape)

    # Statistical measures of the data
    print(legit.Amount.describe())
    print(fraud.Amount.describe())

    # Compare the values for both transactions
    print(credit_card_data.groupby('Class').mean())

    # Adjust sample size if necessary
    sample_size = min(2, len(legit))

    legit_sample = legit.sample(n=sample_size)
    new_dataset = pd.concat([legit_sample, fraud], axis=0)

    print(new_dataset['Class'].value_counts())
    print(new_dataset.groupby('Class').mean())

    # Split the data into features and target
    X = new_dataset.drop(columns='Class', axis=1)
    Y = new_dataset['Class']

    # Ensure there are no missing values or inconsistent data types
    X = X.apply(pd.to_numeric, errors='coerce')
    Y = Y.apply(pd.to_numeric, errors='coerce')
    X = X.dropna()
    Y = Y[X.index]

    # Split the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42)

    # Initialize the model
    model = LogisticRegression()

    # Train the model
    model.fit(X_train, Y_train)

    # Predict on the test data
    Y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(Y_test, Y_pred)
    print(f'Accuracy: {accuracy}')
else:
    print("Kolom 'Class' tidak ditemukan dalam dataset. Pastikan dataset memiliki kolom 'Class'.")
