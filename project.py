import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE

# Membaca dataset
credit_card_data = pd.read_csv('clean_dataset.csv')

# Memeriksa distribusi kelas
print("Distribusi kelas sebelum SMOTE:")
print(credit_card_data['Class'].value_counts())

# Memisahkan fitur (X) dan target (Y)
X = credit_card_data.drop(columns='Class', axis=1)
Y = credit_card_data['Class']

# Menggunakan SMOTE untuk oversampling
smote = SMOTE(random_state=42)
X_resampled, Y_resampled = smote.fit_resample(X, Y)

# Memeriksa distribusi kelas setelah SMOTE
print("Distribusi kelas setelah SMOTE:")
print(Y_resampled.value_counts())

# Membagi data menjadi data latih dan data uji dengan stratify setelah resampling
X_train, X_test, Y_train, Y_test = train_test_split(X_resampled, Y_resampled, test_size=0.2, stratify=Y_resampled, random_state=42)

# Inisialisasi model
model = LogisticRegression(max_iter=1000)

# Melatih model
model.fit(X_train, Y_train)

# Melakukan prediksi pada data uji
Y_pred = model.predict(X_test)

# Menghitung akurasi
accuracy = accuracy_score(Y_test, Y_pred)
print(f"Akurasi: {accuracy}")

# Menampilkan beberapa prediksi contoh
print("Prediksi contoh:")
for i in range(5):
    print(f"Prediksi: {Y_pred[i]}, Aktual: {Y_test.values[i]}")
