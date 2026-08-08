"""
train.py - AI Prediksi Kelulusan Mahasiswa
Melatih model AI untuk memprediksi apakah mahasiswa LULUS atau TIDAK LULUS
berdasarkan nilai tugas, UTS, UAS, dan persentase kehadiran.

Cara pakai:
    python train.py
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

DATA_PATH = "data/dataset.csv"
df = pd.read_csv(DATA_PATH)

print(f"Total data: {len(df)}")
print(df["status"].value_counts())

# Fitur (input) dan label (output)
X = df[["nilai_tugas", "nilai_uts", "nilai_uas", "kehadiran"]]
y = df["status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nAkurasi model: {acc * 100:.2f}%")
print("\nLaporan klasifikasi:")
print(classification_report(y_test, y_pred))

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/kelulusan_model.pkl")

print("\nModel berhasil disimpan di folder 'models/'")
