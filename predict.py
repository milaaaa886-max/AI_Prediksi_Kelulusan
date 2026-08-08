"""
predict.py - Coba model AI Prediksi Kelulusan yang sudah dilatih.
Cara pakai:
    python predict.py
"""

import joblib
import os
import pandas as pd

MODEL_PATH = "models/kelulusan_model.pkl"

if not os.path.exists(MODEL_PATH):
    print("Model belum ditemukan. Jalankan 'python train.py' terlebih dahulu.")
    exit()

model = joblib.load(MODEL_PATH)


def predict_kelulusan(nilai_tugas, nilai_uts, nilai_uas, kehadiran):
    data = pd.DataFrame([{
        "nilai_tugas": nilai_tugas,
        "nilai_uts": nilai_uts,
        "nilai_uas": nilai_uas,
        "kehadiran": kehadiran,
    }])
    return model.predict(data)[0]


if __name__ == "__main__":
    print("=== AI Prediksi Kelulusan Mahasiswa ===")
    print("Masukkan nilai (0-100). Ketik 'exit' pada nilai tugas untuk keluar.\n")

    while True:
        inp = input("Nilai tugas: ")
        if inp.lower() == "exit":
            print("Sampai jumpa!")
            break
        tugas = float(inp)
        uts = float(input("Nilai UTS: "))
        uas = float(input("Nilai UAS: "))
        hadir = float(input("Persentase kehadiran: "))

        hasil = predict_kelulusan(tugas, uts, uas, hadir)
        print(f">> Prediksi: {hasil.upper()}\n")
