"""
train.py - Tahap BELAJAR (AI Prediksi Kelulusan)
=================================================
Teknik: Logistic Regression (scikit-learn).

CATATAN PENTING soal pemilihan metode:
Data proyek ini berupa ANGKA (nilai tugas, UTS, UAS, kehadiran),
BUKAN teks. Karena itu kita TIDAK memakai embedding + cosine
similarity seperti proyek teks — memaksa embedding ke data angka
adalah metode yang salah. Untuk data numerik terstruktur seperti
ini, Logistic Regression adalah pilihan yang tepat dan umum dipakai.

Program mempelajari pola: kombinasi nilai & kehadiran seperti apa
yang cenderung LULUS (1) atau TIDAK (0).

Jalankan:  python train.py
"""

import csv
import pickle

try:
    from sklearn.linear_model import LogisticRegression
except ImportError:
    print("Pustaka belum lengkap. Jalankan dulu:")
    print("   pip install -r requirements.txt")
    raise SystemExit(1)

DATA_PATH = "data/dataset.csv"
OUTPUT = "model.pkl"


def baca_dataset(path):
    """Baca CSV. X = fitur angka, y = label lulus/tidak."""
    X, y = [], []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for baris in reader:
            X.append([
                float(baris["nilai_tugas"]),
                float(baris["nilai_uts"]),
                float(baris["nilai_uas"]),
                float(baris["kehadiran"]),
            ])
            y.append(int(baris["lulus"]))
    return X, y


def main():
    print("Membaca dataset...")
    X, y = baca_dataset(DATA_PATH)
    print(f"   {len(X)} data mahasiswa dimuat.")

    print("Melatih model Logistic Regression...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    akurasi = model.score(X, y)
    print(f"   Akurasi pada data latih: {akurasi*100:.1f}%")

    with open(OUTPUT, "wb") as f:
        pickle.dump(model, f)

    print(f"\nSelesai! Model disimpan di '{OUTPUT}'.")
    print("Lanjut jalankan:  python predict.py")


if __name__ == "__main__":
    main()
