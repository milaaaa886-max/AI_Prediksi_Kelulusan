"""
predict.py - Tahap MENEBAK (AI Prediksi Kelulusan)
===================================================
Teknik: Logistic Regression (scikit-learn).

Model yang sudah dilatih dipakai untuk memprediksi: dengan nilai &
kehadiran tertentu, seorang mahasiswa diprediksi LULUS atau TIDAK.
Program juga menampilkan peluang (persentase) kelulusannya.

Jalankan:  python predict.py
"""

import pickle

try:
    from sklearn.linear_model import LogisticRegression  # noqa: F401
except ImportError:
    print("Pustaka belum lengkap. Jalankan dulu:")
    print("   pip install -r requirements.txt")
    raise SystemExit(1)

MODEL_FILE = "model.pkl"


def muat_model():
    try:
        with open(MODEL_FILE, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        print("Model belum ada. Jalankan dulu:")
        print("   python train.py")
        raise SystemExit(1)


def prediksi(model, nilai_tugas, nilai_uts, nilai_uas, kehadiran):
    fitur = [[nilai_tugas, nilai_uts, nilai_uas, kehadiran]]
    hasil = model.predict(fitur)[0]
    peluang = model.predict_proba(fitur)[0][1]  # peluang kelas "lulus"
    label = "LULUS" if hasil == 1 else "TIDAK LULUS"
    return label, peluang


def main():
    model = muat_model()

    # Data mahasiswa yang mau diprediksi (tugas, uts, uas, kehadiran).
    # Silakan ganti angkanya sesukamu.
    contoh_uji = [
        (85, 82, 88, 95),
        (45, 40, 48, 62),
        (68, 65, 70, 83),
        (35, 38, 34, 50),
    ]

    print("\n===== HASIL PREDIKSI KELULUSAN =====")
    for (tg, uts, uas, hdr) in contoh_uji:
        label, peluang = prediksi(model, tg, uts, uas, hdr)
        print(f"\nNilai -> Tugas:{tg} UTS:{uts} UAS:{uas} Kehadiran:{hdr}")
        print(f"Prediksi : {label}  (peluang lulus {peluang*100:.1f}%)")


if __name__ == "__main__":
    main()
