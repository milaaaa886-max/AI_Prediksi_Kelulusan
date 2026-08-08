# AI Prediksi Kelulusan Mahasiswa

Program AI sederhana untuk memprediksi apakah seorang mahasiswa **LULUS** atau **TIDAK LULUS** berdasarkan nilai tugas, UTS, UAS, dan persentase kehadiran, menggunakan metode *Machine Learning*: **Logistic Regression**.

## Identitas

- Nama   : Sarmila
- NIM    : 24120120009
- Jurusan: Sistem Informasi
- Semester: 4

## Deskripsi Project

Model AI ini dilatih menggunakan data nilai mahasiswa (tugas, UTS, UAS, kehadiran) beserta status kelulusannya. Model mempelajari pola hubungan antara nilai-nilai tersebut dengan hasil akhir (lulus/tidak lulus), lalu digunakan untuk memprediksi status mahasiswa baru berdasarkan nilainya.

## Struktur Folder

```
proj_sarmila/
├── data/
│   └── dataset.csv
├── models/            # otomatis dibuat setelah training
├── train.py
├── predict.py
├── requirements.txt
└── README.md
```

## Cara Menjalankan

```bash
pip install -r requirements.txt
python train.py
python predict.py
```

## Contoh Penggunaan

```
Nilai tugas: 85
Nilai UTS: 80
Nilai UAS: 88
Persentase kehadiran: 95
>> Prediksi: LULUS
```

## Teknologi yang Digunakan

- Python 3
- Pandas
- Scikit-learn (Logistic Regression)
- Joblib
