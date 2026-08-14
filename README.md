# AI Prediksi Kelulusan

Proyek PAS mata kuliah **AI Computing Platform** — Sistem Informasi, STIKOM Cipta Karya Informatika.

## Identitas

- **Nama** : Sarmila
- **NIM** : 24120120009
- **Jurusan** : Sistem Informasi

## Apa Ini?

Program AI yang memprediksi apakah seorang mahasiswa akan **LULUS**
atau **TIDAK LULUS** berdasarkan nilai tugas, UTS, UAS, dan kehadiran.

## Kenapa Teknik Beda dari Proyek Teks?

Proyek ini memakai **Logistic Regression**, bukan embedding + cosine
similarity seperti proyek teks (spam, sentimen, ujaran kasar). Alasannya:

> **Datanya berupa ANGKA, bukan teks.**

Embedding dirancang untuk menangkap makna **kalimat**. Kalau dipaksakan
ke data numerik seperti nilai dan kehadiran, itu justru salah metode.
Untuk data angka terstruktur, Logistic Regression adalah pilihan yang
tepat, sederhana, dan mudah dijelaskan.

## Cara Kerja (Singkat)

1. Model mempelajari pola dari data: kombinasi nilai & kehadiran
   seperti apa yang cenderung lulus atau tidak.
2. Untuk prediksi, model menghitung **peluang** mahasiswa lulus
   berdasarkan angka-angka yang dimasukkan.
3. Peluang di atas 50% → diprediksi LULUS.

## Struktur File

```
AI_Prediksi_Kelulusan_Sarmila/
├── data/dataset.csv    # data nilai mahasiswa berlabel
├── train.py            # tahap belajar (Logistic Regression)
├── predict.py          # tahap memprediksi
├── requirements.txt    # daftar pustaka
└── README.md           # file ini
```

## Cara Menjalankan

```bash
pip install -r requirements.txt
python train.py
python predict.py
```

> Proyek ini TIDAK mengunduh model besar, jadi langsung jalan cepat.

## Catatan Akurasi

Pada dataset ini akurasi bisa mencapai 100% karena datanya terpisah
dengan jelas (nilai bagus konsisten lulus, nilai rendah tidak). Pada
data dunia nyata yang lebih bervariasi, peluang kelulusan tidak akan
sebulat 0% atau 100%.
