# 📖 Dokumentasi Proyek Analisis Sentimen
## Lagu "Arteri" - .FEAST menggunakan Metode SVM (Support Vector Machine)

---

**Tanggal Mulai**: 18 Januari 2026  
**Update Terakhir**: 1 Februari 2026  
**Keperluan**: Seminar Proposal (SEMPRO)  
**Topik**: Machine Learning / Natural Language Processing  
**Author**: Rafi Maulana Firdaus (312210382)

---

## 📋 Daftar Isi

1. [Pendahuluan](#1-pendahuluan)
2. [Struktur Folder Proyek](#2-struktur-folder-proyek)
3. [Tahap 1: Pengumpulan Data (Scraping)](#3-tahap-1-pengumpulan-data-scraping)
4. [Tahap 2: Preprocessing Data](#4-tahap-2-preprocessing-data)
5. [Tahap 3: Labeling Sentimen](#5-tahap-3-labeling-sentimen)
6. [Tahap 4: Ekstraksi Fitur TF-IDF](#6-tahap-4-ekstraksi-fitur-tf-idf)
7. [Tahap 5: Pembuatan Model SVM](#7-tahap-5-pembuatan-model-svm)
8. [Cara Menjalankan](#8-cara-menjalankan)
9. [Hasil dan Kesimpulan](#9-hasil-dan-kesimpulan)

---

## 1. Pendahuluan

### 1.1 Latar Belakang
Proyek ini bertujuan untuk melakukan **analisis sentimen** terhadap komentar YouTube pada video lagu **"Arteri"** oleh band **.FEAST** menggunakan metode **Support Vector Machine (SVM)** dengan ekstraksi fitur **TF-IDF (Term Frequency-Inverse Document Frequency)**.

### 1.2 Objek Penelitian
- **Lagu**: Arteri
- **Artis**: .FEAST
- **Sumber Data**: Komentar YouTube
- **Video yang Dianalisis**:
  - Official Music Video: https://www.youtube.com/watch?v=eulYTPXOaio

### 1.3 Profil Band .FEAST
**.FEAST** adalah band rock Indonesia yang terbentuk di Jakarta. Personil utama:
- **Baskara Putra** (Bas) - Vokalis

> **Catatan Penting**: Dalam komentar YouTube, kata "Bas" merujuk kepada **Baskara Putra** (vokalis), bukan alat musik bass.

### 1.4 Metodologi
1. **Pengumpulan Data** - Scraping komentar dari YouTube
2. **Preprocessing** - Pembersihan dan normalisasi teks
3. **Labeling** - Pelabelan sentimen (positif/negatif/netral)
4. **Ekstraksi Fitur** - TF-IDF Vectorization
5. **Modeling** - Pembuatan model SVM untuk klasifikasi sentimen
6. **Evaluasi** - Pengujian dan evaluasi performa model

### 1.5 Mengapa SVM?
SVM dipilih sebagai metode utama karena:
- **Robust** untuk menangani data teks berukuran besar dengan dimensi fitur tinggi
- **Efektif** untuk dataset berskala terbatas (~1000 sampel)
- **Mampu** menemukan hyperplane pemisah terbaik untuk klasifikasi multi-kelas
- **Tidak membutuhkan** sumber daya komputasi besar seperti model Deep Learning

---

## 2. Struktur Folder Proyek

```
📁 e:\Kuliah\01. SEMPRO\SOURCE CODE\
│
├── 📁 Data/                              # Dataset
│   ├── comments_arteri_full.csv          # Komentar raw (1146)
│   ├── comments_preprocessed.csv         # Setelah preprocessing
│   ├── comments_labeled.csv              # Dengan label sentimen
│   ├── preprocessing_report.txt          # Laporan preprocessing
│   ├── labeling_report.txt               # Laporan labeling
│   └── sentiment_distribution.png        # Visualisasi distribusi
│
├── 📁 Scripts/                           # Script Python
│   ├── youtube_scraper.py                # Script utama scraping
│   ├── run_scraper_simple.py             # Runner scraping
│   ├── scrape_1000.py                    # Script untuk 1000+ komentar
│   ├── preprocessing.py                  # Script preprocessing
│   ├── labeling.py                       # Script labeling sentimen
│   └── svm_model.py                      # ⭐ Script model SVM
│
├── 📁 Notebooks/                         # Jupyter Notebooks
│   └── Analisis_Sentimen_SVM.ipynb       # ⭐ NOTEBOOK UTAMA
│
├── 📁 Output/                            # Output hasil analisis
│   ├── svm_sentiment_model.pkl           # Model SVM terlatih
│   ├── tfidf_vectorizer.pkl              # TF-IDF Vectorizer
│   ├── confusion_matrix.png              # Confusion matrix
│   ├── metrics_comparison.png            # Grafik metrik evaluasi
│   └── model_report.txt                  # Laporan model
│
├── 📄 DOKUMENTASI.md                     # Dokumentasi lengkap (file ini)
├── 📄 README.md                          # Panduan singkat
└── 📄 requirements.txt                   # Dependencies
```

---

## 3. Tahap 1: Pengumpulan Data (Scraping)

### 3.1 Deskripsi
Mengumpulkan komentar dari video YouTube lagu "Arteri" menggunakan library `youtube-comment-downloader`.

### 3.2 Tools yang Digunakan
- Python 3.12
- Library: `youtube-comment-downloader`, `pandas`, `tqdm`

### 3.3 Script: `Scripts/scrape_1000.py`

### 3.4 Hasil Scraping

| Video | URL | Jumlah Komentar |
|-------|-----|-----------------|
| Official MV | youtube.com/watch?v=eulYTPXOaio | **1146** |

### 3.5 Format Data

```csv
no,author,text,time,likes,replies,is_reply
1,@username,komentar di sini,22 jam yang lalu,5,2,False
```

### 3.6 File Output
- `Data/comments_arteri_full.csv` - Data mentah hasil scraping

---

## 4. Tahap 2: Preprocessing Data

### 4.1 Deskripsi
Membersihkan dan menormalisasi data komentar agar siap untuk analisis sentimen.

### 4.2 Script: `Scripts/preprocessing.py`

### 4.3 Tahapan Preprocessing

| No | Tahapan | Deskripsi |
|----|---------|-----------|
| 1 | Case Folding | Konversi semua huruf ke lowercase |
| 2 | Data Cleaning | Hapus URL, mention (@), hashtag (#), timestamp |
| 3 | Emoji Removal | Hapus semua emoji dan emoticon |
| 4 | Spam Detection | Deteksi dan hapus komentar spam |
| 5 | Lyrics Detection | Deteksi dan hapus komentar berisi lirik lagu |
| 6 | Punctuation Normalization | Normalisasi tanda baca |
| 7 | Slang Normalization | Konversi kata slang ke bentuk baku (500+ kata) |
| 8 | Stopwords Removal | Hapus stopwords Bahasa Indonesia (200+ kata) |
| 9 | Stemming | Stemming menggunakan Sastrawi |
| 10 | Quality Filtering | Filter berdasarkan panjang teks (min 2, max 100 kata) |
| 11 | Duplicate Removal | Hapus komentar duplikat |

### 4.4 Hasil Preprocessing

| Keterangan | Jumlah |
|------------|--------|
| Total komentar awal | **1146** |
| Setelah preprocessing | ~**821** |
| **Persentase tersisa** | **~71.6%** |

### 4.5 File Output
- `Data/comments_preprocessed.csv`
- `Data/preprocessing_report.txt`

---

## 5. Tahap 3: Labeling Sentimen

### 5.1 Deskripsi
Melabeli setiap komentar dengan kategori sentimen menggunakan metode **Lexicon-Based**.

### 5.2 Script: `Scripts/labeling.py`

### 5.3 Metode Labeling
- Lexicon-Based Sentiment Analysis
- Kamus Sentimen Bahasa Indonesia (custom)
- Negation Handling
- Intensifier Recognition

### 5.4 Threshold Klasifikasi
- **Positif**: Score ≥ 1 (Label: 2)
- **Negatif**: Score ≤ -1 (Label: 0)
- **Netral**: -1 < Score < 1 (Label: 1)

### 5.5 File Output
- `Data/comments_labeled.csv`
- `Data/labeling_report.txt`
- `Data/sentiment_distribution.png`

---

## 6. Tahap 4: Ekstraksi Fitur TF-IDF

### 6.1 Deskripsi
**TF-IDF (Term Frequency-Inverse Document Frequency)** digunakan untuk mengkonversi teks menjadi vektor numerik yang dapat diproses oleh algoritma SVM.

### 6.2 Rumus TF-IDF

```
TF-IDF(t,d) = TF(t,d) × IDF(t)

dimana:
- TF(t,d) = Frekuensi term t dalam dokumen d
- IDF(t) = log(N / df(t))
- N = Total jumlah dokumen
- df(t) = Jumlah dokumen yang mengandung term t
```

### 6.3 Konfigurasi TF-IDF

| Parameter | Nilai | Deskripsi |
|-----------|-------|-----------|
| max_features | 5000 | Vocabulary size maksimal |
| ngram_range | (1, 2) | Unigram + Bigram |
| min_df | 2 | Minimum document frequency |
| max_df | 0.95 | Maximum document frequency |
| sublinear_tf | True | Menggunakan log scaling |

### 6.4 Keuntungan TF-IDF
- Menangkap pentingnya kata dalam konteks dokumen
- Mengurangi bobot kata yang terlalu umum
- Cocok untuk klasifikasi teks dengan SVM

---

## 7. Tahap 5: Pembuatan Model SVM

### 7.1 Deskripsi
Membangun dan melatih model **Support Vector Machine (SVM)** untuk klasifikasi sentimen.

### 7.2 Notebook: `Notebooks/Analisis_Sentimen_SVM.ipynb` ⭐

**Ini adalah file utama untuk menjalankan analisis lengkap!**

### 7.3 Konsep SVM

SVM bekerja dengan prinsip mencari **hyperplane pemisah terbaik** (best separating hyperplane) yang memaksimalkan margin antara kelas-kelas sentimen.

```
┌─────────────────────────────────────┐
│     TF-IDF Vectorization            │
│     (5000 features max)             │
└──────────────┬──────────────────────┘
               ↓
┌─────────────────────────────────────┐
│     Support Vector Machine          │
│     (Linear/RBF Kernel)             │
└──────────────┬──────────────────────┘
               ↓
┌─────────────────────────────────────┐
│     GridSearchCV                    │
│     (Hyperparameter Tuning)         │
└──────────────┬──────────────────────┘
               ↓
┌─────────────────────────────────────┐
│     Output: 3 Classes               │
│     [negatif, netral, positif]      │
└─────────────────────────────────────┘
```

### 7.4 Konfigurasi Model SVM

| Parameter | Nilai | Deskripsi |
|-----------|-------|-----------|
| Kernel | linear, rbf | Kernel function |
| C | [0.1, 1, 10] | Regularization parameter |
| Gamma | scale, auto | Kernel coefficient |
| CV Folds | 5 | Cross-validation folds |

### 7.5 Pembagian Data

| Set | Persentase | Keterangan |
|-----|------------|------------|
| Training | 80% | Untuk melatih model |
| Test | 20% | Untuk evaluasi akhir |

### 7.6 Metrik Evaluasi

| Metrik | Deskripsi |
|--------|-----------|
| **Accuracy** | Proporsi prediksi benar dari total prediksi |
| **Precision** | Proporsi prediksi positif yang benar |
| **Recall** | Proporsi sampel positif yang berhasil diidentifikasi |
| **F1-Score** | Harmonic mean dari Precision dan Recall |

---

## 8. Cara Menjalankan

### 8.1 Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt
pip install Sastrawi
pip install scikit-learn
pip install jupyter
```

### 8.2 Opsi 1: Menggunakan Jupyter Notebook (Recommended) ⭐

```bash
# Buka Jupyter Notebook
jupyter notebook

# Buka file: Notebooks/Analisis_Sentimen_SVM.ipynb
# Jalankan semua cell (Shift+Enter atau Run All)
```

### 8.3 Opsi 2: Menggunakan Script Python

```bash
cd "e:\Kuliah\01. SEMPRO\SOURCE CODE"

# 1. Scraping (jika belum ada data)
py -3 Scripts/scrape_1000.py

# 2. Preprocessing
py -3 Scripts/preprocessing.py

# 3. Labeling
py -3 Scripts/labeling.py

# 4. Training SVM
py -3 Scripts/svm_model.py
```

---

## 9. Hasil dan Kesimpulan

### 9.1 Ringkasan Data

| Tahap | Jumlah Data |
|-------|-------------|
| Scraping | 1146 komentar |
| Setelah Preprocessing | ~821 komentar |
| Main Comments | 831 |
| Replies | 315 |

### 9.2 Keunggulan Metode SVM + TF-IDF

1. **Efisien** - Tidak membutuhkan GPU seperti Deep Learning
2. **Robust** - Performa baik pada dataset kecil-menengah
3. **Interpretable** - Fitur TF-IDF dapat dianalisis
4. **Cepat** - Training hanya membutuhkan beberapa detik

### 9.3 Temuan Utama

1. **Mayoritas komentar bersifat positif atau netral**
2. **Banyak referensi ke MCR** (My Chemical Romance), menunjukkan pengaruh musik yang terasa
3. **Tema utama komentar**: apresiasi lagu, relate dengan lirik, perbandingan dengan artis lain

---

## 📝 Changelog

| Tanggal | Waktu | Kegiatan |
|---------|-------|----------|
| 18/01/2026 | 21:00 | Pembuatan script scraping YouTube |
| 18/01/2026 | 21:05 | Scraping 697 komentar awal |
| 18/01/2026 | 21:20 | Pembuatan script preprocessing |
| 18/01/2026 | 21:28 | Preprocessing data awal |
| 18/01/2026 | 21:33 | Labeling sentimen awal |
| 18/01/2026 | 21:40 | Reorganisasi folder |
| 18/01/2026 | 22:00 | Scraping ulang: 1146 komentar |
| 18/01/2026 | 22:03 | Preprocessing ulang |
| 18/01/2026 | 22:37 | Labeling ulang dengan data baru |
| 01/02/2026 | 08:30 | **Migrasi dari LSTM ke SVM** |

---

## 👤 Author

**Rafi Maulana Firdaus**  
NIM: 312210382  
Universitas Pelita Bangsa  
Februari 2026

---

*Dokumentasi ini diperbarui pada: 1 Februari 2026*
