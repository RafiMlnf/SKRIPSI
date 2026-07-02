# Analisis Sentimen "Arteri" - .FEAST
## 🎵 Menggunakan Metode SVM (Support Vector Machine)

Proyek analisis sentimen komentar YouTube pada lagu "Arteri" oleh .FEAST untuk keperluan Seminar Proposal.

**Author**: Rafi Maulana Firdaus (312210382)  
**Universitas**: Pelita Bangsa

---

## 📁 Struktur Folder

```
├── Data/              # Dataset (raw, preprocessed, labeled)
├── Scripts/           # Script Python untuk processing
├── Notebooks/         # Jupyter Notebook untuk analisis ⭐
├── Output/            # Output model dan visualisasi
├── DOKUMENTASI.md     # Dokumentasi lengkap
└── requirements.txt   # Dependencies
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
pip install Sastrawi scikit-learn jupyter matplotlib seaborn
```

### 2. Jalankan Notebook
```bash
jupyter notebook
# Buka: Notebooks/Analisis_Sentimen_SVM.ipynb
```

### 3. Atau Jalankan Script
```bash
# Preprocessing
py -3 Scripts/preprocessing.py

# Labeling
py -3 Scripts/labeling.py

# Training SVM
py -3 Scripts/svm_model.py
```

## 🔧 Metodologi

1. **Pengumpulan Data** - Scraping komentar YouTube
2. **Preprocessing** - Pembersihan dan normalisasi teks
3. **Labeling** - Pelabelan sentimen (positif/negatif/netral)
4. **TF-IDF** - Ekstraksi fitur dengan TF-IDF Vectorizer
5. **SVM** - Klasifikasi dengan Support Vector Machine
6. **Evaluasi** - Akurasi, Presisi, Recall, F1-Score

## 📊 Dataset

| Tahap | Jumlah |
|-------|--------|
| Raw Data | 1146 komentar |
| Setelah Preprocessing | ~821 komentar |

---

Lihat `DOKUMENTASI.md` untuk detail lengkap.
