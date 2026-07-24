# PRESENTASI SKRIPSI
## Analisis Sentimen Pendengar Lagu "Arteri" oleh .Feast  
## Menggunakan Algoritma Support Vector Machine (SVM)

**Rafi Maulana Firdaus — 312210382**  
Program Studi Teknik Informatika, Universitas Pelita Bangsa

---

## SLIDE 1 — JUDUL & IDENTITAS

> **Judul Penelitian**
>
> **"Analisis Sentimen Pendengar Lagu 'Arteri' oleh .Feast  
> Menggunakan Algoritma Support Vector Machine (SVM)"**

| | |
|---|---|
| **Nama** | Rafi Maulana Firdaus |
| **NIM** | 312210382 |
| **Program Studi** | Teknik Informatika |
| **Universitas** | Universitas Pelita Bangsa |
| **Pembimbing I** | Abdul Halim Anshor, S.Kom., M.Kom. |
| **Pembimbing II** | Karina Imelda, S.Kom., M.Kom. |

---

## SLIDE 2 — LATAR BELAKANG

### Mengapa Penelitian Ini Penting?

**Fenomena yang terjadi:**
- Lagu **"Arteri"** oleh **.Feast** viral di YouTube → ribuan komentar publik tentang *quarter-life crisis*
- Komentar bersifat tidak terstruktur: bahasa gaul, slang, sarkasme, singkatan

**Masalah utama:**
- ❌ Analisis manual → tidak efisien, rentan bias
- ❌ Tools konvensional → gagal memproses bahasa Indonesia informal
- ✅ Solusi: **otomatisasi klasifikasi sentimen** dengan Machine Learning

**Pertanyaan Penelitian:**
> Bagaimana SVM + TF-IDF dapat mengklasifikasikan sentimen komentar YouTube secara akurat?

---

## SLIDE 3 — RUMUSAN & TUJUAN MASALAH

### 3 Rumusan Masalah Utama:

| # | Rumusan Masalah |
|---|---|
| **1** | Bagaimana penerapan SVM + TF-IDF untuk klasifikasi sentimen komentar YouTube? |
| **2** | Seberapa baik kinerja model SVM diukur dengan Confusion Matrix (Akurasi, Presisi, Recall, F1-Score)? |
| **3** | Bagaimana kecenderungan dominasi sentimen pendengar (Positif/Negatif/Netral)? |

### Tujuan:
1. Implementasi pipeline klasifikasi sentimen otomatis berbasis SVM
2. Evaluasi performa model pada teks informal
3. Pemetaan tren persepsi pendengar terhadap lagu "Arteri"

---

## SLIDE 4 — LANDASAN TEORI (RINGKAS)

### Teknologi Inti yang Digunakan:

```
KOMENTAR YOUTUBE (Teks Mentah)
         ↓
   [NLP Preprocessing]
         ↓
   [TF-IDF Vectorizer]
         ↓
  [SVM Classifier]
         ↓
  Positif / Netral / Negatif
```

**Support Vector Machine (SVM)**
- Mencari **hyperplane optimal** pemisah kelas sentimen
- Persamaan: `w · x + b = 0`
- Unggul pada data berdimensi tinggi dan dataset terbatas
- Kernel **Linear** terbukti paling efektif untuk klasifikasi teks

**TF-IDF (Term Frequency – Inverse Document Frequency)**
- Mengubah teks menjadi **vektor numerik**
- Memberi bobot tinggi pada kata spesifik yang bermakna
- Menekan bobot kata umum yang tidak informatif

---

## SLIDE 5 — METODOLOGI & LANGKAH-LANGKAH PENELITIAN

### Kerangka KDD (*Knowledge Discovery in Databases*) — 6 Tahap:

```
TAHAP 1: Pengumpulan Data
    └─ Web scraping komentar YouTube resmi .Feast - Arteri
    └─ Hasil: 1.531 komentar mentah → disimpan .CSV

TAHAP 2: Preprocessing Teks (7 Sub-tahap)
    ├─ Cleaning       → hapus URL, emoji, simbol, angka
    ├─ Case Folding   → ubah ke huruf kecil (lowercase)
    ├─ Tokenizing     → pecah kalimat menjadi token kata
    ├─ Normalisasi    → koreksi kata slang ("gak" → "tidak")
    ├─ Stopword Removal → hapus kata tidak bermakna
    ├─ Stemming       → kembalikan ke kata dasar (Sastrawi)
    └─ Pelabelan Leksikon → label otomatis (Positif/Netral/Negatif)

TAHAP 3: Ekstraksi Fitur
    └─ TF-IDF Vectorization → matriks fitur 1.175 × 745 dimensi

TAHAP 4: Pembagian Data
    └─ Training Set: 80% (940 data)
    └─ Testing Set:  20% (235 data)

TAHAP 5: Pelatihan Model SVM
    └─ GridSearchCV (5-fold CV) → cari hyperparameter terbaik
    └─ Random Oversampling → atasi ketidakseimbangan kelas Negatif
    └─ Kernel Linear, C = 1.0, Strategi OvO (One-vs-One)

TAHAP 6: Evaluasi Model
    └─ Confusion Matrix → Akurasi, Presisi, Recall, F1-Score
```

---

## SLIDE 6 — DATA & PREPROCESSING

### Statistik Dataset:

| Keterangan | Nilai |
|---|---|
| Komentar mentah awal | **1.531** |
| Komentar valid (bersih) | **1.175** |
| Komentar dihapus | 347 (22,7%) |

### Distribusi Kelas Sentimen (Setelah Pelabelan):

| Kelas | Jumlah | Persentase |
|---|:---:|:---:|
| **Netral** | 754 | **64,2%** |
| **Positif** | 308 | **26,2%** |
| **Negatif** | 113 | **9,6%** |
| **Total** | **1.175** | **100%** |

![Distribusi Sentimen](SOURCE%20CODE/Data/sentiment_distribution.png)


### Contoh Hasil Preprocessing:

| Teks Mentah | Setelah Preprocessing | Label |
|---|---|:---:|
| *"dulu sering denger sampai suka ini lagu..."* | `dengar suka lagu makna nyaman` | **Positif** |
| *"aduh kaya cerita saat ini, menyakitkan sialannnn"* | `kayak cerita sakit sialannnn` | **Negatif** |
| *"i'm not okay MCR? wkwkwkkw sama persis..."* | `not okay mcr sama persis awal` | **Netral** |

---

## SLIDE 7 — HASIL PELATIHAN MODEL SVM

### Hyperparameter Terbaik (GridSearchCV, 5-Fold CV):

| Parameter | Nilai |
|---|---|
| **Kernel** | Linear |
| **C (Regularisasi)** | 1.0 |
| **Strategi Multi-kelas** | One-vs-One (OvO) |
| **Best CV Score (F1 Weighted)** | ~80% |

### Matriks Fitur TF-IDF:

| Bagian | Sampel | Dimensi Fitur |
|---|:---:|:---:|
| Training Set | 940 | 745 |
| Testing Set | 235 | 745 |

### Penanganan Imbalance Class (Kelas Negatif Minoritas):

- **Masalah:** Negatif hanya 9,6% → model bias ke kelas mayoritas
- **Solusi:** Random Oversampling pada kelas Negatif
- **Hasil:** Recall Negatif naik dari **~31% → 65%** tanpa mengorbankan akurasi global

---

## SLIDE 8 — HASIL EVALUASI & PERFORMA MODEL

### Metrik Evaluasi Utama (235 Data Uji):

| Metrik | Nilai |
|---|:---:|
| **Akurasi** | **80,85%** |
| **Presisi (Weighted)** | **80,55%** |
| **Recall (Weighted)** | **80,85%** |
| **F1-Score (Weighted)** | **80,66%** |

> ✅ Model mengklasifikasikan **190 dari 235** komentar dengan benar.

### Performa Per Kelas:

| Kelas | Precision | Recall | F1-Score | Support |
|---|:---:|:---:|:---:|:---:|
| **Netral** | 0,86 | 0,89 | **0,87** | 151 |
| **Positif** | 0,74 | 0,69 | **0,71** | 61 |
| **Negatif** | 0,64 | 0,61 | **0,62** | 23 |

### Confusion Matrix:

| Aktual \ Prediksi | Negatif | Netral | Positif |
|---|:---:|:---:|:---:|
| **Negatif** | **14** | 6 | 3 |
| **Netral** | 5 | **134** | 12 |
| **Positif** | 3 | 16 | **42** |

![Confusion Matrix](SOURCE%20CODE/Output/confusion_matrix.png)

> ⚠️ Kesalahan terbesar: Positif → diprediksi Netral (16 kasus)  
> Penyebab: batas semantik antara apresiasi dan komentar informatif sangat tipis.


### Perbandingan SVM vs Logistic Regression:

| Algoritma | Akurasi | F1-Score |
|---|:---:|:---:|
| **SVM Linear** ✅ | **80,85%** | **80,66%** |
| Logistic Regression | 79,15% | 79,14% |

> SVM unggul **+1,70% akurasi** — prinsip *Maximal Margin Classifier* lebih tahan terhadap noise teks informal dibanding sigmoid Logistic Regression.

---

## SLIDE 9 — ANALISIS TREN SENTIMEN PENDENGAR

### Distribusi Sentimen Final:

```
Netral   ████████████████████████████████ 64,2% (754 komentar)
Positif  █████████████ 26,2% (308 komentar)
Negatif  █████  9,6%  (113 komentar)
```

![Distribusi Sentimen](SOURCE%20CODE/Data/sentiment_distribution.png)

### Mengapa Netral Mendominasi (64,2%)?
Bukan ketidakpedulian, melainkan karakteristik komunikasi pendengar:
1. **Narasi Pengalaman Pribadi** → cerita quarter-life crisis tanpa penilaian eksplisit
2. **Diskusi Komparatif** → membandingkan atmosfer lagu dengan MCR
3. **Kutipan & Interpretasi Lirik** → membahas makna tanpa kata sifat evaluatif
4. **Penyebutan Nama Vokalis "Baskara"** → nama entitas = skor leksikon 0

### Kata Kunci Dominan per Sentimen:

| Sentimen Positif | Sentimen Netral | Sentimen Negatif |
|:---:|:---:|:---:|
| relate, bagus, keren | feast, mcr, helena | sakit, bosan, jelek |
| baskara, suka, indah | cerita, hidup, lirik | tidak suka, kepharat |
| *Apresiasi musikal & resonansi emosional* | *Diskusi & narasi pribadi* | *Kritik selera & luapan emosi* |

### Word Cloud Kata Kunci Dominan (Output Streamlit):

| Sentimen Positif | Sentimen Netral | Sentimen Negatif |
|:---:|:---:|:---:|
| ![Word Cloud Positif](SOURCE%20CODE/Output/wordcloud_positif.png) | ![Word Cloud Netral](SOURCE%20CODE/Output/wordcloud_netral.png) | ![Word Cloud Negatif](SOURCE%20CODE/Output/wordcloud_negatif.png) |


### Kesimpulan Tren:
> 🎵 Lagu **"Arteri"** diterima sangat baik. Sentimen negatif (9,6%) lebih bersumber dari kenangan pribadi yang menyakitkan, **bukan penolakan terhadap kualitas karya**.

---

## SLIDE 10 — KESIMPULAN & SARAN

### Kesimpulan (Menjawab 3 Rumusan Masalah):

| # | Rumusan Masalah | Jawaban |
|---|---|---|
| **1** | Implementasi SVM + TF-IDF? | Pipeline KDD 6 tahap berhasil: *Scraping → Preprocessing (7 tahap) → TF-IDF → SVM Linear* |
| **2** | Kinerja model SVM? | **Akurasi 80,85%**, F1 80,66% — kompetitif untuk klasifikasi 3 kelas teks informal |
| **3** | Tren sentimen pendengar? | **Netral 64,2%** dominant → lagu diterima positif-netral, resonansi emosional tinggi |

### Kontribusi Penelitian:
- ✅ Pipeline NLP lengkap untuk teks bahasa Indonesia informal
- ✅ Solusi imbalance class → Recall Negatif naik dari 31% → 65%
- ✅ Dashboard Streamlit interaktif untuk prediksi sentimen real-time
- ✅ Dataset bersih berlabel 1.175 komentar sebagai *baseline* penelitian lanjutan

### Keterbatasan:
- ❌ Leksikon otomatis rentan pada sarkasme dan ironi implisit
- ❌ Bias nama entitas (Baskara, judul album) mempengaruhi pelabelan
- ❌ Model belum mampu deteksi sarkasme secara memadai

### Saran Pengembangan:
1. **IndoBERT / Transformer** → tingkatkan pemahaman konteks semantik
2. **Human Annotation** → label manual oleh 2–3 anotator ahli
3. **Named Entity Recognition (NER)** → filter nama tokoh dari skor leksikon
4. **Aspect-Based Sentiment Analysis (ABSA)** → analisis per aspek (vokal, lirik, aransemen)

---

> *Terima kasih*  
> **Rafi Maulana Firdaus | 312210382 | Teknik Informatika, Universitas Pelita Bangsa**
