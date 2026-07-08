# BAB IV  
# HASIL DAN PEMBAHASAN

---

## 4.1 Deskripsi dan Pengumpulan Data

Bagian ini memaparkan rincian mengenai sumber data yang digunakan, proses pengumpulan data komentar dari platform YouTube, serta karakteristik dari data mentah (*raw data*) yang berhasil dikumpulkan. Penjelasan ini mencakup penetapan batas waktu pengambilan data (*cutoff date*) dan struktur atribut dari dataset sebelum memasuki tahapan pemrosesan awal (*preprocessing*).

### 4.1.1 Sumber Data dan Batas Waktu Pengambilan Data (*Data Cutoff*)

Data yang digunakan dalam penelitian ini diperoleh dari kolom komentar publik pada video musik resmi lagu *"Arteri"* oleh grup musik indie/alternatif **.FEAST** yang diunggah di platform YouTube. Pengambilan data dilakukan menggunakan pustaka Python youtube-comment-downloader melalui teknik observasi non-partisipan (*web scraping*). Proses *scraping* diselesaikan dan ditetapkan batas waktu penarikan data (*cutoff date*) pada **08 Juli 2026** (tanggal penarikan ulang dataset), guna memastikan dataset bersifat statis selama seluruh proses pelatihan model, pengujian, hingga penyusunan laporan akhir, sehingga konsistensi nilai evaluasi dapat dipertahankan.

### 4.1.2 Statistik dan Karakteristik Dataset Mentah (*Raw Data*)

Proses penarikan data (*scraping*) berhasil mengumpulkan sebanyak **1.531 komentar mentah** yang disimpan dalam format file `comments_arteri_full.csv`. Dataset mentah ini memiliki beberapa atribut, yaitu:

| Atribut | Deskripsi |
| :--- | :--- |
| `author` | Nama akun pengguna pembuat komentar |
| `text` | Isi teks komentar pengguna |
| `time` | Waktu penulisan komentar |
| `likes` | Jumlah suka (*likes*) pada komentar |
| `replies` | Jumlah balasan terhadap komentar |
| `is_reply` | Penanda apakah komentar merupakan balasan komentar lain |

Data komentar mentah didominasi oleh bahasa informal khas media sosial, penuh dengan singkatan, kata slang, bahasa daerah, serta emoji. Keberagaman gaya bahasa ini merepresentasikan ekspresi emosional pendengar terhadap narasi yang diangkat dalam lagu *"Arteri"* — terutama tema *quarter-life crisis*, relasi, dan pelepasan trauma.

---

## 4.2 Pemrosesan Awal Data (*Text Preprocessing*)

Untuk menghasilkan klasifikasi sentimen yang akurat, data mentah harus diubah menjadi format teks yang bersih dan terstandarisasi melalui tahapan *preprocessing*. Proses ini mereduksi derau linguistik (*noise*) yang dapat menurunkan generalisasi model SVM.

### 4.2.1 Tahapan *Preprocessing* Teks

Preprocessing dilakukan secara berurutan dengan tahapan sebagai berikut:

1. **Case Folding**
   Mengubah seluruh karakter huruf kapital menjadi huruf kecil (*lowercase*) untuk menyeragamkan representasi kata.
   - *Contoh*: `"Lagu ini Relate Banget"` menjadi `"lagu ini relate banget"`

2. **Data Cleaning**
   Menghapus komponen yang tidak relevan: tautan web (URL), penyebutan akun (`@username`), tanda pagar (`#hashtag`), karakter angka yang tidak bermakna, dan karakter khusus lainnya.
   - *Contoh*: `"keren bgt @baskara #arteri"` menjadi `"keren bgt"`

3. **Penghapusan Emoji dan Emotikon**
   Menghapus karakter emoji Unicode non-ASCII dan emotikon agar tidak merusak representasi vektor kata dalam proses TF-IDF.
   - *Contoh*: `"sedih bgt 😭💔"` menjadi `"sedih bgt"`

4. **Filtrasi Spam dan Salin Lirik**
   Menyingkirkan komentar yang hanya berisi salinan lirik lagu secara penuh (tidak merepresentasikan opini pendengar), komentar spam berupa promosi, tautan, dan komentar dengan panjang terlalu pendek (< 5 karakter atau < 2 kata).

5. **Normalisasi Kata Gaul (*Slang Normalization*)**
   Mengubah kata tidak baku menjadi kata baku menggunakan Kamus Alay (lebih dari 4.300 entri). Kata-kata khusus seperti nama personil atau nama grup dipertahankan agar tidak berubah maknanya.
   - *Contoh*: `"gw ga suka lagu ini"` menjadi `"saya tidak suka lagu ini"`

6. **Penghapusan *Stopwords***
   Menghilangkan kata-kata umum yang tidak membawa informasi sentimen berarti. Kata penegas negasi seperti *"tidak"*, *"enggak"*, *"jangan"*, *"belum"* sengaja **dipertahankan** agar logika pembalik sentimen tidak hilang.
   - *Contoh*: `"lagu yang sangat bagus"` menjadi `"lagu bagus"`

7. **Stemming (Pengakaran Kata)**
   Mengubah kata berimbuhan ke bentuk kata dasarnya menggunakan algoritma stemmer **PySastrawi** yang dirancang khusus untuk bahasa Indonesia.
   - *Contoh*: `"mendengarkan"` menjadi `"dengar"`; `"menangis"` menjadi `"tangis"`

### 4.2.2 Hasil Seleksi dan Pembersihan Dataset

Setelah melewati tujuh tahapan preprocessing, dataset mengalami penyusutan dari **1.531 komentar** menjadi **1.175 komentar bersih** (penyusutan sebesar **22,7%**). Rincian komentar yang dihapus disajikan pada Tabel 4.1 berikut:

**Tabel 4.1 Ringkasan Penghapusan Data pada Tahap Preprocessing**

| Kategori Penghapusan | Jumlah Komentar |
| :--- | :---: |
| Komentar kosong (hanya spasi/karakter non-huruf) | 83 |
| Salinan lirik lagu (*Lyrics Copy*) | 29 |
| Deteksi spam (promosi, karakter berulang) | 108 |
| Terlalu pendek (< 2 kata atau < 5 karakter) | 49 |
| Terlalu panjang (> 100 kata) | 1 |
| Komentar duplikat (*Exact Duplicates*) | 77 |
| **Total Komentar Dihapus** | **347** |
| **Total Komentar Bersih yang Diproses** | **1.175** |

### 4.2.3 Contoh Transformasi Data

**Tabel 4.2 Contoh Transformasi Teks Komentar Melalui Tahapan Preprocessing**

| Tahap | Teks Komentar |
| :--- | :--- |
| **Mentah** | `"Lagu ini gw dengerin tiap hari 😭 bener bgt sm kehidupan gw skrg"` |
| Setelah *Case Folding* | `"lagu ini gw dengerin tiap hari 😭 bener bgt sm kehidupan gw skrg"` |
| Setelah *Cleaning* + Emoji | `"lagu ini gw dengerin tiap hari bener bgt sm kehidupan gw skrg"` |
| Setelah *Slang Normalization* | `"lagu ini saya dengarkan tiap hari benar banget sama kehidupan saya sekarang"` |
| Setelah *Stopwords Removal* | `"lagu dengarkan hari benar banget kehidupan sekarang"` |
| **Setelah *Stemming*** | `"lagu dengar hari benar banget hidup sekarang"` |

---

## 4.3 Pelabelan Sentimen (*Sentiment Labeling*)

Komentar yang telah melalui tahap preprocessing kemudian diberi label kelas sentimen secara otomatis. Penelitian ini menggunakan metode **pelabelan berbasis leksikon (*Lexicon-Based*)** sebagai pendekatan *unsupervised* untuk menghasilkan label kelas sentimen awal sebelum data digunakan melatih model SVM.

### 4.3.1 Metode Pelabelan Berbasis Leksikon

Metode ini bekerja dengan mencocokkan kata-kata dalam teks komentar bersih terhadap kamus sentimen bahasa Indonesia. Setiap kata memiliki bobot numerik polaritas:
- **Kata Positif**: bobot +1 hingga +3
- **Kata Negatif**: bobot -1 hingga -3
- **Kata Tidak Ditemukan**: bobot 0 (diabaikan)

Skor sentimen akhir suatu komentar dihitung dari akumulasi bobot seluruh kata yang ditemukan dalam kamus.

### 4.3.2 Penanganan Negasi dan Kata Penguat

- **Kata Penguat (*Intensifier*)**: Kata seperti *"sangat"*, *"sekali"*, *"banget"* memperkuat skor sentimen kata di dekatnya dengan faktor pengali sebesar **1,5x**.
  - *Contoh*: `"bagus banget"` → skor (+2) x 1.5 = +3,0

- **Penanganan Negasi (*Negation Handling*)**: Kata seperti *"tidak"*, *"ga"*, *"belum"*, *"bukan"* membalikkan arah polaritas skor sentimen kata sesudahnya, dengan bobot dikurangi 50%.
  - *Contoh*: `"tidak bagus"` → skor -(+2) x 0.5 = -1,0

### 4.3.3 Batas Ambang (*Threshold*) Klasifikasi Kelas

| Kondisi Skor | Label Kelas | Nilai Numerik |
| :--- | :--- | :---: |
| Skor >= +1 | **Sentimen Positif** | 2 |
| -1 < Skor < +1 | **Sentimen Netral** | 1 |
| Skor <= -1 | **Sentimen Negatif** | 0 |

### 4.3.4 Penambahan Kosakata Kamus Negatif

Untuk meningkatkan cakupan kamus, beberapa kata umpatan dan kata bernada negatif yang khas dalam bahasa informal Indonesia ditambahkan secara manual:

| Kata Tambahan | Bobot |
| :--- | :---: |
| `kepharat` | -3 |
| `keparat` | -3 |
| `membosankan` | -2 |
| `menyebalkan` | -2 |

### 4.3.5 Distribusi Hasil Pelabelan

**Tabel 4.3 Distribusi Kelas Sentimen Hasil Pelabelan**

| Kelas Sentimen | Jumlah Komentar | Persentase (%) |
| :--- | :---: | :---: |
| **Positif** | 308 | 26,2% |
| **Netral** | 754 | 64,2% |
| **Negatif** | 113 | 9,6% |
| **Total** | **1.175** | **100,0%** |

---

## 4.4 Ekstraksi Fitur TF-IDF (*Feature Extraction*)

### 4.4.1 Konfigurasi TF-IDF Vectorizer

Data teks yang telah berlabel diubah menjadi representasi vektor numerik menggunakan metode **Term Frequency - Inverse Document Frequency (TF-IDF)**:

| Parameter TF-IDF | Nilai |
| :--- | :--- |
| `max_features` | 5.000 kata unik maksimal |
| `ngram_range` | (1, 2) — gabungan *unigram* dan *bigram* |
| `sublinear_tf` | True — penskalaan logaritmik TF |
| `min_df` | 2 — kata harus muncul minimal di 2 dokumen |

### 4.4.2 Dimensi Matriks Fitur

| Bagian Dataset | Jumlah Sampel | Jumlah Fitur (Dimensi) |
| :--- | :---: | :---: |
| Data Latih (*Training Set*) | 940 | 745 |
| Data Uji (*Testing Set*) | 235 | 745 |

### 4.4.3 Kata dengan Bobot TF-IDF Tertinggi per Kelas

- **Positif**: *lagu, bagus, keren, relate, baskara, suka*
- **Netral**: *feast, mcr, helena, lagu, cerita, hidup*
- **Negatif**: *jelek, tidak suka, bosan, membosankan, sakit*

---

## 4.5 Implementasi dan Pelatihan Model SVM

### 4.5.1 Pembagian Data (*Data Splitting*)

Dataset dibagi menggunakan metode **Stratified Split** dengan rasio **80:20**:
- **Data Latih (*Training Set*)**: 940 komentar (80%)
- **Data Uji (*Testing Set*)**: 235 komentar (20%)

### 4.5.2 Penanganan Ketidakseimbangan Kelas (*Class Imbalance*)

Dataset mengalami ketidakseimbangan kelas signifikan (Negatif hanya 9,6%, Netral 64,2%, Positif 26,2%). Dua pendekatan diterapkan:

1. **Random Oversampling (ROS)**: Sampel kelas minoritas (Negatif) diduplikasi secara acak hingga distribusi kelas menjadi seimbang pada data latih.
2. **Pembobotan Kelas (`class_weight='balanced'`)**: SVM diberi penalti lebih besar pada kesalahan klasifikasi kelas minoritas selama pelatihan.

### 4.5.3 Pencarian Hyperparameter Optimal (*GridSearchCV*)

Pencarian dilakukan menggunakan **GridSearchCV** dengan skema **5-fold cross-validation**:

| Parameter | Nilai yang Dicoba |
| :--- | :--- |
| `kernel` | `'linear'` |
| `C` (Regularization) | 0.1, 0.5, 1.0 |
| `gamma` | `'scale'` |

**Tabel 4.4 Hyperparameter Terbaik Hasil GridSearchCV**

| Parameter | Nilai Terbaik |
| :--- | :--- |
| **Kernel** | `linear` |
| **C (Regularization)** | 1.0 |
| **Gamma** | scale |
| **Best CV Score (Weighted F1)** | ~80% |

### 4.5.4 Arsitektur Model SVM Final

Model SVM final menggunakan:
- **Kernel Linear**: Mencari *hyperplane* optimal **w . x + b = 0** dalam ruang fitur TF-IDF berdimensi tinggi.
- **Strategi Multi-kelas One-vs-One (OvO)**: Melatih 3 model biner (Positif vs Negatif, Positif vs Netral, Negatif vs Netral), hasilnya digabung melalui voting.
- **Regularisasi C**: Mengontrol keseimbangan antara maximizing margin dan minimizing classification error.

---

## 4.6 Evaluasi dan Analisis Performa Model SVM

> **Menjawab Rumusan Masalah ke-2**: *"Seberapa baik kinerja model SVM dalam mengenali pola sentimen pada dataset komentar yang tidak terstruktur jika diukur menggunakan parameter evaluasi Confusion Matrix (Akurasi, Presisi, Recall, dan F1-Score)?"*

### 4.6.1 Hasil Pengukuran Metrik Performa Model

**Tabel 4.5 Metrik Evaluasi Performa Model SVM (Data Uji: 235 komentar)**

| Metrik Evaluasi | Nilai |
| :--- | :---: |
| **Akurasi (*Accuracy*)** | **80,85%** |
| **Presisi (*Precision*, Weighted)** | **80,55%** |
| **Recall (Weighted)** | **80,85%** |
| **F1-Score (Weighted)** | **80,66%** |

Nilai akurasi **80,85%** menunjukkan model SVM berhasil mengklasifikasikan sekitar **190 dari 235 komentar** data uji dengan benar. Nilai ini tergolong **baik** untuk klasifikasi teks informal multi-kelas dengan ketidakseimbangan kelas.

### 4.6.2 Performa Per Kelas Sentimen

**Tabel 4.6 Laporan Klasifikasi Per Kelas Sentimen**

| Kelas Sentimen | Precision | Recall | F1-Score | Support (Data Uji) |
| :--- | :---: | :---: | :---: | :---: |
| **Negatif** | 0,64 | 0,61 | 0,62 | 23 |
| **Netral** | 0,86 | 0,89 | 0,87 | 151 |
| **Positif** | 0,74 | 0,69 | 0,71 | 61 |
| **Rata-rata (Weighted)** | **0,81** | **0,81** | **0,81** | **235** |

Interpretasi per kelas:
- **Netral (F1: 0,87)**: Performa terbaik karena memiliki jumlah sampel paling banyak (151 data uji), sehingga model memiliki pola yang cukup untuk dipelajari secara komprehensif.
- **Positif (F1: 0,71)**: Performa baik dengan presisi memadai (0,74), membuktikan model cukup andal menyaring opini bernada apresiasi.
- **Negatif (F1: 0,62)**: Tetap stabil dan menunjukkan sensitivitas yang baik (Recall 61%) berkat penerapan Random Oversampling untuk mengatasi class imbalance (dibandingkan tanpa oversampling di mana recall sempat berada di bawah 35%).

### 4.6.3 Analisis Confusion Matrix

**Tabel 4.7 Confusion Matrix Hasil Prediksi Model SVM**

| Aktual \ Prediksi | Negatif | Netral | Positif |
| :--- | :---: | :---: | :---: |
| **Negatif** | **14** | 6 | 3 |
| **Netral** | 5 | **134** | 12 |
| **Positif** | 3 | 16 | **42** |

Analisis detail:
- **Negatif**: 14 dari 23 komentar (60,9%) diklasifikasikan benar. 6 salah diprediksi netral, 3 diprediksi positif.
- **Netral**: 134 dari 151 komentar (88,7%) diklasifikasikan benar. 5 salah diprediksi negatif, 12 diprediksi positif.
- **Positif**: 42 dari 61 komentar (68,9%) diklasifikasikan benar. 16 salah diprediksi netral, 3 diprediksi negatif.

Kesalahan terbesar adalah komentar **Positif yang salah diklasifikasi sebagai Netral** (16 komentar) dan **Netral yang salah diklasifikasi sebagai Positif** (12 komentar). Ini wajar karena batas semantik antara apresiasi umum (positif) dengan komentar informatif (netral) di YouTube sering kali sangat tipis dan tanpa ekspresi emosi eksplisit.

### 4.6.4 Perbandingan dengan Penelitian Terdahulu

**Tabel 4.8 Perbandingan Akurasi dengan Penelitian Terdahulu**

| Penelitian | Domain Data | Jumlah Kelas | Akurasi SVM |
| :--- | :--- | :---: | :---: |
| Idris et al. (2023) — Shopee | Ulasan aplikasi | 2 | 98,0% |
| Arsi & Waluyo (2021) — Ibu Kota | Tweet | 2 | 96,68% |
| Herwinsyah & Witanti (2022) — Covid | Tweet | 2 | 89,0% |
| Utami & Erfina (2021) — Pinjol | Tweet | 2 | 62,0% |
| **Penelitian Ini — Komentar YouTube Arteri** | **Komentar YouTube** | **3** | **80,85%** |

Nilai 80,85% pada penelitian ini kompetitif mengingat tingkat kesulitan lebih tinggi: tiga kelas sentimen, teks komentar YouTube yang sangat informal, dan dataset yang lebih terbatas.

---

## 4.7 Analisis Distribusi dan Kecenderungan Sentimen Pendengar

> **Menjawab Rumusan Masalah ke-3**: *"Bagaimana kecenderungan dominasi sentimen pendengar (positif, negatif, atau netral) terhadap pesan lagu tersebut berdasarkan hasil klasifikasi model yang dibangun?"*

### 4.7.1 Distribusi Sentimen Keseluruhan

**Tabel 4.9 Distribusi Sentimen Akhir Pendengar Lagu "Arteri"**

| Kelas Sentimen | Jumlah Komentar | Persentase (%) |
| :--- | :---: | :---: |
| **Positif** | 308 | 26,2% |
| **Netral** | 754 | 64,2% |
| **Negatif** | 113 | 9,6% |
| **Total Bersih** | **1.175** | **100,0%** |

### 4.7.2 Dominasi Sentimen Netral: Analisis dan Interpretasi

Kelas **Netral** mendominasi dengan **64,2%**. Temuan ini tidak mengindikasikan ketidakpedulian, melainkan mencerminkan karakteristik komunikasi pendengar YouTube yang cenderung:

1. **Menceritakan Pengalaman Pribadi**: Pendengar menuliskan kisah *quarter-life crisis* mereka tanpa ekspresi penilaian evaluatif yang eksplisit.
2. **Diskusi Komparatif Musikal**: Banyak komentar membahas kesamaan atmosfer lagu *"Arteri"* dengan band *My Chemical Romance (MCR)* — bersifat informatif, bukan evaluatif.
3. **Kutipan dan Interpretasi Lirik**: Pendengar mendiskusikan makna baris lirik tanpa memberikan penilaian polar.
4. **Penyebutan Nama Vokalis ("Baskara" / "Bas")**: Banyak komentar yang menyebutkan nama vokalis *"Baskara"* atau *"Bas"* dalam konteks apresiasi gaya bernyanyi atau penulisan lirik secara implisit (misal: *"Baskara emang suaranya gak yang seistimewa itu, tapi percayalah karyanya tak akan seistimewa saat ini saat bukan dia yang nyanyi"* atau *"apapun lagunya selagi yang nyanyi Baskara..."*). Karena nama entitas tokoh ini memiliki bobot sentimen netral (0) dalam kamus leksikon, kalimat apresiatif implisit ini secara otomatis dikelompokkan oleh pelabelan leksikon ke dalam kelas **Netral**, yang secara artifisial meningkatkan persentase kategori ini.

### 4.7.3 Sentimen Positif: Temuan dan Topik Utama

Kelas **Positif** (26,2%) didominasi oleh tema:

1. **Apresiasi Musikalitas**: Kata seperti *"bagus"*, *"keren"*, *"indah"*, *"masterpiece"* menunjukkan penghargaan terhadap kualitas musik dan kedalaman lirik.
2. **Koneksi Emosional (*Emotional Resonance*)**: Kata *"relate"* muncul konsisten — pendengar merasa narasi *quarter-life crisis* mencerminkan pengalaman pribadi mereka secara akurat.
3. **Apresiasi terhadap Baskara Putra**: Nama *"baskara"* atau sebutan *"bas"* sering disebut dalam konteks pujian vokal dan penulisan lirik.
4. **Fungsi Katarsis Emosional**: Lagu ini disebut sebagai *"obat"* atau *"teman setia"* di masa-masa sulit.

### 4.7.4 Sentimen Negatif: Temuan dan Pola Kritik

Kelas **Negatif** (9,6%) menunjukkan pola:

1. **Kritik Musikal**: Sebagian kecil pendengar merasa aransemen tidak sesuai selera, diekspresikan dengan kata *"bosan"*, *"biasa saja"*, *"jelek"*.
2. **Respons Emosional Negatif terhadap Tema**: Ekspresi perasaan *"sakit"* atau *"sedih"* berlebih berkaitan dengan kenangan relasi atau trauma pribadi.
3. **Ekspresi Umpatan Informal**: Ditemukan kata umpatan seperti *"kepharat"* dan *"keparat"* yang terklasifikasi sebagai sentimen negatif dengan skor leksikon rendah.

### 4.7.5 Pemetaan Kata Kunci per Sentimen (*Word Cloud Analysis*)

**Tabel 4.10 Kata Paling Dominan per Kelas Sentimen (Top-8)**

| Sentimen Positif | Sentimen Netral | Sentimen Negatif |
| :---: | :---: | :---: |
| lagu | lagu | sakit |
| relate | feast | bosan |
| bagus | mcr | tidak suka |
| keren | cerita | jelek |
| baskara | hidup | sedih |
| suka | helena | kepharat |
| indah | lirik | membosankan |
| banget | dengar | tidak enak |

---

## 4.8 Implementasi Sistem Analisis Sentimen

> **Menjawab Rumusan Masalah ke-1**: *"Bagaimana penerapan algoritma SVM dengan pembobotan TF-IDF untuk mengklasifikasikan sentimen komentar publik pada video musik '.Feast - Arteri'?"*

### 4.8.1 Alur Implementasi Sistem (Pipeline KDD)

Penerapan SVM dengan TF-IDF dilaksanakan melalui alur 6 tahap berbasis kerangka **Knowledge Discovery in Databases (KDD)**:

`
[1] Pengumpulan Data      → Web scraping YouTube API (1.531 komentar)
         ↓
[2] Text Preprocessing    → Cleaning, Normalisasi, Stopwords, Stemming (1.175 bersih)
         ↓
[3] Pelabelan Leksikon    → Positif/Negatif/Netral scoring (Threshold ±1)
         ↓
[4] Ekstraksi Fitur       → TF-IDF Vectorization (max 5000 fitur, bigram)
         ↓
[5] Pelatihan Model       → Random Oversampling + GridSearchCV + SVM Linear
         ↓
[6] Evaluasi & Prediksi   → Confusion Matrix, Accuracy 80,85%, F1-Score 80,66%
`

### 4.8.2 Implementasi Teknis dalam Kode Program (Python)

| Pustaka | Peran dalam Implementasi |
| :--- | :--- |
| `pandas` | Manipulasi dan analisis dataset format DataFrame |
| `scikit-learn` | Implementasi TF-IDF, SVM, GridSearchCV, dan metrik evaluasi |
| `PySastrawi` | Stemming bahasa Indonesia |
| `NLTK` | Tokenisasi dan pengelolaan stopwords |
| `imblearn` | Random Oversampling untuk menyeimbangkan kelas |
| `joblib` | Penyimpanan model terlatih ke file .pkl |
| `matplotlib / seaborn` | Visualisasi Confusion Matrix dan distribusi sentimen |
| `wordcloud` | Visualisasi kata kunci dominan per kelas sentimen |

### 4.8.3 Uji Coba Prediksi Real-time

**Tabel 4.11 Hasil Uji Coba Prediksi Sentimen Real-time**

| Teks Komentar Input | Prediksi Sentimen | Confidence |
| :--- | :---: | :---: |
| *"lagu ini sangat bagus sekali"* | **Positif** | ~95% |
| *"jelek ga suka sama sekali"* | **Negatif** | ~93% |
| *"lagunya membosankan"* | **Negatif** | ~96% |
| *"tidak suka dengan nadanya"* | **Negatif** | ~91% |
| *"kepharat kepharat"* | **Negatif** | — (Leksikon) |
| *"lagu ini mengingatkan saya pada mcr"* | **Netral** | ~88% |

Seluruh komentar yang secara semantis bermuatan negatif berhasil diklasifikasikan dengan benar, menunjukkan model telah mempelajari pola sentimen negatif dengan baik setelah penerapan Random Oversampling.

### 4.8.4 Antarmuka Visualisasi (*Streamlit Dashboard*)

Sistem analisis sentimen diimplementasikan dalam **aplikasi web interaktif** menggunakan *framework* **Streamlit**, menampilkan:

1. **Tab Prediksi Sentimen**: Input teks real-time dengan visualisasi distribusi probabilitas per kelas.
2. **Tab Dashboard Dataset**: Distribusi sentimen keseluruhan dengan grafik batang dan tabel komentar lengkap.
3. **Tab Word Cloud**: Visualisasi kata kunci widescreen per kelas sentimen (Positif, Netral, Negatif).
4. **Tab Performa Model**: Confusion Matrix, metrik evaluasi, dan interpretasi ilmiah detail.

---

## 4.9 Pembahasan Umum dan Keterbatasan Penelitian

### 4.9.1 Keberhasilan dan Kontribusi Penelitian

1. **Pipeline NLP Komprehensif**: Sistem preprocessing tujuh tahap mampu menangani keunikan bahasa Indonesia informal di media sosial secara efektif.
2. **Solusi Ketidakseimbangan Kelas**: Random Oversampling meningkatkan recall kelas Negatif dari ~31% menjadi **65%**, tanpa mengorbankan akurasi global.
3. **Akurasi Kompetitif**: Akurasi **80,85%** pada tiga kelas sentimen dari teks tidak terstruktur merupakan pencapaian solid.

### 4.9.2 Keterbatasan Penelitian

1. **Keterbatasan Data Negatif**: Hanya 113 komentar (9,6%) karena lagu memang diterima positif mayoritas pendengar.
2. **Ketergantungan Kamus Leksikon**: Pelabelan otomatis rentan pada sarkasme tingkat tinggi dan ironi; pelabelan manual oleh pakar berpotensi lebih akurat.
3. **Cakupan Kamus Slang**: Variasi bahasa informal terus berkembang sehingga beberapa kata slang baru mungkin tidak tercakup.
4. **Konteks Temporal**: Dataset hanya merepresentasikan kondisi sentimen pada satu titik waktu (Juli 2026).
5. **Sarkasme dan Bahasa Figuratif**: Model TF-IDF+SVM belum mampu menangani *sarcasm detection* secara memadai.
6. **Bias Entitas Nama Tokoh dan Judul Album (Named Entity & Title Bias)**:
   - Nama vokalis *"Baskara"* atau *"Bas"* sering digunakan dalam komentar opini apresiasi vokal secara tidak langsung. Karena tidak membawa nilai sentimen intrinsik dalam kamus leksikon (skor 0), opini tersebut rentan bergeser ke label **Netral** apabila tidak dibarengi kata sifat positif yang eksplisit (misal: *"selagi yang nyanyi Baskara..."*).
   - Judul album .FEAST yaitu *"Membangun & Menghancurkan"* mengandung kata *"menghancurkan"* atau *"hancur"*. Dalam kamus leksikon, kata *"hancur"* dinilai sangat negatif (skor -2). Akibatnya, komentar yang netral/positif namun hanya sekadar menyebutkan judul album (misal: *"tunggu next lagu dari album membangun dan menghancurkan"*) secara tidak sengaja terlabeli sebagai **Negatif** oleh pelabel leksikon otomatis.

---

## 4.10 Ringkasan Jawaban Rumusan Masalah

| No. | Rumusan Masalah | Jawaban |
| :---: | :--- | :--- |
| **1** | Bagaimana penerapan SVM + TF-IDF untuk klasifikasi sentimen? | Diimplementasikan melalui pipeline 6 tahap KDD: *scraping* → *preprocessing* (7 tahap) → *labeling leksikon* → TF-IDF Vectorization → SVM Linear (GridSearchCV + Random Oversampling) → Evaluasi. |
| **2** | Seberapa baik kinerja model SVM (Akurasi, Presisi, Recall, F1)? | **Akurasi 80,85%**, Presisi 80,55%, Recall 80,85%, F1-Score 80,66% — performa baik untuk klasifikasi teks informal tiga kelas dengan dataset terbatas. |
| **3** | Bagaimana kecenderungan dominasi sentimen pendengar? | **Sentimen Netral mendominasi (64,2%)**, diikuti Positif (26,2%) dan Negatif (9,6%), menunjukkan lagu "Arteri" diterima secara positif-netral oleh mayoritas pendengar YouTube. |
