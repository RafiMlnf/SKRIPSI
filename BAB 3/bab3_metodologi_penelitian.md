# BAB III

# METODE PENELITIAN

---

## 3.1 Jenis dan Pendekatan Penelitian

Penelitian ini menggunakan **pendekatan kuantitatif** dengan jenis **penelitian eksperimen komputasi**. Pendekatan kuantitatif digunakan karena seluruh hasil analisis diukur dan dievaluasi menggunakan data numerik yang dikumpulkan secara objektif dari proses ekstraksi fitur teks. Metode ini bersifat deskriptif-kuantitatif, dengan fokus utama pada evaluasi ketepatan model dalam mengklasifikasikan sentimen serta menggambarkan tren persepsi pendengar lagu _"Arteri"_ berdasarkan distribusi kelas sentimen yang dihasilkan.

Algoritma yang digunakan adalah **Support Vector Machine (SVM)**, sebuah algoritma _supervised learning_ yang bekerja dengan prinsip pencarian _hyperplane_ pemisah terbaik (_best separating hyperplane_) untuk memisahkan kelas data sentimen dengan margin yang maksimal.

---

## 3.2 Desain Penelitian

Desain penelitian ini disesuaikan dengan kerangka kerja **Knowledge Discovery in Databases (KDD)**, yang memproses data teks dalam tiga blok tahapan terintegrasi:

- **Blok Input**: Pengambilan data komentar mentah dari kolom komentar YouTube melalui teknik _web scraping_.
- **Blok Proses**: _Preprocessing_ teks (pembersihan, normalisasi, penghapusan stopwords, stemming), pembobotan kata dengan TF-IDF, pelabelan sentimen berbasis leksikon, dan pelatihan model SVM.
- **Blok Output**: Evaluasi kinerja model menggunakan _Confusion Matrix_ (Akurasi, Presisi, Recall, F1-Score) dan identifikasi tren sentimen dominan pendengar.

---

## 3.3 Tahapan Penelitian

Sebuah alur desain penelitian dibuat untuk memastikan penelitian ini berjalan secara sistematis, terarah, dan dapat dipertanggungjawabkan. Tahapan mencakup seluruh proses komputasi, mulai dari proses pengadaan data mentah hingga penarikan kesimpulan akhir dari kinerja algoritma.

```
[1] Pengumpulan Data      → Web scraping komentar YouTube (1.531 komentar)
         ↓
[2] Text Preprocessing    → Cleaning, Normalisasi, Stopwords, Stemming (1.175 bersih)
         ↓
[3] Pelabelan Sentimen    → Pelabelan berbasis leksikon (Positif/Netral/Negatif)
         ↓
[4] Ekstraksi Fitur       → TF-IDF Vectorization (max 5.000 fitur, bigram)
         ↓
[5] Pelatihan Model       → Random Oversampling + GridSearchCV + SVM Linear
         ↓
[6] Evaluasi & Analisis   → Confusion Matrix, Akurasi, Presisi, Recall, F1-Score
```

---

## 3.4 Variabel Penelitian

Dalam eksperimen komputasi ini, variabel penelitian didefinisikan sebagai sifat atau atribut objek penelitian yang memiliki variasi tertentu yang ditetapkan untuk diperiksa dan diambil kesimpulan.

### 3.4.1 Variabel Independen (X)

Variabel independen dalam penelitian ini adalah **fitur teks komentar**. Variabel ini berasal dari data tekstual mentah yang diambil dari kolom komentar video musik _"Arteri"_ di YouTube. Sebelum diproses oleh model, variabel ini mengalami perubahan melalui tahap _preprocessing_ dan pembobotan TF-IDF. Algoritma akan menggunakan nilai bobotnya untuk mempelajari pola setiap kata (_term_) yang berbeda dalam komentar tersebut.

### 3.4.2 Variabel Dependen (Y)

Variabel dependen dalam penelitian ini adalah **kelas sentimen** — label kategori _output_ yang diprediksi oleh model SVM berdasarkan fitur teks. Kinerja model diukur menggunakan metrik Akurasi, Presisi, Recall, dan F1-Score. Tiga kelas sentimen yang digunakan adalah:

- **Positif**: komentar yang menunjukkan dukungan, penghargaan, atau dorongan emosional.
- **Negatif**: komentar yang mengandung kritik tajam, ketidaksukaan, atau ekspresi negatif.
- **Netral**: komentar yang tidak memihak atau tidak mengandung emosi polar yang jelas.

---

## 3.5 Sumber dan Jenis Data

Penelitian ini menggunakan **data sekunder** yang diperoleh secara tidak langsung melalui media perantara dan tersedia secara publik. Sumber data adalah teks tidak terstruktur berupa komentar publik yang mengandung kritik, pendapat, dan tanggapan audiens dari kolom komentar video musik resmi lagu _"Arteri"_ oleh **.FEAST** di platform YouTube.

Data yang terkumpul menggunakan berbagai gaya bahasa, termasuk bahasa Indonesia baku, bahasa informal (_slang_), dan majas sarkasme, yang merepresentasikan perasaan pendengar terhadap tema yang diangkat dalam lagu.

---

## 3.6 Teknik Pengumpulan Data

Pengumpulan data menggunakan metode **observasi non-partisipan** dengan teknik **_web scraping_**. Teknik ini memungkinkan pengumpulan data dalam jumlah besar dari platform digital tanpa perlu berinteraksi langsung dengan subjek (pendengar).

### 3.6.1 Proses Scraping Data Mentah

Data komentar diperoleh melalui teknik _web scraping_ pada video musik resmi lagu _"Arteri"_ oleh .FEAST di platform YouTube menggunakan pustaka Python `youtube-comment-downloader`. Proses _scraping_ diselesaikan dan ditetapkan batas waktu penarikan data (_cutoff date_) pada **08 Juli 2026**, guna memastikan dataset bersifat statis selama seluruh proses pelatihan model, pengujian, hingga penyusunan laporan akhir, sehingga konsistensi nilai evaluasi dapat dipertahankan.

Dari proses _scraping_ ini, berhasil dikumpulkan sebanyak **1.531 komentar mentah** yang disimpan dalam format file `comments_arteri_full.csv`. Dataset mentah ini memiliki atribut sebagai berikut:

| Atribut    | Deskripsi                                               |
| :--------- | :------------------------------------------------------ |
| `author`   | Nama akun pengguna pembuat komentar                     |
| `text`     | Isi teks komentar pengguna                              |
| `time`     | Waktu penulisan komentar                                |
| `likes`    | Jumlah suka (_likes_) pada komentar                     |
| `replies`  | Jumlah balasan terhadap komentar                        |
| `is_reply` | Penanda apakah komentar merupakan balasan komentar lain |

### 3.6.2 Validasi dan Finalisasi Dataset

Setelah data mentah dikumpulkan, tahap validasi dilakukan untuk menentukan kelayakan komentar yang akan diolah lebih lanjut. Kriteria pengeluaran data meliputi komentar kosong, komentar yang hanya berisi simbol atau angka, salinan lirik lagu, komentar spam, dan komentar duplikat.

Dari **1.531 komentar mentah**, sebanyak **347 komentar dinyatakan tidak layak** sehingga dataset final yang siap diproses berjumlah **1.175 komentar bersih** (penyusutan 22,7%).

---

## 3.7 Instrumen Penelitian

### 3.7.1 Perangkat Keras (_Hardware_)

Seluruh proses komputasi dijalankan menggunakan perangkat keras sebagai berikut:

| Komponen         | Spesifikasi                                    |
| :--------------- | :--------------------------------------------- |
| Jenis Perangkat  | Personal Computer (PC)                         |
| Prosesor (CPU)   | AMD Ryzen 5 2600                               |
| Memori (RAM)     | 16 GB                                          |
| Penyimpanan      | SSD                                            |
| Koneksi Internet | LAN / WiFi                                     |

### 3.7.2 Perangkat Lunak (_Software_)

| Komponen                     | Keterangan                                                  |
| :--------------------------- | :---------------------------------------------------------- |
| Sistem Operasi               | Windows 11                                                  |
| Bahasa Pemrograman           | Python                                                      |
| Lingkungan Pengembangan      | Visual Studio Code / Jupyter Notebook                       |
| `pandas`                     | Manipulasi dan analisis dataset format DataFrame            |
| `scikit-learn`               | Implementasi TF-IDF, SVM, GridSearchCV, dan metrik evaluasi |
| `PySastrawi`                 | Stemming bahasa Indonesia                                   |
| `NLTK`                       | Tokenisasi dan pengelolaan stopwords                        |
| `imblearn`                   | Random Oversampling untuk menyeimbangkan kelas              |
| `joblib`                     | Penyimpanan model terlatih ke file `.pkl`                   |
| `matplotlib / seaborn`       | Visualisasi Confusion Matrix dan distribusi sentimen        |
| `wordcloud`                  | Visualisasi kata kunci dominan per kelas sentimen           |
| `youtube-comment-downloader` | Pengambilan data komentar YouTube                           |

---

## 3.8 Teknik Analisis Data

Penelitian ini menganalisis data melalui pendekatan _machine learning_ yang dilaksanakan dalam beberapa tahapan sistematis berikut:

### 3.8.1 Pemrosesan Awal Teks (_Text Preprocessing_)

Tahap awal analisis adalah pra-pemrosesan data mentah untuk menghilangkan gangguan (_noise_) dan menyeragamkan format data. Penelitian ini menerapkan tujuh tahap _preprocessing_ secara bertahap:

**1. Penyeragaman Huruf (_Case Folding_)**
Seluruh huruf dikonversi menjadi huruf kecil (_lowercase_) untuk menghindari perlakuan berbeda terhadap kata yang sama namun berbeda kapitalisasi.

- **Contoh**: `"Lagu ini Relate Banget"` → `"lagu ini relate banget"`

**2. Pembersihan Karakter (_Data Cleaning_)**
Menghapus elemen non-teks menggunakan _regular expression_ (regex): URL, penyebutan akun (`@mention`), tanda pagar (`#hashtag`), penunjuk waktu (_timestamp_), dan angka berlebih.

- **Contoh**: `"keren bgt @baskara #arteri"` → `"keren bgt"`

**3. Penghapusan Emoji dan Emotikon**
Emoji Unicode dan emotikon berbasis teks dihapus karena dapat mengacaukan proses tokenisasi dan vektorisasi TF-IDF.

- **Contoh**: `"sedih bgt 😭💔"` → `"sedih bgt"`

**4. Penyaringan Spam dan Lirik Lagu (_Spam & Lyrics Filtering_)**
Komentar yang hanya menyalin lirik lagu (_lyrics copy_) dan komentar spam dideteksi dan dipindahkan ke dataset terpisah. Deteksi lirik menggunakan tiga kriteria: keberadaan kata kunci struktural lagu, jumlah baris komentar > 10, dan keberadaan frase lirik spesifik lagu _"Arteri"_.

**5. Normalisasi Kata Gaul (_Slang Normalization_)**
Kata tidak baku, singkatan, dan bahasa gaul dinormalisasi menggunakan **Kamus Alay** kustom dengan lebih dari **4.300 entri kata**. Kata khas konteks (seperti _"feast"_, _"arteri"_, _"baskara"_) dimasukkan ke daftar pengecualian (_preserve words_).

- **Contoh**: `"gw ga suka lagu ini"` → `"saya tidak suka lagu ini"`

**6. Penghapusan Kata Henti (_Stopwords Removal_)**
Kata-kata umum tanpa nilai sentimen dihapus menggunakan kamus stopwords bahasa Indonesia yang diperluas. Kata negasi (_"tidak"_, _"bukan"_, _"belum"_) **dipertahankan** karena berperan sebagai pembalik polaritas sentimen.

- **Contoh**: `"lagu yang sangat bagus"` → `"lagu bagus"`

**7. Pengakaran Kata (_Stemming_)**
Kata berimbuhan dipetakan ke kata dasarnya menggunakan pustaka **PySastrawi** (Algoritma Nazief & Adriani). Kata kunci konteks dikecualikan dari proses _stemming_.

- **Contoh**: `"mendengarkan"` → `"dengar"`; `"menangis"` → `"tangis"`

**Tabel 3.1 Contoh Transformasi Teks Melalui Tahapan Preprocessing**

| Tahap                         | Teks Komentar                                                                   |
| :---------------------------- | :------------------------------------------------------------------------------ |
| **Mentah**                    | `"Lagu ini gw dengerin tiap hari 😭 bener bgt sm kehidupan gw skrg"`            |
| Setelah _Case Folding_        | `"lagu ini gw dengerin tiap hari 😭 bener bgt sm kehidupan gw skrg"`            |
| Setelah _Cleaning_ + Emoji    | `"lagu ini gw dengerin tiap hari bener bgt sm kehidupan gw skrg"`               |
| Setelah _Slang Normalization_ | `"lagu ini saya dengarkan tiap hari benar banget sama kehidupan saya sekarang"` |
| Setelah _Stopwords Removal_   | `"lagu dengarkan hari benar banget kehidupan sekarang"`                         |
| **Setelah _Stemming_**        | `"lagu dengar hari benar banget hidup sekarang"`                                |

**Tabel 3.2 Ringkasan Penghapusan Data pada Tahap Preprocessing**

| Kategori Penghapusan                             | Jumlah Komentar |
| :----------------------------------------------- | :-------------: |
| Komentar kosong (hanya spasi/karakter non-huruf) |       83        |
| Salinan lirik lagu (_Lyrics Copy_)               |       29        |
| Deteksi spam (promosi, karakter berulang)        |       108       |
| Terlalu pendek (< 2 kata atau < 5 karakter)      |       49        |
| Terlalu panjang (> 100 kata)                     |        1        |
| Komentar duplikat (_Exact Duplicates_)           |       77        |
| **Total Komentar Dihapus**                       |     **347**     |
| **Total Komentar Bersih yang Diproses**          |    **1.175**    |

### 3.8.2 Pelabelan Sentimen (_Sentiment Labeling_)

Komentar yang telah melalui _preprocessing_ diberi label kelas sentimen secara otomatis menggunakan metode **pelabelan berbasis leksikon (_Lexicon-Based_)**. Metode ini mencocokkan kata-kata dalam teks terhadap kamus sentimen bahasa Indonesia, di mana setiap kata memiliki bobot numerik polaritas:

- **Kata Positif**: bobot +1 hingga +3
- **Kata Negatif**: bobot -1 hingga -3
- **Kata Tidak Ditemukan**: bobot 0 (diabaikan)

Skor sentimen akhir suatu komentar dihitung dari akumulasi bobot seluruh kata yang ditemukan dalam kamus, dengan penanganan khusus:

- **Kata Penguat (_Intensifier_)**: Kata seperti _"sangat"_, _"banget"_ memperkuat skor sentimen dengan faktor pengali **1,5x**.
- **Penanganan Negasi**: Kata seperti _"tidak"_, _"bukan"_ membalikkan polaritas skor sentimen kata sesudahnya dengan bobot dikurangi 50%.

**Tabel 3.3 Batas Ambang (_Threshold_) Klasifikasi Kelas**

| Kondisi Skor   | Label Kelas          | Nilai Numerik |
| :------------- | :------------------- | :-----------: |
| Skor >= +1     | **Sentimen Positif** |       2       |
| -1 < Skor < +1 | **Sentimen Netral**  |       1       |
| Skor <= -1     | **Sentimen Negatif** |       0       |

### 3.8.3 Pembobotan Kata — Ekstraksi Fitur TF-IDF

Data teks yang telah berlabel diubah menjadi representasi vektor numerik menggunakan metode **Term Frequency - Inverse Document Frequency (TF-IDF)**. Konfigurasi parameter yang digunakan adalah:

| Parameter TF-IDF | Nilai                                      |
| :--------------- | :----------------------------------------- |
| `max_features`   | 5.000 kata unik maksimal                   |
| `ngram_range`    | (1, 2) — gabungan _unigram_ dan _bigram_   |
| `sublinear_tf`   | True — penskalaan logaritmik TF            |
| `min_df`         | 2 — kata harus muncul minimal di 2 dokumen |

### 3.8.4 Pembagian Data (_Data Splitting_)

Dataset yang telah berbentuk vektor dibagi menggunakan metode **Stratified Split** dengan rasio **80:20**:

- **Data Latih (_Training Set_)**: 940 komentar (80%) — digunakan untuk melatih model SVM.
- **Data Uji (_Testing Set_)**: 235 komentar (20%) — digunakan untuk menguji kinerja model pada data baru.

Metode _stratified split_ memastikan proporsi kelas sentimen tetap representatif di kedua bagian data.

### 3.8.5 Penanganan Ketidakseimbangan Kelas (_Class Imbalance_)

Dataset mengalami ketidakseimbangan kelas signifikan (Negatif 9,6%, Netral 64,2%, Positif 26,2%). Dua pendekatan diterapkan:

1. **Random Oversampling (ROS)**: Sampel kelas minoritas (Negatif) diduplikasi secara acak hingga distribusi kelas menjadi seimbang pada data latih.
2. **Pembobotan Kelas (`class_weight='balanced'`)**: SVM diberi penalti lebih besar pada kesalahan klasifikasi kelas minoritas selama pelatihan.

### 3.8.6 Klasifikasi dengan SVM

Algoritma SVM diterapkan pada data latih untuk membangun model klasifikasi. SVM memetakan data ke dalam ruang vektor berdimensi tinggi dan menemukan _hyperplane_ optimal **w · x + b = 0** dengan margin terbesar antara kelas sentimen. Kernel **Linear** digunakan karena cocok untuk data teks berdimensi tinggi seperti representasi TF-IDF. Strategi multi-kelas yang diterapkan adalah **One-vs-One (OvO)**, yang melatih 3 model biner kemudian menggabungkan hasilnya melalui mekanisme _voting_.

Pencarian _hyperparameter_ optimal dilakukan menggunakan **GridSearchCV** dengan skema **5-fold cross-validation** pada ruang pencarian berikut:

| Parameter            | Nilai yang Dicoba |
| :------------------- | :---------------- |
| `kernel`             | `'linear'`        |
| `C` (Regularization) | 0.1, 0.5, 1.0     |
| `gamma`              | `'scale'`         |

### 3.8.7 Evaluasi Model

Pada tahap akhir, kinerja model dievaluasi menggunakan **Confusion Matrix** dengan empat metrik utama:

- **Akurasi (_Accuracy_)**: Proporsi prediksi yang benar dari keseluruhan data uji.
- **Presisi (_Precision_)**: Tingkat ketepatan prediksi positif model.
- **Recall**: Seberapa banyak data positif sebenarnya yang berhasil dikenali model.
- **F1-Score**: Rata-rata harmonik antara Presisi dan Recall — menjadi acuan utama jika terdapat ketimpangan jumlah data antar kelas.
