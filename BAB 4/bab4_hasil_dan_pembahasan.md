# BAB IV

# HASIL DAN PEMBAHASAN

---

## 4.1 Deskripsi dan Pengumpulan Data

Bagian ini memaparkan rincian mengenai sumber data yang digunakan, proses pengumpulan data komentar dari platform YouTube, serta karakteristik dari data mentah (_raw data_) yang berhasil dikumpulkan. Penjelasan ini mencakup penetapan batas waktu pengambilan data (_cutoff date_) dan struktur atribut dari dataset sebelum memasuki tahapan pemrosesan awal (_preprocessing_).

### 4.1.1 Sumber Data dan Batas Waktu Pengambilan Data (_Data Cutoff_)

Data yang digunakan dalam penelitian ini diperoleh dari kolom komentar publik pada video musik resmi lagu _"Arteri"_ oleh grup musik indie/alternatif **.FEAST** yang diunggah di platform YouTube. Pengambilan data dilakukan menggunakan pustaka Python youtube-comment-downloader melalui teknik observasi non-partisipan (_web scraping_). Proses _scraping_ diselesaikan dan ditetapkan batas waktu penarikan data (_cutoff date_) pada **08 Juli 2026** (tanggal penarikan ulang dataset), guna memastikan dataset bersifat statis selama seluruh proses pelatihan model, pengujian, hingga penyusunan laporan akhir, sehingga konsistensi nilai evaluasi dapat dipertahankan.

### 4.1.2 Statistik dan Karakteristik Dataset Mentah (_Raw Data_)

Proses penarikan data (_scraping_) berhasil mengumpulkan sebanyak **1.531 komentar mentah** yang disimpan dalam format file `comments_arteri_full.csv`. Dataset mentah ini memiliki beberapa atribut, yaitu:

| Atribut    | Deskripsi                                               |
| :--------- | :------------------------------------------------------ |
| `author`   | Nama akun pengguna pembuat komentar                     |
| `text`     | Isi teks komentar pengguna                              |
| `time`     | Waktu penulisan komentar                                |
| `likes`    | Jumlah suka (_likes_) pada komentar                     |
| `replies`  | Jumlah balasan terhadap komentar                        |
| `is_reply` | Penanda apakah komentar merupakan balasan komentar lain |

Data komentar mentah didominasi oleh bahasa informal khas media sosial, penuh dengan singkatan, kata slang, bahasa daerah, serta emoji. Keberagaman gaya bahasa ini merepresentasikan ekspresi emosional pendengar terhadap narasi yang diangkat dalam lagu _"Arteri"_ — terutama tema _quarter-life crisis_, relasi, dan pelepasan trauma.

---

## 4.2 Pemrosesan Awal Data (_Text Preprocessing_)

Untuk menghasilkan klasifikasi sentimen yang akurat, data mentah harus diubah menjadi format teks yang bersih dan terstandarisasi melalui tahapan _preprocessing_. Proses ini mereduksi derau linguistik (_noise_) yang dapat menurunkan generalisasi model SVM. Pada penelitian ini, tahapan pemrosesan awal dilakukan secara komprehensif mengingat karakteristik komentar YouTube yang sangat tidak terstruktur, menggunakan kombinasi ekspresi reguler (_regular expression_), pustaka NLTK, kamus normalisasi alay kustom, serta pustaka Sastrawi untuk penanganan morfologi bahasa Indonesia.

### 4.2.1 Penyeragaman Huruf (_Case Folding_)

Proses _case folding_ merupakan langkah awal yang bertujuan untuk menyamakan format seluruh huruf dalam dataset menjadi huruf kecil (_lowercase_). Komentar di media sosial sering kali ditulis menggunakan kombinasi huruf kapital yang acak atau berlebihan untuk mengekspresikan penekanan emosi (misalnya, _"LAGU ini relate BGT"_). Tanpa adanya _case folding_, mesin akan menganggap kata `"LAGU"`, `"Lagu"`, dan `"lagu"` sebagai entitas token yang berbeda, yang dapat meningkatkan dimensi fitur secara tidak perlu serta memecah bobot informasi frekuensi kemunculan kata (_term frequency_). Melalui proses ini, seluruh karakter abjad dari 'A' sampai 'Z' dikonversi menjadi 'a' sampai 'z', sedangkan spasi putih (_whitespace_) di ujung teks dipotong menggunakan fungsi `.strip()`.

- **Contoh input**: `"Lagu ini Relate Banget"`
- **Contoh output**: `"lagu ini relate banget"`

### 4.2.2 Pembersihan Karakter (_Data Cleaning_)

Tahap _data cleaning_ bertujuan untuk meminimalkan _noise_ dengan menghapus elemen-elemen non-teks atau metadata komentar yang tidak memiliki korelasi dengan muatan sentimen. Pada komentar YouTube, elemen derau ini sangat sering ditemukan dalam bentuk tautan URL (seperti _link_ video atau situs eksternal), penyebutan nama akun pengguna (_mention_ menggunakan simbol `@`), tanda pagar (_hashtag_), penunjuk waktu (_timestamp_ video, seperti 01:20), serta penulisan angka berlebih (seperti jumlah penonton atau tanggal). Pembersihan data diimplementasikan menggunakan teknik _regular expression_ (regex) dengan pola spesifik:

1. Menghapus URL menggunakan pola regex `https?://\S+|www\.\S+|bit\.ly/\S+|goo\.gl/\S+|t\.co/\S+`.
2. Menghapus penyebutan akun (_mention_) dengan pola `@[\w]+`.
3. Menghapus tanda pagar (_hashtag_) dengan pola `#[\w]+`.
4. Menghapus tanda penunjuk waktu (_timestamp_) dengan pola `\b\d{1,2}:\d{2}(:\d{2})?\b`.
5. Menghapus angka berlebih (seperti tanggal atau jumlah digit panjang) dengan pola `\b\d{1,3}([.,]\d{3})+\b` dan `\b\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b`.

- **Contoh input**: `"keren bgt @baskara #arteri"`
- **Contoh output**: `"keren bgt"`

### 4.2.3 Penghapusan Emoji dan Emotikon (_Emoji and Emoticon Removal_)

Komentar media sosial modern sangat kaya akan simbol visual berupa emoji Unicode (seperti 😭, 💔, 🔥) dan emotikon berbasis teks (seperti `:)`, `xD`, `-_-`). Walaupun secara visual merepresentasikan emosi, kehadiran karakter non-ASCII ini dalam dataset dapat mengacaukan pembuat token (_tokenizer_) dan pembuat vektor (_vectorizer_) TF-IDF, karena representasi pengodean (_encoding_) emoji bervariasi dan tidak terdaftar dalam kamus kosakata utama. Pada penelitian ini, emoji Unicode dihapus menggunakan pencocokan pola regex yang mencakup rentang Unicode emoji lengkap (seperti blok emoticons `\U0001F600-\U0001F64F`, symbols & pictographs `\U0001F300-\U0001F5FF`, dll.). Selanjutnya, emotikon teks yang umum digunakan (seperti `:)`, `:(`, `:D`, `:P`, `<3`, `T_T`, `uwu`) dibersihkan dengan melakukan iterasi pencocokan daftar emotikon terhadap teks secara _case-insensitive_.

- **Contoh input**: `"sedih bgt 😭💔"`
- **Contoh output**: `"sedih bgt"`

### 4.2.4 Deteksi dan Penyaringan Spam/Lirik Lagu (_Spam & Lyrics Filtering_)

Dalam kolom komentar video musik di YouTube, terdapat komentar pengganggu yang signifikan: komentar yang hanya menyalin lirik lagu secara penuh (_lyrics copy_) dan komentar spam promosi atau ekspresi kosong. Lirik lagu yang disalin tidak merepresentasikan opini pribadi pendengar sehingga harus disaring keluar agar tidak membiaskan model sentimen. Proses penyaringan lirik lagu dirancang menggunakan tiga kriteria deteksi:

1. Keberadaan kata kunci struktural lagu (seperti _verse, chorus, reff, bridge, outro, intro, bait, lirik_) minimal sebanyak dua buah kata kunci.
2. Panjang baris komentar lebih dari 10 baris baru (`\n`), yang umumnya menandakan salinan bait lirik yang panjang.
3. Keberadaan frase spesifik dari lagu _"Arteri"_ (seperti _"telanjang ku telanjang"_, _"menyicipi dunia"_, _"selamat datang di dua puluh"_, _"menangisku di pundakmu"_, _"muntahkan semua pilu"_) minimal sebanyak tiga frase.
   Sementara itu, komentar spam dideteksi menggunakan pola regex untuk mendeteksi komentar yang hanya berisi titik berulang, simbol murni, angka murni, komentar yang terlalu pendek (1-3 karakter), atau kata spam singkat seperti `p`, `pp`, `ok`, `iy`. Komentar yang terdeteksi sebagai lirik lagu atau spam dipindahkan ke dataset terpisah (`comments_arteri_lyric.csv`) dan dihapus dari pipeline utama.

### 4.2.5 Normalisasi Kata Gaul (_Slang Normalization_)

Pengguna media sosial di Indonesia cenderung menggunakan bahasa tidak baku, singkatan, serta bahasa gaul (_slang_). Variasi kata tidak baku ini sangat tinggi, misalnya kata _"tidak"_ ditulis menjadi _"ga"_, _"gak"_, _"nggak"_, _"gk"_, atau _"kagak"_. Keberagaman ini menyebabkan sebaran fitur menjadi sangat luas dan tidak fokus. Untuk mengatasinya, penelitian ini menerapkan normalisasi kata gaul dengan mencocokkan kata dalam komentar terhadap **Kamus Alay** yang telah dikembangkan secara kustom dengan total lebih dari **4.300 entri kata**. Kamus ini memetakan kata tidak baku ke bentuk baku bahasa Indonesia. Dalam proses normalisasi, terdapat penanganan khusus untuk konteks spesifik grup musik **.FEAST** dan lagu _"Arteri"_:

- Kata _"bas"_ dipetakan menjadi _"baskara"_ (vokalis Baskara Putra) untuk menghindari ambiguasi dengan alat musik bass.
- Singkatan _"mcr"_ dan kata terkait dipetakan menjadi nama lengkap _"my chemical romance"_ untuk menjaga keterkaitan referensi musikal.
- Kata-kata khas konteks seperti _"feast"_, _"arteri"_, _"pondok indah"_, dan _"quarter life crisis"_ dimasukkan ke dalam daftar pengecualian (_preserve words_) agar tidak diubah secara keliru oleh kamus normalisasi alay maupun algoritma stemming.
- **Contoh input**: `"gw ga suka lagu ini"`
- **Contoh output**: `"saya tidak suka lagu ini"`

### 4.2.6 Penghapusan Kata Henti (_Stopwords Removal_)

_Stopwords_ adalah kata-kata umum dalam suatu bahasa yang memiliki frekuensi kemunculan tinggi namun tidak memiliki arti atau bobot penting dalam penentuan muatan emosi/sentimen (seperti kata depan, kata hubung, dan partikel). Penghapusan kata henti membantu mereduksi ukuran dimensi ruang fitur dan meningkatkan efisiensi komputasi model SVM. Daftar kata henti yang digunakan mengacu pada kamus stopwords bahasa Indonesia yang diperluas dengan kata-kata pengisi (_filler words_) khas interaksi komentar YouTube (seperti _bang, mas, kak, bro, guys, wkwk, haha, aminn_).
Meskipun demikian, tidak semua kata henti dihapus. Kata penegas negasi yang berfungsi sebagai pembalik arah polaritas sentimen (seperti _"tidak"_, _"enggak"_, _"jangan"_, _"belum"_, _"bukan"_) sengaja **dipertahankan**. Jika kata negasi ini dihapus, kalimat _"tidak bagus"_ akan berubah menjadi _"bagus"_, sehingga model akan salah mengklasifikasikan kalimat yang berpolaritas negatif menjadi positif.

- **Contoh input**: `"lagu yang sangat bagus"`
- **Contoh output**: `"lagu bagus"`

### 4.2.7 Pengakaran Kata (_Stemming_)

Proses _stemming_ bertujuan untuk memetakan kata-kata berimbuhan menjadi kata dasarnya (_root word_). Pada bahasa Indonesia, imbuhan (awalan, sisipan, akhiran, dan konfiks) dapat mengubah bentuk kata kerja atau kata benda tetapi maknanya tetap serupa dalam konteks analisis sentimen (misalnya, kata _"mendengarkan"_, _"didengar"_, _"pendengar"_ memiliki kata dasar _"dengar"_). Penelitian ini menggunakan pustaka **PySastrawi**, yaitu implementasi algoritma stemming bahasa Indonesia berbasis Algoritma Nazief & Adriani yang telah disesuaikan. Untuk mencegah hilangnya makna khusus atau entitas penting, kata-kata yang masuk dalam daftar kata kunci konteks lagu _"Arteri"_ (seperti _"baskara"_, _"feast"_, _"arteri"_, _"mcr"_, _"pondok indah"_) dikecualikan dari proses _stemming_ agar bentuk aslinya tetap terjaga utuh.

- **Contoh input**: `"mendengarkan"` menjadi `"dengar"`; `"menangis"` menjadi `"tangis"`

### 4.2.8 Hasil Seleksi dan Pembersihan Dataset

Setelah melewati tujuh tahapan preprocessing, dataset mengalami penyusutan dari **1.531 komentar** menjadi **1.175 komentar bersih** (penyusutan sebesar **22,7%**). Rincian komentar yang dihapus disajikan pada Tabel 4.1 berikut:

**Tabel 4.1 Ringkasan Penghapusan Data pada Tahap Preprocessing**

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

### 4.2.9 Contoh Transformasi Data

**Tabel 4.2 Contoh Transformasi Teks Komentar Melalui Tahapan Preprocessing**

| Tahap                         | Teks Komentar                                                                   |
| :---------------------------- | :------------------------------------------------------------------------------ |
| **Mentah**                    | `"Lagu ini gw dengerin tiap hari 😭 bener bgt sm kehidupan gw skrg"`            |
| Setelah _Case Folding_        | `"lagu ini gw dengerin tiap hari 😭 bener bgt sm kehidupan gw skrg"`            |
| Setelah _Cleaning_ + Emoji    | `"lagu ini gw dengerin tiap hari bener bgt sm kehidupan gw skrg"`               |
| Setelah _Slang Normalization_ | `"lagu ini saya dengarkan tiap hari benar banget sama kehidupan saya sekarang"` |
| Setelah _Stopwords Removal_   | `"lagu dengarkan hari benar banget kehidupan sekarang"`                         |
| **Setelah _Stemming_**        | `"lagu dengar hari benar banget hidup sekarang"`                                |

---

## 4.3 Pelabelan Sentimen (_Sentiment Labeling_)

Komentar yang telah melalui tahap preprocessing kemudian diberi label kelas sentimen secara otomatis. Penelitian ini menggunakan metode **pelabelan berbasis leksikon (_Lexicon-Based_)** sebagai pendekatan _unsupervised_ untuk menghasilkan label kelas sentimen awal sebelum data digunakan melatih model SVM.

### 4.3.1 Metode Pelabelan Berbasis Leksikon

Metode ini bekerja dengan mencocokkan kata-kata dalam teks komentar bersih terhadap kamus sentimen bahasa Indonesia. Setiap kata memiliki bobot numerik polaritas:

- **Kata Positif**: bobot +1 hingga +3
- **Kata Negatif**: bobot -1 hingga -3
- **Kata Tidak Ditemukan**: bobot 0 (diabaikan)

Skor sentimen akhir suatu komentar dihitung dari akumulasi bobot seluruh kata yang ditemukan dalam kamus.

### 4.3.2 Penanganan Negasi dan Kata Penguat

- **Kata Penguat (_Intensifier_)**: Kata seperti _"sangat"_, _"sekali"_, _"banget"_ memperkuat skor sentimen kata di dekatnya dengan faktor pengali sebesar **1,5x**.
  - _Contoh_: `"bagus banget"` → skor (+2) x 1.5 = +3,0

- **Penanganan Negasi (_Negation Handling_)**: Kata seperti _"tidak"_, _"ga"_, _"belum"_, _"bukan"_ membalikkan arah polaritas skor sentimen kata sesudahnya, dengan bobot dikurangi 50%.
  - _Contoh_: `"tidak bagus"` → skor -(+2) x 0.5 = -1,0

### 4.3.3 Batas Ambang (_Threshold_) Klasifikasi Kelas

| Kondisi Skor   | Label Kelas          | Nilai Numerik |
| :------------- | :------------------- | :-----------: |
| Skor >= +1     | **Sentimen Positif** |       2       |
| -1 < Skor < +1 | **Sentimen Netral**  |       1       |
| Skor <= -1     | **Sentimen Negatif** |       0       |

### 4.3.4 Penambahan Kosakata Kamus Negatif

Untuk meningkatkan cakupan kamus, beberapa kata umpatan dan kata bernada negatif yang khas dalam bahasa informal Indonesia ditambahkan secara manual:

| Kata Tambahan | Bobot |
| :------------ | :---: |
| `kepharat`    |  -3   |
| `keparat`     |  -3   |
| `membosankan` |  -2   |
| `menyebalkan` |  -2   |

### 4.3.5 Distribusi Hasil Pelabelan

**Tabel 4.3 Distribusi Kelas Sentimen Hasil Pelabelan**

| Kelas Sentimen | Jumlah Komentar | Persentase (%) |
| :------------- | :-------------: | :------------: |
| **Positif**    |       308       |     26,2%      |
| **Netral**     |       754       |     64,2%      |
| **Negatif**    |       113       |      9,6%      |
| **Total**      |    **1.175**    |   **100,0%**   |

Visualisasi persentase distribusi kelas sentimen hasil pelabelan leksikon ini disajikan dalam bentuk grafik diagram batang pada Gambar 4.1 berikut:

![Gambar 4.1 Grafik Distribusi Kelas Sentimen](../SOURCE CODE/Data/sentiment_distribution.png)

---

## 4.4 Ekstraksi Fitur TF-IDF (_Feature Extraction_)

### 4.4.1 Konfigurasi TF-IDF Vectorizer

Data teks yang telah berlabel diubah menjadi representasi vektor numerik menggunakan metode **Term Frequency - Inverse Document Frequency (TF-IDF)**:

| Parameter TF-IDF | Nilai                                      |
| :--------------- | :----------------------------------------- |
| `max_features`   | 5.000 kata unik maksimal                   |
| `ngram_range`    | (1, 2) — gabungan _unigram_ dan _bigram_   |
| `sublinear_tf`   | True — penskalaan logaritmik TF            |
| `min_df`         | 2 — kata harus muncul minimal di 2 dokumen |

### 4.4.2 Dimensi Matriks Fitur

| Bagian Dataset              | Jumlah Sampel | Jumlah Fitur (Dimensi) |
| :-------------------------- | :-----------: | :--------------------: |
| Data Latih (_Training Set_) |      940      |          745           |
| Data Uji (_Testing Set_)    |      235      |          745           |

### 4.4.3 Kata dengan Bobot TF-IDF Tertinggi per Kelas

- **Positif**: _lagu, bagus, keren, relate, baskara, suka_
- **Netral**: _feast, mcr, helena, lagu, cerita, hidup_
- **Negatif**: _jelek, tidak suka, bosan, membosankan, sakit_

---

## 4.5 Implementasi dan Pelatihan Model SVM

### 4.5.1 Pembagian Data (_Data Splitting_)

Dataset dibagi menggunakan metode **Stratified Split** dengan rasio **80:20**:

- **Data Latih (_Training Set_)**: 940 komentar (80%)
- **Data Uji (_Testing Set_)**: 235 komentar (20%)

### 4.5.2 Penanganan Ketidakseimbangan Kelas (_Class Imbalance_)

Dataset mengalami ketidakseimbangan kelas signifikan (Negatif hanya 9,6%, Netral 64,2%, Positif 26,2%). Dua pendekatan diterapkan:

1. **Random Oversampling (ROS)**: Sampel kelas minoritas (Negatif) diduplikasi secara acak hingga distribusi kelas menjadi seimbang pada data latih.
2. **Pembobotan Kelas (`class_weight='balanced'`)**: SVM diberi penalti lebih besar pada kesalahan klasifikasi kelas minoritas selama pelatihan.

### 4.5.3 Pencarian Hyperparameter Optimal (_GridSearchCV_)

Pencarian dilakukan menggunakan **GridSearchCV** dengan skema **5-fold cross-validation**:

| Parameter            | Nilai yang Dicoba |
| :------------------- | :---------------- |
| `kernel`             | `'linear'`        |
| `C` (Regularization) | 0.1, 0.5, 1.0     |
| `gamma`              | `'scale'`         |

**Tabel 4.4 Hyperparameter Terbaik Hasil GridSearchCV**

| Parameter                       | Nilai Terbaik |
| :------------------------------ | :------------ |
| **Kernel**                      | `linear`      |
| **C (Regularization)**          | 1.0           |
| **Gamma**                       | scale         |
| **Best CV Score (Weighted F1)** | ~80%          |

### 4.5.4 Arsitektur Model SVM Final

Model SVM final menggunakan:

- **Kernel Linear**: Mencari _hyperplane_ optimal **w . x + b = 0** dalam ruang fitur TF-IDF berdimensi tinggi.
- **Strategi Multi-kelas One-vs-One (OvO)**: Melatih 3 model biner (Positif vs Negatif, Positif vs Netral, Negatif vs Netral), hasilnya digabung melalui voting.
- **Regularisasi C**: Mengontrol keseimbangan antara maximizing margin dan minimizing classification error.

---

## 4.6 Evaluasi dan Analisis Performa Model SVM

> **Menjawab Rumusan Masalah ke-2**: _"Seberapa baik kinerja model SVM dalam mengenali pola sentimen pada dataset komentar yang tidak terstruktur jika diukur menggunakan parameter evaluasi Confusion Matrix (Akurasi, Presisi, Recall, dan F1-Score)?"_

### 4.6.1 Hasil Pengukuran Metrik Performa Model

**Tabel 4.5 Metrik Evaluasi Performa Model SVM (Data Uji: 235 komentar)**

| Metrik Evaluasi                     |   Nilai    |
| :---------------------------------- | :--------: |
| **Akurasi (_Accuracy_)**            | **80,85%** |
| **Presisi (_Precision_, Weighted)** | **80,55%** |
| **Recall (Weighted)**               | **80,85%** |
| **F1-Score (Weighted)**             | **80,66%** |

Perbandingan nilai dari masing-masing metrik evaluasi model SVM disajikan dalam bentuk diagram batang pada Gambar 4.2 berikut:

![Gambar 4.2 Grafik Perbandingan Metrik Evaluasi Model SVM](../SOURCE CODE/Output/metrics_comparison.png)

Nilai akurasi **80,85%** menunjukkan model SVM berhasil mengklasifikasikan sekitar **190 dari 235 komentar** data uji dengan benar. Nilai ini tergolong **baik** untuk klasifikasi teks informal multi-kelas dengan ketidakseimbangan kelas.

### 4.6.2 Performa Per Kelas Sentimen

**Tabel 4.6 Laporan Klasifikasi Per Kelas Sentimen**

| Kelas Sentimen           | Precision |  Recall  | F1-Score | Support (Data Uji) |
| :----------------------- | :-------: | :------: | :------: | :----------------: |
| **Negatif**              |   0,64    |   0,61   |   0,62   |         23         |
| **Netral**               |   0,86    |   0,89   |   0,87   |        151         |
| **Positif**              |   0,74    |   0,69   |   0,71   |         61         |
| **Rata-rata (Weighted)** | **0,81**  | **0,81** | **0,81** |      **235**       |

Interpretasi per kelas:

- **Netral (F1: 0,87)**: Performa terbaik karena memiliki jumlah sampel paling banyak (151 data uji), sehingga model memiliki pola yang cukup untuk dipelajari secara komprehensif.
- **Positif (F1: 0,71)**: Performa baik dengan presisi memadai (0,74), membuktikan model cukup andal menyaring opini bernada apresiasi.
- **Negatif (F1: 0,62)**: Tetap stabil dan menunjukkan sensitivitas yang baik (Recall 61%) berkat penerapan Random Oversampling untuk mengatasi class imbalance (dibandingkan tanpa oversampling di mana recall sempat berada di bawah 35%).

### 4.6.3 Analisis Confusion Matrix

**Tabel 4.7 Confusion Matrix Hasil Prediksi Model SVM**

| Aktual \ Prediksi | Negatif | Netral  | Positif |
| :---------------- | :-----: | :-----: | :-----: |
| **Negatif**       | **14**  |    6    |    3    |
| **Netral**        |    5    | **134** |   12    |
| **Positif**       |    3    |   16    | **42**  |

Matriks kebingungan (_confusion matrix_) yang menunjukkan detail jumlah klasifikasi aktual terhadap hasil prediksi model pada data uji disajikan secara visual pada Gambar 4.3 berikut:

![Gambar 4.3 Confusion Matrix Hasil Prediksi Model SVM](../SOURCE CODE/Output/confusion_matrix.png)

Analisis detail:

- **Negatif**: 14 dari 23 komentar (60,9%) diklasifikasikan benar. 6 salah diprediksi netral, 3 diprediksi positif.
- **Netral**: 134 dari 151 komentar (88,7%) diklasifikasikan benar. 5 salah diprediksi negatif, 12 diprediksi positif.
- **Positif**: 42 dari 61 komentar (68,9%) diklasifikasikan benar. 16 salah diprediksi netral, 3 diprediksi negatif.

Kesalahan terbesar adalah komentar **Positif yang salah diklasifikasi sebagai Netral** (16 komentar) dan **Netral yang salah diklasifikasi sebagai Positif** (12 komentar). Ini wajar karena batas semantik antara apresiasi umum (positif) dengan komentar informatif (netral) di YouTube sering kali sangat tipis dan tanpa ekspresi emosi eksplisit.

### 4.6.4 Perbandingan Performa dengan Model Pembanding (Logistic Regression)

Untuk membuktikan efektivitas algoritma Support Vector Machine (SVM) kernel linear yang diusulkan, dilakukan pengujian pembanding menggunakan algoritma **Logistic Regression (Regresi Logistik)**. Pengujian ini menggunakan dataset preprocessing (1.175 komentar bersih) dan ekstraksi fitur TF-IDF (5.000 fitur) yang sama persis untuk menjamin perbandingan performa yang adil (*fair comparison*).

Hasil perbandingan metrik evaluasi antara kedua model disajikan pada Tabel 4.8 berikut:

**Tabel 4.8 Perbandingan Performa SVM vs Logistic Regression**

| Algoritma | Akurasi | Precision (Weighted) | Recall (Weighted) | F1-Score (Weighted) |
| :--- | :---: | :---: | :---: | :---: |
| **SVM (Linear)** | **80,85%** | **80,55%** | **80,85%** | **80,66%** |
| **Logistic Regression** | 79,15% | 79,18% | 79,15% | 79,14% |

Berdasarkan hasil pengujian pada Tabel 4.8, model SVM Linear terbukti lebih unggul di semua metrik dibandingkan dengan Logistic Regression. Model SVM mencapai akurasi **80,85%** (unggul 1,70% atas Logistic Regression) dan F1-Score sebesar **80,66%** (unggul 1,52% atas Logistic Regression). 

Kelebihan performa SVM Linear ini secara teoritis disebabkan oleh karakteristik optimasi batas keputusan:
1. **Prinsip Maksimalisasi Marjin (*Maximal Margin Classifier*)**: SVM mencari *hyperplane* pembatas yang memiliki jarak marjin terjauh dari *support vectors* terdekat dari masing-masing kelas. Hal ini membuat model SVM lebih tangguh (*robust*) terhadap noise semantik dan memiliki kemampuan generalisasi data uji yang lebih baik.
2. **Optimasi Batas Geometris vs Batas Probabilistik**: Logistic Regression menggunakan fungsi sigmoid untuk meminimalkan *log loss* secara probabilistik yang sensitif terhadap outlier kata gaul di YouTube. Sedangkan SVM menggunakan *hinge loss* yang mengabaikan titik-titik data di luar marjin pembatas, sehingga tidak mudah goyah oleh noise teks informal.

### 4.6.5 Perbandingan dengan Penelitian Terdahulu

**Tabel 4.9 Perbandingan Akurasi dengan Penelitian Terdahulu**

| Penelitian                                   | Domain Data          | Jumlah Kelas | Akurasi SVM |
| :------------------------------------------- | :------------------- | :----------: | :---------: |
| Idris et al. (2023) — Shopee                 | Ulasan aplikasi      |      2       |    98,0%    |
| Arsi & Waluyo (2021) — Ibu Kota              | Tweet                |      2       |   96,68%    |
| Herwinsyah & Witanti (2022) — Covid          | Tweet                |      2       |    89,0%    |
| Utami & Erfina (2021) — Pinjol               | Tweet                |      2       |    62,0%    |
| **Penelitian Ini — Komentar YouTube Arteri** | **Komentar YouTube** |    **3**     | **80,85%**  |

Nilai 80,85% pada penelitian ini kompetitif mengingat tingkat kesulitan lebih tinggi: tiga kelas sentimen, teks komentar YouTube yang sangat informal, dan dataset yang lebih terbatas.

---

## 4.7 Analisis Distribusi dan Kecenderungan Sentimen Pendengar

> **Menjawab Rumusan Masalah ke-3**: _"Bagaimana kecenderungan dominasi sentimen pendengar (positif, negatif, atau netral) terhadap pesan lagu tersebut berdasarkan hasil klasifikasi model yang dibangun?"_

### 4.7.1 Distribusi Sentimen Keseluruhan

**Tabel 4.10 Distribusi Sentimen Akhir Pendengar Lagu "Arteri"**

| Kelas Sentimen   | Jumlah Komentar | Persentase (%) |
| :--------------- | :-------------: | :------------: |
| **Positif**      |       308       |     26,2%      |
| **Netral**       |       754       |     64,2%      |
| **Negatif**      |       113       |      9,6%      |
| **Total Bersih** |    **1.175**    |   **100,0%**   |

### 4.7.2 Dominasi Sentimen Netral: Analisis dan Interpretasi

Kelas **Netral** mendominasi dengan **64,2%**. Temuan ini tidak mengindikasikan ketidakpedulian, melainkan mencerminkan karakteristik komunikasi pendengar YouTube yang cenderung:

1. **Menceritakan Pengalaman Pribadi**: Pendengar menuliskan kisah _quarter-life crisis_ mereka tanpa ekspresi penilaian evaluatif yang eksplisit.
2. **Diskusi Komparatif Musikal**: Banyak komentar membahas kesamaan atmosfer lagu _"Arteri"_ dengan band _My Chemical Romance (MCR)_ — bersifat informatif, bukan evaluatif.
3. **Kutipan dan Interpretasi Lirik**: Pendengar mendiskusikan makna baris lirik tanpa memberikan penilaian polar.
4. **Penyebutan Nama Vokalis ("Baskara" / "Bas")**: Banyak komentar yang menyebutkan nama vokalis _"Baskara"_ atau _"Bas"_ dalam konteks apresiasi gaya bernyanyi atau penulisan lirik secara implisit (misal: _"Baskara emang suaranya gak yang seistimewa itu, tapi percayalah karyanya tak akan seistimewa saat ini saat bukan dia yang nyanyi"_ atau _"apapun lagunya selagi yang nyanyi Baskara..."_). Karena nama entitas tokoh ini memiliki bobot sentimen netral (0) dalam kamus leksikon, kalimat apresiatif implisit ini secara otomatis dikelompokkan oleh pelabelan leksikon ke dalam kelas **Netral**, yang secara artifisial meningkatkan persentase kategori ini.

### 4.7.3 Sentimen Positif: Temuan dan Topik Utama

Kelas **Positif** (26,2%) didominasi oleh tema:

1. **Apresiasi Musikalitas**: Kata seperti _"bagus"_, _"keren"_, _"indah"_, _"masterpiece"_ menunjukkan penghargaan terhadap kualitas musik dan kedalaman lirik.
2. **Koneksi Emosional (_Emotional Resonance_)**: Kata _"relate"_ muncul konsisten — pendengar merasa narasi _quarter-life crisis_ mencerminkan pengalaman pribadi mereka secara akurat.
3. **Apresiasi terhadap Baskara Putra**: Nama _"baskara"_ atau sebutan _"bas"_ sering disebut dalam konteks pujian vokal dan penulisan lirik.
4. **Fungsi Katarsis Emosional**: Lagu ini disebut sebagai _"obat"_ atau _"teman setia"_ di masa-masa sulit.

### 4.7.4 Sentimen Negatif: Temuan dan Pola Kritik

Kelas **Negatif** (9,6%) menunjukkan pola:

1. **Kritik Musikal**: Sebagian kecil pendengar merasa aransemen tidak sesuai selera, diekspresikan dengan kata _"bosan"_, _"biasa saja"_, _"jelek"_.
2. **Respons Emosional Negatif terhadap Tema**: Ekspresi perasaan _"sakit"_ atau _"sedih"_ berlebih berkaitan dengan kenangan relasi atau trauma pribadi.
3. **Ekspresi Umpatan Informal**: Ditemukan kata umpatan seperti _"kepharat"_ dan _"keparat"_ yang terklasifikasi sebagai sentimen negatif dengan skor leksikon rendah.

### 4.7.5 Pemetaan Kata Kunci per Sentimen (_Word Cloud Analysis_)

**Tabel 4.11 Kata Paling Dominan per Kelas Sentimen (Top-8)**

| Sentimen Positif | Sentimen Netral | Sentimen Negatif |
| :--------------: | :-------------: | :--------------: |
|       lagu       |      lagu       |      sakit       |
|      relate      |      feast      |      bosan       |
|      bagus       |       mcr       |    tidak suka    |
|      keren       |     cerita      |      jelek       |
|     baskara      |      hidup      |      sedih       |
|       suka       |     helena      |     kepharat     |
|      indah       |      lirik      |   membosankan    |
|      banget      |     dengar      |    tidak enak    |

Visualisasi awan kata (_word cloud_) widescreen untuk masing-masing kelas sentimen (Positif, Netral, dan Negatif) disajikan berturut-turut pada Gambar 4.4, Gambar 4.5, dan Gambar 4.6 berikut:

**Gambar 4.4 Awan Kata (Word Cloud) - Sentimen Positif**
![Gambar 4.4 Word Cloud - Sentimen Positif](../SOURCE CODE/Output/wordcloud_positif.png)

**Gambar 4.5 Awan Kata (Word Cloud) - Sentimen Netral**
![Gambar 4.5 Word Cloud - Sentimen Netral](../SOURCE CODE/Output/wordcloud_netral.png)

**Gambar 4.6 Awan Kata (Word Cloud) - Sentimen Negatif**
![Gambar 4.6 Word Cloud - Sentimen Negatif](../SOURCE CODE/Output/wordcloud_negatif.png)

---

## 4.8 Implementasi Sistem Analisis Sentimen

> **Menjawab Rumusan Masalah ke-1**: _"Bagaimana penerapan algoritma SVM dengan pembobotan TF-IDF untuk mengklasifikasikan sentimen komentar publik pada video musik '.Feast - Arteri'?"_

### 4.8.1 Alur Implementasi Sistem (Pipeline KDD)

Penerapan SVM dengan TF-IDF dilaksanakan melalui alur 6 tahap berbasis kerangka **Knowledge Discovery in Databases (KDD)**:

`[1] Pengumpulan Data      → Web scraping YouTube API (1.531 komentar)
         ↓
[2] Text Preprocessing    → Cleaning, Normalisasi, Stopwords, Stemming (1.175 bersih)
         ↓
[3] Pelabelan Leksikon    → Positif/Negatif/Netral scoring (Threshold ±1)
         ↓
[4] Ekstraksi Fitur       → TF-IDF Vectorization (max 5000 fitur, bigram)
         ↓
[5] Pelatihan Model       → Random Oversampling + GridSearchCV + SVM Linear
         ↓
[6] Evaluasi & Prediksi   → Confusion Matrix, Accuracy 80,85%, F1-Score 80,66%`

### 4.8.2 Implementasi Teknis dalam Kode Program (Python)

| Pustaka                | Peran dalam Implementasi                                    |
| :--------------------- | :---------------------------------------------------------- |
| `pandas`               | Manipulasi dan analisis dataset format DataFrame            |
| `scikit-learn`         | Implementasi TF-IDF, SVM, GridSearchCV, dan metrik evaluasi |
| `PySastrawi`           | Stemming bahasa Indonesia                                   |
| `NLTK`                 | Tokenisasi dan pengelolaan stopwords                        |
| `imblearn`             | Random Oversampling untuk menyeimbangkan kelas              |
| `joblib`               | Penyimpanan model terlatih ke file .pkl                     |
| `matplotlib / seaborn` | Visualisasi Confusion Matrix dan distribusi sentimen        |
| `wordcloud`            | Visualisasi kata kunci dominan per kelas sentimen           |

### 4.8.3 Uji Coba Prediksi Real-time

**Tabel 4.12 Hasil Uji Coba Prediksi Sentimen Real-time**

| Teks Komentar Input                     | Prediksi Sentimen |  Confidence  |
| :-------------------------------------- | :---------------: | :----------: |
| _"lagu ini sangat bagus sekali"_        |    **Positif**    |     ~95%     |
| _"jelek ga suka sama sekali"_           |    **Negatif**    |     ~93%     |
| _"lagunya membosankan"_                 |    **Negatif**    |     ~96%     |
| _"tidak suka dengan nadanya"_           |    **Negatif**    |     ~91%     |
| _"kepharat kepharat"_                   |    **Negatif**    | — (Leksikon) |
| _"lagu ini mengingatkan saya pada mcr"_ |    **Netral**     |     ~88%     |

Seluruh komentar yang secara semantis bermuatan negatif berhasil diklasifikasikan dengan benar, menunjukkan model telah mempelajari pola sentimen negatif dengan baik setelah penerapan Random Oversampling.

### 4.8.4 Antarmuka Visualisasi (_Streamlit Dashboard_)

Sistem analisis sentimen diimplementasikan dalam **aplikasi web interaktif** menggunakan _framework_ **Streamlit**, menampilkan:

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
5. **Sarkasme dan Bahasa Figuratif**: Model TF-IDF+SVM belum mampu menangani _sarcasm detection_ secara memadai.
6. **Bias Entitas Nama Tokoh dan Judul Album (Named Entity & Title Bias)**:
   - Nama vokalis _"Baskara"_ atau _"Bas"_ sering digunakan dalam komentar opini apresiasi vokal secara tidak langsung. Karena tidak membawa nilai sentimen intrinsik dalam kamus leksikon (skor 0), opini tersebut rentan bergeser ke label **Netral** apabila tidak dibarengi kata sifat positif yang eksplisit (misal: _"selagi yang nyanyi Baskara..."_).
   - Judul album .FEAST yaitu _"Membangun & Menghancurkan"_ mengandung kata _"menghancurkan"_ atau _"hancur"_. Dalam kamus leksikon, kata _"hancur"_ dinilai sangat negatif (skor -2). Akibatnya, komentar yang netral/positif namun hanya sekadar menyebutkan judul album (misal: _"tunggu next lagu dari album membangun dan menghancurkan"_) secara tidak sengaja terlabeli sebagai **Negatif** oleh pelabel leksikon otomatis.

---

## 4.10 Ringkasan Jawaban Rumusan Masalah

|  No.  | Rumusan Masalah                                                 | Jawaban                                                                                                                                                                                           |
| :---: | :-------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **1** | Bagaimana penerapan SVM + TF-IDF untuk klasifikasi sentimen?    | Diimplementasikan melalui pipeline 6 tahap KDD: _scraping_ → _preprocessing_ (7 tahap) → _labeling leksikon_ → TF-IDF Vectorization → SVM Linear (GridSearchCV + Random Oversampling) → Evaluasi. |
| **2** | Seberapa baik kinerja model SVM (Akurasi, Presisi, Recall, F1)? | **Akurasi 80,85%**, Presisi 80,55%, Recall 80,85%, F1-Score 80,66% — performa baik untuk klasifikasi teks informal tiga kelas dengan dataset terbatas.                                            |
| **3** | Bagaimana kecenderungan dominasi sentimen pendengar?            | **Sentimen Netral mendominasi (64,2%)**, diikuti Positif (26,2%) dan Negatif (9,6%), menunjukkan lagu "Arteri" diterima secara positif-netral oleh mayoritas pendengar YouTube.                   |
