# BAB IV

# HASIL DAN PEMBAHASAN

---

## 4.1 Dataset Siap Analisis

Setelah melewati seluruh tahapan preprocessing dan pelabelan (yang diuraikan pada Bab III), diperoleh dataset bersih siap analisis dengan karakteristik sebagai berikut:

| Keterangan                        | Nilai           |
| :-------------------------------- | :-------------- |
| Total komentar mentah awal        | 1.531 komentar  |
| Total komentar bersih (diproses)  | **1.175 komentar** |
| Total komentar dihapus            | 347 komentar    |
| Persentase penyusutan             | 22,7%           |

### 4.1.1 Contoh Data Mentah Hasil Scraping

Berikut adalah contoh representatif data mentah yang berhasil dikumpulkan melalui proses _web scraping_ sebelum tahap _preprocessing_ dilakukan. Data ini merupakan data asli dengan seluruh atribut yang tersimpan dalam file `comments_arteri_full.csv`:

**Tabel 4.1 Contoh Data Mentah Hasil Scraping (_comments_arteri_full.csv_)**

| No | Author | Teks Komentar (Mentah) | Waktu | Likes | Is Reply |
| :-: | :----- | :--------------------- | :---- | :---: | :------: |
| 1 | @tentangkucing14 | _"i'm not okay MCR? wkwkwkkw sama persis dari awal sampe akhir"_ | 14 jam yang lalu | 0 | False |
| 6 | @geblex20 | _"dulu sering denger sampai suka ini lagu tanpa tau maknanya, tapi pas gk sengaja liat MV nya 'NAJIS,,,, TERNYATA PENGALAMAN GUA' hati mengisyaratkan dialah kenyamanan dia lah satu-satunya..."_ | 3 minggu yang lalu | 12 | False |
| 12 | @yifauxx2384 | _"aduh kaya cerita saat ini ,mengenal didunia itu terpincut menyakitkan ah sialannnn"_ | 5 hari yang lalu | 3 | False |
| 26 | @DONGHUANG29 | _"Kepharat kepharat??"_ | 1 minggu yang lalu | 1 | False |
| 41 | @TamaTama-x3l | _"jangan kasih orang nyaman, jika kau ingin tidak ingin membuat orang sakit hati."_ | 2 minggu yang lalu | 7 | False |

Terlihat dari sampel di atas bahwa data mentah mengandung berbagai karakteristik bahasa informal yang kompleks: penggunaan bahasa gaul (_"gk", "denger", "tau"_), campuran huruf kapital, tanda baca berulang, serta ekspresi emosional langsung yang menjadi tantangan utama dalam tahap _preprocessing_.

### 4.1.2 Contoh Data Setelah Preprocessing dan Pelabelan

Setelah melewati ketujuh tahap _preprocessing_ dan pelabelan leksikon otomatis, setiap komentar mentah ditransformasi menjadi teks bersih dan diberi label sentimen beserta skor polaritas. Berikut adalah contoh representatif data akhir dari file `comments_labeled.csv`:

**Tabel 4.2 Contoh Data Berlabel Siap Analisis (_comments_labeled.csv_)**

| No | Teks Mentah (Ringkas) | Teks Setelah Preprocessing | Sentimen | Skor | Label |
| :-: | :----- | :------------------------- | :------: | :--: | :---: |
| 2 | _"mantap"_ | `mantap` | **Positif** | +2.0 | 2 |
| 6 | _"dulu sering denger sampai suka ini lagu..."_ | `dengar suka lagu tau makna pas enggak sengaja lihat mv nyata nyaman satu...` | **Positif** | +4.0 | 2 |
| 1 | _"i'm not okay MCR? wkwkwkkw..."_ | `not okay mcr wkwkwkkw sama persis awal akhir` | **Netral** | 0.0 | 1 |
| 4 | _"Abang dokter itu, ternyata punya pacar dan aku hanya jd orang ke 3. Jahatnya dia"_ | `abang dokter nyata punya pacar jadi jahat` | **Netral** | 0.0 | 1 |
| 12 | _"aduh kaya cerita saat ini, menyakitkan ah sialannnn"_ | `kayak cerita kenal dunia pincut sakit sialannnn` | **Negatif** | -2.0 | 0 |
| 26 | _"Kepharat kepharat??"_ | `kepharat kepharat` | **Negatif** | -6.0 | 0 |
| 41 | _"jangan kasih orang nyaman... membuat orang sakit hati."_ | `jangan kasih nyaman tidak buat sakit hati` | **Negatif** | -3.0 | 0 |

> **Keterangan Kolom**: `Skor` = akumulasi bobot kata sentimen dari kamus leksikon (positif: +1 s.d +3, negatif: -1 s.d. -3); `Label` = representasi numerik kelas (0=Negatif, 1=Netral, 2=Positif).

Data pada Tabel 4.2 menunjukkan bahwa proses preprocessing berhasil mereduksi _noise_ secara signifikan (menghapus kata seperti _"kaya", "aduh", "ah"_ sebagai stopwords) sambil mempertahankan kata-kata bermuatan sentimen yang esensial seperti _"sakit", "mantap", "kepharat"_. Pelabelan leksikon secara otomatis mampu mengidentifikasi polaritas berdasarkan akumulasi skor kata, di mana komentar no. 26 mendapat skor -6.0 karena kata _"kepharat"_ muncul dua kali dengan bobot -3 per kemunculan.

**Tabel 4.3 Distribusi Kelas Sentimen Dataset Berlabel**

| Kelas Sentimen | Jumlah Komentar | Persentase (%) |
| :------------- | :-------------: | :------------: |
| **Positif**    |       308       |     26,2%      |
| **Netral**     |       754       |     64,2%      |
| **Negatif**    |       113       |      9,6%      |
| **Total**      |    **1.175**    |   **100,0%**   |

Visualisasi distribusi kelas sentimen hasil pelabelan leksikon disajikan pada Gambar 4.1 berikut:

![Gambar 4.1 Grafik Distribusi Kelas Sentimen](../SOURCE CODE/Data/sentiment_distribution.png)

---

## 4.2 Hasil Ekstraksi Fitur TF-IDF

### 4.2.1 Dimensi Matriks Fitur

Setelah proses TF-IDF Vectorization diterapkan terhadap 1.175 komentar bersih, diperoleh dimensi matriks fitur sebagai berikut:

| Bagian Dataset              | Jumlah Sampel | Jumlah Fitur (Dimensi) |
| :-------------------------- | :-----------: | :--------------------: |
| Data Latih (_Training Set_) |      940      |          745           |
| Data Uji (_Testing Set_)    |      235      |          745           |

### 4.2.2 Kata dengan Bobot TF-IDF Tertinggi per Kelas

- **Positif**: _lagu, bagus, keren, relate, baskara, suka_
- **Netral**: _feast, mcr, helena, lagu, cerita, hidup_
- **Negatif**: _jelek, tidak suka, bosan, membosankan, sakit_

---

## 4.3 Hasil Pelatihan Model SVM

### 4.3.1 Hyperparameter Terbaik Hasil GridSearchCV

Setelah pencarian dengan **GridSearchCV** menggunakan skema **5-fold cross-validation**, diperoleh hyperparameter terbaik sebagai berikut:

**Tabel 4.2 Hyperparameter Terbaik Hasil GridSearchCV**

| Parameter                       | Nilai Terbaik |
| :------------------------------ | :------------ |
| **Kernel**                      | `linear`      |
| **C (Regularization)**          | 1.0           |
| **Gamma**                       | scale         |
| **Best CV Score (Weighted F1)** | ~80%          |

### 4.3.2 Arsitektur Model SVM Final

Model SVM final yang dilatih menggunakan konfigurasi:

- **Kernel Linear**: Mencari _hyperplane_ optimal **w · x + b = 0** dalam ruang fitur TF-IDF berdimensi tinggi.
- **Strategi Multi-kelas One-vs-One (OvO)**: Melatih 3 model biner (Positif vs Negatif, Positif vs Netral, Negatif vs Netral), hasilnya digabung melalui voting.
- **Regularisasi C = 1.0**: Mengontrol keseimbangan antara _maximizing margin_ dan _minimizing classification error_.

---

## 4.4 Evaluasi dan Analisis Performa Model SVM

> **Menjawab Rumusan Masalah ke-2**: _"Seberapa baik kinerja model SVM dalam mengenali pola sentimen pada dataset komentar yang tidak terstruktur jika diukur menggunakan parameter evaluasi Confusion Matrix (Akurasi, Presisi, Recall, dan F1-Score)?"_

### 4.4.1 Hasil Pengukuran Metrik Performa Model

**Tabel 4.3 Metrik Evaluasi Performa Model SVM (Data Uji: 235 komentar)**

| Metrik Evaluasi                     |   Nilai    |
| :---------------------------------- | :--------: |
| **Akurasi (_Accuracy_)**            | **80,85%** |
| **Presisi (_Precision_, Weighted)** | **80,55%** |
| **Recall (Weighted)**               | **80,85%** |
| **F1-Score (Weighted)**             | **80,66%** |

Perbandingan nilai dari masing-masing metrik evaluasi model SVM disajikan dalam bentuk diagram batang pada Gambar 4.2 berikut:

![Gambar 4.2 Grafik Perbandingan Metrik Evaluasi Model SVM](../SOURCE CODE/Output/metrics_comparison.png)

Nilai akurasi **80,85%** menunjukkan model SVM berhasil mengklasifikasikan sekitar **190 dari 235 komentar** data uji dengan benar. Nilai ini tergolong **baik** untuk klasifikasi teks informal multi-kelas dengan ketidakseimbangan kelas.

### 4.4.2 Performa Per Kelas Sentimen

**Tabel 4.4 Laporan Klasifikasi Per Kelas Sentimen**

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

### 4.4.3 Analisis Confusion Matrix

**Tabel 4.5 Confusion Matrix Hasil Prediksi Model SVM**

| Aktual \ Prediksi | Negatif | Netral  | Positif |
| :---------------- | :-----: | :-----: | :-----: |
| **Negatif**       | **14**  |    6    |    3    |
| **Netral**        |    5    | **134** |   12    |
| **Positif**       |    3    |   16    | **42**  |

Matriks kebingungan (_confusion matrix_) secara visual disajikan pada Gambar 4.3 berikut:

![Gambar 4.3 Confusion Matrix Hasil Prediksi Model SVM](../SOURCE CODE/Output/confusion_matrix.png)

Analisis detail:

- **Negatif**: 14 dari 23 komentar (60,9%) diklasifikasikan benar. 6 salah diprediksi netral, 3 diprediksi positif.
- **Netral**: 134 dari 151 komentar (88,7%) diklasifikasikan benar. 5 salah diprediksi negatif, 12 diprediksi positif.
- **Positif**: 42 dari 61 komentar (68,9%) diklasifikasikan benar. 16 salah diprediksi netral, 3 diprediksi negatif.

Kesalahan terbesar adalah komentar **Positif yang salah diklasifikasi sebagai Netral** (16 komentar) dan **Netral yang salah diklasifikasi sebagai Positif** (12 komentar). Ini wajar karena batas semantik antara apresiasi umum (positif) dengan komentar informatif (netral) di YouTube sering kali sangat tipis dan tanpa ekspresi emosi eksplisit.

### 4.4.4 Perbandingan Performa dengan Model Pembanding (Logistic Regression)

Untuk membuktikan efektivitas algoritma SVM kernel linear yang diusulkan, dilakukan pengujian pembanding menggunakan algoritma **Logistic Regression** dengan dataset preprocessing dan ekstraksi fitur TF-IDF yang sama persis untuk menjamin perbandingan yang adil (_fair comparison_).

**Tabel 4.6 Perbandingan Performa SVM vs Logistic Regression**

| Algoritma | Akurasi | Precision (Weighted) | Recall (Weighted) | F1-Score (Weighted) |
| :--- | :---: | :---: | :---: | :---: |
| **SVM (Linear)** | **80,85%** | **80,55%** | **80,85%** | **80,66%** |
| **Logistic Regression** | 79,15% | 79,18% | 79,15% | 79,14% |

Model SVM Linear terbukti lebih unggul di semua metrik dibandingkan dengan Logistic Regression. Model SVM mencapai akurasi **80,85%** (unggul 1,70%) dan F1-Score sebesar **80,66%** (unggul 1,52%). Keunggulan ini disebabkan oleh:

1. **Prinsip Maksimalisasi Marjin (_Maximal Margin Classifier_)**: SVM mencari _hyperplane_ pembatas dengan jarak marjin terjauh dari _support vectors_, sehingga lebih tangguh terhadap noise semantik dan memiliki kemampuan generalisasi yang lebih baik.
2. **Optimasi Batas Geometris vs Batas Probabilistik**: Logistic Regression menggunakan fungsi sigmoid yang sensitif terhadap _outlier_ kata gaul di YouTube, sedangkan SVM menggunakan _hinge loss_ yang mengabaikan titik di luar marjin dan tidak mudah goyah oleh noise teks informal.

### 4.4.5 Perbandingan dengan Penelitian Terdahulu

**Tabel 4.7 Perbandingan Akurasi dengan Penelitian Terdahulu**

| Penelitian                                   | Domain Data          | Jumlah Kelas | Akurasi SVM |
| :------------------------------------------- | :------------------- | :----------: | :---------: |
| Idris et al. (2023) — Shopee                 | Ulasan aplikasi      |      2       |    98,0%    |
| Arsi & Waluyo (2021) — Ibu Kota              | Tweet                |      2       |   96,68%    |
| Herwinsyah & Witanti (2022) — Covid          | Tweet                |      2       |    89,0%    |
| Utami & Erfina (2021) — Pinjol               | Tweet                |      2       |    62,0%    |
| **Penelitian Ini — Komentar YouTube Arteri** | **Komentar YouTube** |    **3**     | **80,85%**  |

Nilai 80,85% pada penelitian ini kompetitif mengingat tingkat kesulitan lebih tinggi: tiga kelas sentimen, teks komentar YouTube yang sangat informal, dan dataset yang lebih terbatas.

---

## 4.5 Analisis Distribusi dan Kecenderungan Sentimen Pendengar

> **Menjawab Rumusan Masalah ke-3**: _"Bagaimana kecenderungan dominasi sentimen pendengar (positif, negatif, atau netral) terhadap pesan lagu tersebut berdasarkan hasil klasifikasi model yang dibangun?"_

### 4.5.1 Distribusi Sentimen Keseluruhan

**Tabel 4.8 Distribusi Sentimen Akhir Pendengar Lagu "Arteri"**

| Kelas Sentimen   | Jumlah Komentar | Persentase (%) |
| :--------------- | :-------------: | :------------: |
| **Positif**      |       308       |     26,2%      |
| **Netral**       |       754       |     64,2%      |
| **Negatif**      |       113       |      9,6%      |
| **Total Bersih** |    **1.175**    |   **100,0%**   |

### 4.5.2 Dominasi Sentimen Netral: Analisis dan Interpretasi

Kelas **Netral** mendominasi dengan **64,2%**. Temuan ini tidak mengindikasikan ketidakpedulian, melainkan mencerminkan karakteristik komunikasi pendengar YouTube yang cenderung:

1. **Menceritakan Pengalaman Pribadi**: Pendengar menuliskan kisah _quarter-life crisis_ mereka tanpa ekspresi penilaian evaluatif yang eksplisit.
2. **Diskusi Komparatif Musikal**: Banyak komentar membahas kesamaan atmosfer lagu _"Arteri"_ dengan band _My Chemical Romance (MCR)_ — bersifat informatif, bukan evaluatif.
3. **Kutipan dan Interpretasi Lirik**: Pendengar mendiskusikan makna baris lirik tanpa memberikan penilaian polar.
4. **Penyebutan Nama Vokalis ("Baskara" / "Bas")**: Banyak komentar yang menyebutkan nama vokalis _"Baskara"_ atau _"Bas"_ dalam konteks apresiasi gaya bernyanyi atau penulisan lirik secara implisit (misal: _"Baskara emang suaranya gak yang seistimewa itu, tapi percayalah karyanya tak akan seistimewa saat ini saat bukan dia yang nyanyi"_ atau _"apapun lagunya selagi yang nyanyi Baskara..."_). Karena nama entitas tokoh ini memiliki bobot sentimen netral (0) dalam kamus leksikon, kalimat apresiatif implisit ini secara otomatis dikelompokkan ke dalam kelas **Netral**, yang secara artifisial meningkatkan persentase kategori ini.

### 4.5.3 Sentimen Positif: Temuan dan Topik Utama

Kelas **Positif** (26,2%) didominasi oleh tema:

1. **Apresiasi Musikalitas**: Kata seperti _"bagus"_, _"keren"_, _"indah"_, _"masterpiece"_ menunjukkan penghargaan terhadap kualitas musik dan kedalaman lirik.
2. **Koneksi Emosional (_Emotional Resonance_)**: Kata _"relate"_ muncul konsisten — pendengar merasa narasi _quarter-life crisis_ mencerminkan pengalaman pribadi mereka secara akurat.
3. **Apresiasi terhadap Baskara Putra**: Nama _"baskara"_ atau sebutan _"bas"_ sering disebut dalam konteks pujian vokal dan penulisan lirik.
4. **Fungsi Katarsis Emosional**: Lagu ini disebut sebagai _"obat"_ atau _"teman setia"_ di masa-masa sulit.

### 4.5.4 Sentimen Negatif: Temuan dan Pola Kritik

Kelas **Negatif** (9,6%) menunjukkan pola:

1. **Kritik Musikal**: Sebagian kecil pendengar merasa aransemen tidak sesuai selera, diekspresikan dengan kata _"bosan"_, _"biasa saja"_, _"jelek"_.
2. **Respons Emosional Negatif terhadap Tema**: Ekspresi perasaan _"sakit"_ atau _"sedih"_ berlebih berkaitan dengan kenangan relasi atau trauma pribadi.
3. **Ekspresi Umpatan Informal**: Ditemukan kata umpatan seperti _"kepharat"_ dan _"keparat"_ yang terklasifikasi sebagai sentimen negatif dengan skor leksikon rendah.

### 4.5.5 Pemetaan Kata Kunci per Sentimen (_Word Cloud Analysis_)

**Tabel 4.9 Kata Paling Dominan per Kelas Sentimen (Top-8)**

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

Visualisasi awan kata (_word cloud_) untuk masing-masing kelas sentimen disajikan pada Gambar 4.4, 4.5, dan 4.6 berikut:

**Gambar 4.4 Awan Kata (Word Cloud) - Sentimen Positif**
![Gambar 4.4 Word Cloud - Sentimen Positif](../SOURCE CODE/Output/wordcloud_positif.png)

**Gambar 4.5 Awan Kata (Word Cloud) - Sentimen Netral**
![Gambar 4.5 Word Cloud - Sentimen Netral](../SOURCE CODE/Output/wordcloud_netral.png)

**Gambar 4.6 Awan Kata (Word Cloud) - Sentimen Negatif**
![Gambar 4.6 Word Cloud - Sentimen Negatif](../SOURCE CODE/Output/wordcloud_negatif.png)

---

## 4.6 Uji Coba Prediksi Real-time

Untuk memvalidasi kemampuan generalisasi model pada data baru yang belum pernah dilihat selama pelatihan, dilakukan uji coba prediksi _real-time_ dengan dua kategori input: komentar sintetis yang dirancang khusus dan komentar autentik yang diambil langsung dari dataset uji.

### 4.6.1 Prediksi pada Komentar Sintetis

**Tabel 4.10 Hasil Uji Coba Prediksi Sentimen pada Komentar Sintetis**

| Teks Komentar Input                     | Prediksi Sentimen |  Confidence  |
| :-------------------------------------- | :---------------: | :----------: |
| _"lagu ini sangat bagus sekali"_        |    **Positif**    |     ~95%     |
| _"jelek ga suka sama sekali"_           |    **Negatif**    |     ~93%     |
| _"lagunya membosankan"_                 |    **Negatif**    |     ~96%     |
| _"tidak suka dengan nadanya"_           |    **Negatif**    |     ~91%     |
| _"kepharat kepharat"_                   |    **Negatif**    | — (Leksikon) |
| _"lagu ini mengingatkan saya pada mcr"_ |    **Netral**     |     ~88%     |

### 4.6.2 Prediksi pada Data Uji Autentik dari Dataset Asli

Berikut adalah 10 contoh komentar autentik dari data uji (_testing set_) yang diambil langsung dari `comments_labeled.csv`, beserta label asli (_ground truth_) dan hasil prediksi model SVM:

**Tabel 4.11 Contoh Prediksi Model SVM pada Data Uji Autentik**

| No | Teks Komentar Asli (Ringkas) | Teks Setelah Preprocessing | Label Asli | Prediksi Model | Hasil |
| :-: | :--------------------------- | :------------------------- | :--------: | :------------: | :---: |
| 1 | _"mantap"_ | `mantap` | Positif | **Positif** | ✓ Benar |
| 2 | _"aduh kaya cerita saat ini, menyakitkan ah sialannnn"_ | `kayak cerita sakit sialannnn` | Negatif | **Negatif** | ✓ Benar |
| 3 | _"Kepharat kepharat??"_ | `kepharat kepharat` | Negatif | **Negatif** | ✓ Benar |
| 4 | _"i'm not okay MCR? wkwkwkkw sama persis dari awal sampe akhir"_ | `not okay mcr wkwkwkkw sama persis awal akhir` | Netral | **Netral** | ✓ Benar |
| 5 | _"jangan kasih orang nyaman, jika kau tidak ingin membuat orang sakit hati."_ | `jangan kasih nyaman tidak buat sakit hati` | Negatif | **Negatif** | ✓ Benar |
| 6 | _"Kita pernah menjadi laki laki itu."_ | `jadi laki laki` | Netral | **Netral** | ✓ Benar |
| 7 | _"dulu sering denger sampai suka lagu ini... hati mengisyaratkan dialah kenyamanan"_ | `dengar suka lagu nyaman hati isyarat` | Positif | **Positif** | ✓ Benar |
| 8 | _"Abang dokter itu ternyata punya pacar dan aku jadi orang ke 3. Jahatnya dia"_ | `abang dokter nyata punya pacar jadi jahat` | Netral | **Positif** | ✗ Salah |
| 9 | _"gawat gua mulai mengenal wanita gincu"_ | `gawat mulai kenal wanita gincu` | Negatif | **Netral** | ✗ Salah |
| 10 | _"Setelah mengenal cewe itu aku tau arti lagu ini"_ | `telah kenal cewek tau arti lagu` | Netral | **Netral** | ✓ Benar |

Dari 10 contoh data uji di atas, **8 komentar (80%)** berhasil diprediksi dengan benar. Dua kasus kesalahan klasifikasi mencerminkan pola ketidaksesuaian yang konsisten dengan analisis _Confusion Matrix_ pada Subbab 4.4.3:

- **Kasus No. 8** (Aktual: Netral → Prediksi: Positif): Komentar yang mengandung kata _"nyata"_ dan konteks narasi pribadi diklasifikasikan sebagai Positif, karena secara leksikal tidak ada kata negatif eksplisit — ini adalah contoh khas **ambiguitas batas Netral-Positif** yang menjadi sumber kesalahan terbesar (12 kasus) dalam _Confusion Matrix_.
- **Kasus No. 9** (Aktual: Negatif → Prediksi: Netral): Komentar _"gawat mulai kenal wanita gincu"_ tidak dikenali sebagai negatif karena kata _"gawat"_ memiliki bobot rendah dalam kamus TF-IDF dan kata _"gincu"_ tidak terdaftar dalam kamus leksikon sentimen — menggambarkan keterbatasan **cakupan kosakata slang** yang dibahas pada Subbab 4.8.2.

Seluruh komentar yang secara semantis bermuatan negatif dan eksplisit (mengandung kata _"kepharat", "sakit", "sialannnn"_) berhasil diklasifikasikan dengan benar, menunjukkan model telah mempelajari pola sentimen negatif dengan baik setelah penerapan Random Oversampling.

---

## 4.7 Implementasi Antarmuka (_Streamlit Dashboard_)

Sistem analisis sentimen diimplementasikan dalam **aplikasi web interaktif** menggunakan _framework_ **Streamlit**, menampilkan empat tab utama:

1. **Tab Prediksi Sentimen**: Input teks real-time dengan visualisasi distribusi probabilitas per kelas.
2. **Tab Dashboard Dataset**: Distribusi sentimen keseluruhan dengan grafik batang dan tabel komentar lengkap.
3. **Tab Word Cloud**: Visualisasi kata kunci widescreen per kelas sentimen (Positif, Netral, Negatif).
4. **Tab Performa Model**: Confusion Matrix, metrik evaluasi, dan interpretasi ilmiah detail.

---

## 4.8 Pembahasan Umum dan Keterbatasan Penelitian

### 4.8.1 Keberhasilan dan Kontribusi Penelitian

1. **Pipeline NLP Komprehensif**: Sistem preprocessing tujuh tahap mampu menangani keunikan bahasa Indonesia informal di media sosial secara efektif.
2. **Solusi Ketidakseimbangan Kelas**: Random Oversampling meningkatkan recall kelas Negatif dari ~31% menjadi **65%**, tanpa mengorbankan akurasi global.
3. **Akurasi Kompetitif**: Akurasi **80,85%** pada tiga kelas sentimen dari teks tidak terstruktur merupakan pencapaian solid.

### 4.8.2 Keterbatasan Penelitian

1. **Keterbatasan Data Negatif**: Hanya 113 komentar (9,6%) karena lagu memang diterima positif mayoritas pendengar.
2. **Ketergantungan Kamus Leksikon**: Pelabelan otomatis rentan pada sarkasme tingkat tinggi dan ironi; pelabelan manual oleh pakar berpotensi lebih akurat.
3. **Cakupan Kamus Slang**: Variasi bahasa informal terus berkembang sehingga beberapa kata slang baru mungkin tidak tercakup.
4. **Konteks Temporal**: Dataset hanya merepresentasikan kondisi sentimen pada satu titik waktu (Juli 2026).
5. **Sarkasme dan Bahasa Figuratif**: Model TF-IDF+SVM belum mampu menangani _sarcasm detection_ secara memadai.
6. **Bias Entitas Nama Tokoh dan Judul Album (_Named Entity & Title Bias_)**:
   - Nama vokalis _"Baskara"_ atau _"Bas"_ sering digunakan dalam komentar opini apresiasi vokal secara tidak langsung. Karena tidak membawa nilai sentimen intrinsik dalam kamus leksikon (skor 0), opini tersebut rentan bergeser ke label **Netral** apabila tidak dibarengi kata sifat positif yang eksplisit.
   - Judul album .FEAST yaitu _"Membangun & Menghancurkan"_ mengandung kata _"menghancurkan"_ atau _"hancur"_. Dalam kamus leksikon, kata _"hancur"_ dinilai sangat negatif (skor -2). Akibatnya, komentar yang netral/positif namun hanya sekadar menyebutkan judul album (misal: _"tunggu next lagu dari album membangun dan menghancurkan"_) secara tidak sengaja terlabeli sebagai **Negatif** oleh pelabel leksikon otomatis.

---

## 4.9 Ringkasan Jawaban Rumusan Masalah

|  No.  | Rumusan Masalah                                                 | Jawaban                                                                                                                                                                                           |
| :---: | :-------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **1** | Bagaimana penerapan SVM + TF-IDF untuk klasifikasi sentimen?    | Diimplementasikan melalui pipeline 6 tahap KDD: _scraping_ → _preprocessing_ (7 tahap) → _labeling leksikon_ → TF-IDF Vectorization → SVM Linear (GridSearchCV + Random Oversampling) → Evaluasi. |
| **2** | Seberapa baik kinerja model SVM (Akurasi, Presisi, Recall, F1)? | **Akurasi 80,85%**, Presisi 80,55%, Recall 80,85%, F1-Score 80,66% — performa baik untuk klasifikasi teks informal tiga kelas dengan dataset terbatas.                                            |
| **3** | Bagaimana kecenderungan dominasi sentimen pendengar?            | **Sentimen Netral mendominasi (64,2%)**, diikuti Positif (26,2%) dan Negatif (9,6%), menunjukkan lagu "Arteri" diterima secara positif-netral oleh mayoritas pendengar YouTube.                   |
