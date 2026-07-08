# 📝 DRAF BAB IV: HASIL DAN PEMBAHASAN

Berikut adalah draf tulisan ilmiah lengkap untuk **BAB IV** laporan skripsi Anda, berdasarkan hasil eksekusi program Machine Learning SVM dan preprocessing terbaru. Anda dapat langsung menyalin naskah ini ke dalam dokumen Microsoft Word skripsi Anda.

---

## **BAB IV: HASIL DAN PEMBAHASAN**

### **4.1. Deskripsi dan Pengumpulan Data**

#### **4.1.1. Sumber Data dan Batasan Pengambilan (Data Cutoff)**
Data yang digunakan dalam penelitian ini diperoleh dari komentar publik pada video musik resmi lagu *"Arteri"* oleh grup musik indie/alternatif **.FEAST** yang diunggah di platform YouTube (tautan video: `https://www.youtube.com/watch?v=eulYTPXOaio`). 

Pengambilan data dilakukan menggunakan pustaka python `youtube-comment-downloader`. Proses *scraping* diselesaikan pada tanggal **18 Januari 2026** yang ditetapkan sebagai batas waktu penarikan data (*cutoff date*). Penentuan batas waktu ini penting dilakukan untuk memastikan dataset bersifat statis selama proses pelatihan model, pengujian, hingga penyusunan laporan, sehingga konsistensi nilai evaluasi dapat dipertahankan.

#### **4.1.2. Statistik dan Karakteristik Dataset Mentah (Raw Data)**
Proses penarikan data berhasil mengumpulkan sebanyak **1.146 komentar mentah** (tercantum dalam file `comments_arteri_full.csv`). Karakteristik data mentah tersebut meliputi informasi berupa nama pembuat komentar (`author`), isi teks komentar (`text`), waktu penulisan (`time`), jumlah suka (`likes`), jumlah balasan (`replies`), dan penanda apakah komentar tersebut merupakan balasan (*reply*) terhadap komentar lain (`is_reply`). 

Data komentar mentah didominasi oleh bahasa informal khas media sosial (penuh dengan singkatan, kata slang, bahasa daerah, serta emoji) yang merepresentasikan ekspresi emosional pendengar terkait makna lagu *"Arteri"*.

---

### **4.2. Pemrosesan Awal Data (Data Preprocessing)**
Untuk menghasilkan klasifikasi sentimen yang akurat, data mentah harus dibersihkan terlebih dahulu melalui beberapa tahapan preprocessing. Proses ini mereduksi derau linguistik (*noise*) yang dapat menurunkan akurasi model SVM. Berikut adalah tahapan pembersihan yang dilakukan beserta contoh transformasinya:

1. **Case Folding**: Mengubah semua karakter huruf kapital menjadi huruf kecil (*lowercase*).
   * *Contoh*: `"Lagu ini Relate Banget"` $\rightarrow$ `"lagu ini relate banget"`
2. **Data Cleaning**: Menghapus tautan web (URL), penyebutan akun (`@username`), tanda pagar (`#`), angka yang tidak bermakna, dan karakter khusus lainnya.
   * *Contoh*: `"keren bgt @baskara #arteri"` $\rightarrow$ `"keren bgt"`
3. **Emoji & Emoticon Removal**: Menghapus karakter emoji non-ASCII dan emotikon agar tidak merusak representasi vektor kata.
   * *Contoh*: `"sedih bgt 😭💔"` $\rightarrow$ `"sedih bgt"`
4. **Spam & Lyrics Filtering**: Menyingkirkan komentar yang hanya berisi salinan lirik lagu secara penuh (karena tidak merepresentasikan sentimen opini pendengar) serta komentar spam (seperti promo, link penipuan, atau komentar yang terlalu pendek).
5. **Normalisasi Kata Gaul (Slang Normalization)**: Mengubah kata tidak baku menjadi kata baku berdasarkan Kamus Alay (berisi lebih dari 4.300 entri kata slang). Pada tahap ini, kata khusus seperti nama personil (*"bas"* $\rightarrow$ *"baskara"*) dan nama grup (*"feast"* $\rightarrow$ *"feast"*) dipertahankan agar tidak berubah maknanya.
   * *Contoh*: `"gw ga suka lagu ini"` $\rightarrow$ `"saya tidak suka lagu ini"`
6. **Stopwords Removal**: Menghilangkan kata-kata umum dalam bahasa Indonesia yang tidak membawa informasi sentimen yang berarti (seperti *"yang"*, *"dan"*, *"di"*). Namun, kata penegas negasi (seperti *"tidak"*, *"enggak"*, *"jangan"*, *"belum"*) sengaja **dipertahankan** agar logika pembalik sentimen tidak hilang.
   * *Contoh*: `"lagu yang sangat bagus"` $\rightarrow$ `"lagu bagus"`
7. **Stemming (Pengakaran Kata)**: Mengubah kata berimbuhan menjadi kata dasarnya menggunakan algoritma stemmer **Sastrawi**.
   * *Contoh*: `"mendengarkan"` $\rightarrow$ `"dengar"`, `"menangis"` $\rightarrow$ `"tangis"`

#### **Perbandingan Dataset Sebelum dan Sesudah Preprocessing**
Setelah melewati sembilan tahapan preprocessing, dataset mengalami penyusutan ukuran dari **1.146 komentar** menjadi **824 komentar bersih**. Penyusutan data sebesar **28,1%** (315 komentar dihapus) disebabkan oleh alasan kualitas yang dirangkum pada Tabel 4.1 berikut:

| Kategori Penghapusan | Jumlah Komentar Dihapus |
| :--- | :---: |
| Komentar Kosong (Hanya spasi/karakter non-huruf) | 59 |
| Salinan Lirik Lagu (*Lyrics Copy*) | 25 |
| Deteksi Spam (Promo/Karakter berulang) | 79 |
| Terlalu Pendek (< 2 kata atau < 5 karakter) | 125 |
| Terlalu Panjang (> 100 kata) | 1 |
| Komentar Duplikat (*Exact Duplicates*) | 26 |
| **Total Komentar Dihapus** | **315** |
| **Total Komentar Bersih Terpilih** | **824** |

---

### **4.3. Pelabelan Sentimen (Sentiment Labeling)**
Komentar bersih yang telah disiapkan kemudian diberi label kelas sentimen (Positif, Negatif, Netral) secara otomatis menggunakan metode **Lexicon-Based**.

#### **4.3.1. Metode Pelabelan Berbasis Leksikon**
Metode ini bekerja dengan membandingkan kata-kata dalam komentar bersih terhadap kamus sentimen bahasa Indonesia. Setiap kata positif memiliki bobot numerik $+1$ hingga $+3$, sedangkan kata negatif memiliki bobot $-1$ hingga $-3$. Total skor sentimen dari suatu komentar dihitung berdasarkan akumulasi bobot kata yang ditemukan.

#### **4.3.2. Penanganan Negasi dan Kata Penguat**
Untuk meningkatkan akurasi pelabelan, algoritma dilengkapi dengan penanganan negasi (*negation handling*) dan kata penguat (*intensifier*):
* **Kata Penguat**: Kata seperti *"sangat"*, *"sekali"*, *"banget"* akan memperkuat skor sentimen kata di dekatnya dengan faktor pengali sebesar $1.5$.
* **Negasi**: Kata seperti *"tidak"*, *"ga"*, *"belum"* akan membalikkan arah polaritas skor sentimen kata sesudahnya (positif menjadi negatif, dan sebaliknya) dengan bobot dikurangi $50\%$ untuk mensimulasikan pelemahan makna opini.

#### **4.3.3. Batas Ambang (Threshold) Klasifikasi Kelas**
Skor akumulatif dari perhitungan leksikon dikonversi menjadi label sentimen dengan batasan ambang (*threshold*) sebagai berikut:
* **Sentimen Positif**: $\text{Skor} \ge 1$ (Label Kelas: `2`)
* **Sentimen Negatif**: $\text{Skor} \le -1$ (Label Kelas: `0`)
* **Sentimen Netral**: $-1 < \text{Skor} < 1$ (Label Kelas: `1`)

#### **4.3.4. Distribusi Hasil Pelabelan**
Proses pelabelan terhadap 824 komentar menghasilkan distribusi sentimen yang disajikan pada Tabel 4.2:

| Kelas Sentimen | Jumlah Komentar | Persentase (%) |
| :--- | :---: | :---: |
| **Positif** | 370 | 44,9% |
| **Netral** | 374 | 45,4% |
| **Negatif** | 80 | 9,7% |
| **Total** | **824** | **100,0%** |

Distribusi data menunjukkan bahwa mayoritas pendengar memberikan respons bernada netral (membahas lirik, aransemen, atau perbandingan band) dan positif (mengapresiasi keindahan lagu, relate dengan kehidupan pribadi). Kelas negatif memiliki proporsi paling kecil (hanya 9,7%), yang mengindikasikan lagu *"Arteri"* diterima dengan sangat baik oleh publik.

---

### **4.4. Ekstraksi Fitur TF-IDF**
Vektor numerik dari data tekstual diekstraksi menggunakan metode **Term Frequency - Inverse Document Frequency (TF-IDF)** dengan batas ukuran kosakata maksimal (*max_features*) sebesar 5.000 kata, menggunakan gabungan *unigram* dan *bigram* (`ngram_range=(1,2)`). 

Proses ekstraksi menghasilkan **745 fitur kata unik** dari data latih. Representasi data berubah menjadi matriks berdimensi tinggi:
* Matriks Data Latih (*Train Shape*): $659 \times 745$
* Matriks Data Uji (*Test Shape*): $165 \times 745$

---

### **4.5. Implementasi dan Pelatihan Model SVM**

#### **4.5.1. Pembagian Data (Data Splitting)**
Dataset dibagi secara acak dengan rasio **80% untuk data latih (659 komentar)** dan **20% untuk data uji (165 komentar)**. Pembagian ini menggunakan metode *Stratified Split* untuk memastikan proporsi kelas sentimen (Positif, Netral, Negatif) pada data latih dan data uji tetap seimbang (serupa dengan distribusi asli dataset).

#### **4.5.2. Penyetelan Hyperparameter (GridSearchCV)**
Pencarian hyperparameter optimal untuk model Support Vector Machine (SVM) dilakukan menggunakan metode **GridSearchCV** dengan skema *5-fold cross-validation*. Grid pencarian mencakup variasi kernel (`linear`, `rbf`), nilai parameter regularisasi $C$ (`0.1`, `1`, `10`), dan koefisien kernel $\gamma$ (`scale`, `auto`).

Berdasarkan pencarian GridSearchCV, diperoleh kombinasi hyperparameter terbaik:
* **Kernel**: `linear`
* **Regularization Parameter ($C$)**: `10`
* **Gamma ($\gamma$)**: `scale`
* **Best CV Score (Weighted F1)**: **69,46%**

Untuk mengatasi ketidakseimbangan kelas (*class imbalance*) di mana kelas negatif sangat sedikit (9,7%), parameter `class_weight='balanced'` diterapkan pada SVM. Ini memberikan bobot penalti kesalahan klasifikasi kelas negatif $3,43 \times$ lebih besar daripada kelas positif/netral selama proses latih.

---

### **4.6. Evaluasi dan Analisis Performa Model**

#### **4.6.1. Pengukuran Metrik Performa Model**
Performa model SVM diuji menggunakan data uji (165 komentar) yang belum pernah dilihat model sebelumnya. Hasil pengujian menunjukkan performa yang cukup baik, dengan rincian metrik pada Tabel 4.3:

| Metrik Evaluasi | Nilai (%) |
| :--- | :---: |
| **Akurasi** | 67,88% |
| **Precision (Weighted)** | 67,90% |
| **Recall (Weighted)** | 67,88% |
| **F1-Score (Weighted)** | 67,80% |

Rincian klasifikasi per kelas sentimen ditunjukkan pada Tabel 4.4 berikut:

| Kelas Sentimen | Precision | Recall | F1-Score | Support (Jumlah Data Uji) |
| :--- | :---: | :---: | :---: | :---: |
| **Negatif** | 0,33 | 0,31 | 0,32 | 16 |
| **Netral** | 0,68 | 0,73 | 0,71 | 75 |
| **Positif** | 0,75 | 0,70 | 0,73 | 74 |
| **Rata-rata (Weighted)** | **0,68** | **0,68** | **0,68** | **165** |

Model memiliki performa terbaik dalam memprediksi kelas **Positif** (F1-Score 73%) dan kelas **Netral** (F1-Score 71%). Kelas **Negatif** memiliki performa yang lebih rendah (F1-Score 32%) disebabkan keterbatasan jumlah sampel latih kelas negatif (hanya 64 komentar pada data latih), namun berkat pembobotan penalti kelas (*balanced weights*), model tetap mampu mengidentifikasi sebagian sentimen negatif dengan baik.

#### **4.6.2. Analisis Confusion Matrix**
Persebaran hasil prediksi model SVM pada data uji digambarkan melalui Confusion Matrix pada Tabel 4.5:

| Aktual \ Prediksi | Negatif | Netral | Positif |
| :--- | :---: | :---: | :---: |
| **Negatif** | **5** (Benar) | 8 | 3 |
| **Netral** | 6 | **55** (Benar) | 14 |
| **Positif** | 4 | 18 | **52** (Benar) |

Berdasarkan Confusion Matrix:
* Dari **16 komentar negatif aktual**, model memprediksi 5 komentar dengan benar sebagai negatif, sedangkan 8 terprediksi netral, dan 3 terprediksi positif.
* Dari **75 komentar netral aktual**, model memprediksi 55 komentar dengan benar sebagai netral, sedangkan 6 terprediksi negatif, dan 14 terprediksi positif.
* Dari **74 komentar positif aktual**, model memprediksi 52 komentar dengan benar sebagai positif, sedangkan 4 terprediksi negatif, dan 18 terprediksi netral.

#### **4.6.3. Analisis Temuan Sentimen terhadap Lagu "Arteri"**
Hasil klasifikasi memperlihatkan dominasi respons **Netral (45,4%)** dan **Positif (44,9%)** dari pendengar. Temuan utama dari analisis ini menunjukkan:
1. **Validasi Emosional Quarter-Life Crisis**: Banyak pendengar menuliskan komentar panjang menceritakan kecemasan hidup di usia 20-an (*quarter-life crisis*) yang diungkapkan secara jujur. Lagu ini berfungsi sebagai sarana koping dan katarsis emosional.
2. **Koneksi dengan Band .FEAST**: Baskara Putra (vokalis) menerima banyak apresiasi positif atas penulisan lirik yang dalam (*deep*). Kata *"baskara"* atau *"bas"* sangat sering muncul dalam komentar positif.
3. **Pengaruh Musik Alternatif**: Banyak komentar yang membandingkan atmosfer lagu *"Arteri"* dengan band legendaris *My Chemical Romance (MCR)* (khususnya lagu *Helena* atau *I'm Not Okay*), yang diklasifikasikan sebagai sentimen positif karena dinilai sebagai pujian terhadap aransemen lagu.
