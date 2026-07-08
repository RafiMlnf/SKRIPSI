# Rencana Struktur BAB IV: HASIL DAN PEMBAHASAN

## Analisis Sentimen Lagu "Arteri" - .FEAST Menggunakan Metode SVM

Dokumen ini berisi rancangan sub-bab dan sub-sub-bab untuk penyusunan **BAB IV** laporan skripsi Anda, yang disesuaikan dengan alur source code dan data yang telah dikerjakan.

---

### **BAB IV: HASIL DAN PEMBAHASAN**

#### **4.1. Deskfripsi dan Pengumpulan Data**

Bagian ini menjelaskan sumber data, waktu pengambilan, serta karakteristik awal dataset yang diperoleh sebelum dilakukan pemrosesan.

- **4.1.1. Sumber Data dan Batasan Pengambilan (_Data Cutoff_)**
  - Menjelaskan bahwa data diperoleh dari video musik resmi lagu "Arteri" oleh .FEAST ([youtube.com/watch?v=eulYTPXOaio](https://www.youtube.com/watch?v=eulYTPXOaio)).
  - Mencantumkan tanggal pengambilan data pada **18 Januari 2026** menggunakan `youtube-comment-downloader`.
  - Menjelaskan alasan pembatasan tanggal pengambilan data (_cutoff date_) agar dataset bersifat statis dan konsisten untuk kebutuhan penelitian.
- **4.1.2. Statistik dan Karakteristik Dataset Mentah (_Raw Data_)**
  - Menyebutkan jumlah total data mentah yang berhasil dikumpulkan, yaitu sebanyak **1.146 komentar** ([comments_arteri_full.csv](file:///d:/Kuliah/0%20SKRIPSI/SOURCE%20CODE/Data/comments_arteri_full.csv)).
  - Menampilkan contoh beberapa baris data mentah beserta kolom-kolomnya (`no`, `author`, `text`, `time`, `likes`, `replies`, `is_reply`).

#### **4.2. Pemrosesan Awal Data (_Data Preprocessing_)**

Bagian ini menjelaskan secara rinci tahapan pembersihan data mentah hingga menjadi data bersih yang siap diolah. Setiap tahapan harus disertai contoh perubahan teksnya.

- **4.2.1. Penyeragaman Huruf (_Case Folding_)**
  - Mengubah seluruh teks komentar menjadi huruf kecil (_lowercase_).
- **4.2.2. Pembersihan Karakter (_Data Cleaning_)**
  - Menghapus tautan/URL, mention username (`@`), hashtag (`#`), serta karakter khusus/tanda baca yang tidak relevan.
- **4.2.3. Penghapusan Emoji (_Emoji Removal_)**
  - Menghilangkan karakter emoji dan simbol non-ASCII agar tidak mengganggu proses klasifikasi teks.
- **4.2.4. Deteksi dan Penyaringan Spam/Lirik Lagu (_Spam & Lyrics Filtering_)**
  - Menjelaskan proses penyaringan komentar spam serta komentar yang hanya berisi salinan lirik lagu ([comments_arteri_lyric.csv](file:///d:/Kuliah/0%20SKRIPSI/SOURCE%20CODE/Data/comments_arteri_lyric.csv)).
- **4.2.5. Normalisasi Kata Gaul/Singkatan (_Slang Word Normalization_)**
  - Menjelaskan pencocokan kata-kata tidak baku dengan kamus alay ([kamus_alay.csv](file:///d:/Kuliah/0%20SKRIPSI/SOURCE%20CODE/Data/kamus_alay.csv)) yang berisi lebih dari 500+ entri kata tidak baku ke bentuk baku.
- **4.2.6. Penghapusan Kata Henti (_Stopwords Removal_)**
  - Menghilangkan kata-kata umum dalam Bahasa Indonesia yang tidak membawa informasi penting (seperti "yang", "dan", "di", dll.).
- **4.2.7. Pengakaran Kata (_Stemming_)**
  - Menjelaskan proses mengubah kata berimbuhan menjadi kata dasar menggunakan pustaka **Sastrawi**.
- **4.2.8. Hasil Akhir dan Perbandingan Data Preprocessing**
  - Menyajikan tabel perbandingan jumlah data sebelum dan sesudah preprocessing (dari 1.146 menjadi **821 komentar** atau menyusut sekitar **28.4%**).
  - Merujuk ke berkas output preprocessing: [comments_preprocessed.csv](file:///d:/Kuliah/0%20SKRIPSI/SOURCE%20CODE/Data/comments_preprocessed.csv) dan laporan [preprocessing_report.txt](file:///d:/Kuliah/0%20SKRIPSI/SOURCE%20CODE/Data/preprocessing_report.txt).

#### **4.3. Pelabelan Sentimen (_Sentiment Labeling_)**

Bagian ini menerangkan bagaimana data yang telah bersih diberi label kelas sentimen (Positif, Negatif, Netral).

- **4.3.1. Metode Pelabelan Berbasis Leksikon (_Lexicon-Based_)**
  - Menjelaskan penggunaan kamus sentimen kustom Bahasa Indonesia untuk menghitung skor polaritas kata.
- **4.3.2. Penanganan Negasi dan Kata Penguat (_Negation & Intensifier Handling_)**
  - Bagaimana kata seperti "tidak", "kurang" (negasi) atau "sangat", "sekali" (intensifier) mempengaruhi kalkulasi skor polaritas.
- **4.3.3. Batas Ambang (_Threshold_) Klasifikasi Kelas**
  - _Positif_ (Score &ge; 1), _Negatif_ (Score &le; -1), dan _Netral_ (-1 < Score < 1).
- **4.3.4. Analisis Distribusi Hasil Pelabelan**
  - Menampilkan grafik distribusi kelas sentimen ([sentiment_distribution.png](file:///d:/Kuliah/0%20SKRIPSI/SOURCE%20CODE/Data/sentiment_distribution.png)) beserta tabel ringkasannya.
  - Merujuk ke berkas hasil pelabelan: [comments_labeled.csv](file:///d:/Kuliah/0%20SKRIPSI/SOURCE%20CODE/Data/comments_labeled.csv) dan laporan [labeling_report.txt](file:///d:/Kuliah/0%20SKRIPSI/SOURCE%20CODE/Data/labeling_report.txt).

#### **4.4. Ekstraksi Fitur TF-IDF**

Proses konversi data teks terlabel menjadi representasi numerik.

- **4.4.1. Bobot Kata Term Frequency - Inverse Document Frequency (TF-IDF)**
  - Menyajikan formula matematika TF-IDF yang digunakan.
- **4.4.2. Matriks Fitur dan Ukuran Vokabulari**
  - Menjelaskan dimensi matriks yang dihasilkan setelah fit-transform menggunakan `TfidfVectorizer` ([tfidf_vectorizer.pkl](file:///d:/Kuliah/0%20SKRIPSI/SOURCE%20CODE/Output/tfidf_vectorizer.pkl)).

#### **4.5. Implementasi dan Pelatihan Model SVM**

Pembahasan teknis pemodelan Support Vector Machine.

- **4.5.1. Pembagian Data Latih dan Data Uji (_Data Splitting_)**
  - Menjelaskan rasio pembagian data (misal: 80% latih, 20% uji) dengan _stratified split_ untuk menjaga proporsi kelas.
- **4.5.2. Penyetelan Hyperparameter (_Hyperparameter Tuning_)**
  - Menjelaskan proses pencarian parameter optimal ($C$, $gamma$, dan $kernel$) menggunakan metode **GridSearchCV**.
- **4.5.3. Pembuatan Model SVM Terpilih**
  - Menyebutkan hyperparameter terbaik yang terpilih untuk model final ([svm_sentiment_model.pkl](file:///d:/Kuliah/0%20SKRIPSI/SOURCE%20CODE/Output/svm_sentiment_model.pkl)).

#### **4.6. Evaluasi dan Analisis Hasil**

Bagian kritis yang mengevaluasi performa model klasifikasi serta interpretasi temuan sentimen publik.

- **4.6.1. Pengukuran Metrik Performa Model**
  - Menampilkan tabel hasil metrik evaluasi: **Akurasi, Presisi, Recall, dan F1-Score** untuk masing-masing kelas (Positif, Negatif, Netral).
- **4.6.2. Analisis Confusion Matrix**
  - Menyajikan grafik matriks kebingungan ([confusion_matrix.png](file:///d:/Kuliah/0%20SKRIPSI/SOURCE%20CODE/Output/confusion_matrix.png)) untuk memperlihatkan performa prediksi model pada data uji secara visual.
- **4.6.3. Analisis Temuan Sentimen terhadap Lagu "Arteri"**
  - Membahas makna dari hasil klasifikasi: sentimen apa yang paling dominan di kalangan pendengar lagu "Arteri" oleh .FEAST (misalnya, respon emosional pendengar terkait makna lagu, vokal Baskara, dll.).
