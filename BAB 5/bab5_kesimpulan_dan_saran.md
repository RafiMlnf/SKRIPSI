# BAB V

# PENUTUP

---

## 5.1 Kesimpulan

Berdasarkan serangkaian proses penelitian yang telah dilaksanakan — mulai dari pengumpulan data komentar YouTube, pemrosesan awal teks, pelabelan sentimen berbasis leksikon, ekstraksi fitur TF-IDF, pelatihan model SVM, hingga evaluasi komprehensif — terdapat tiga kesimpulan utama yang menjawab masing-masing rumusan masalah yang telah ditetapkan pada Bab I.

### 5.1.1 Penerapan Algoritma SVM dengan Pembobotan TF-IDF

Algoritma **Support Vector Machine (SVM)** berhasil diimplementasikan untuk mengklasifikasikan sentimen komentar publik pada video musik _".Feast – Arteri"_ melalui alur _pipeline_ enam tahap berbasis kerangka **Knowledge Discovery in Databases (KDD)**, yaitu:

1. **Pengumpulan Data**: Sebanyak **1.531 komentar mentah** berhasil dikumpulkan dari kolom komentar YouTube menggunakan teknik _web scraping_ melalui pustaka `youtube-comment-downloader` dengan batas waktu penarikan data (_cutoff_) pada 8 Juli 2026.
2. **Pemrosesan Awal (_Text Preprocessing_)**: Teks komentar dibersihkan dan dinormalisasi melalui tujuh tahapan berurutan — _case folding_, _data cleaning_ (regex), penghapusan emoji & emotikon, penyaringan spam & lirik lagu, normalisasi kata gaul menggunakan kamus alay 4.300+ entri, penghapusan _stopwords_ (dengan mempertahankan kata negasi), dan _stemming_ Bahasa Indonesia menggunakan PySastrawi. Proses ini menghasilkan **1.175 komentar bersih** yang siap diolah, setelah membuang 347 komentar yang teridentifikasi sebagai duplikat, spam, lirik lagu, atau terlalu pendek/panjang.
3. **Pelabelan Sentimen**: Komentar bersih diberi label sentimen otomatis menggunakan metode berbasis kamus leksikon (_Lexicon-Based_) dengan ambang batas skor ±1. Distribusi kelas sentimen yang dihasilkan adalah: **Positif (26,2%)**, **Netral (64,2%)**, dan **Negatif (9,6%)**.
4. **Ekstraksi Fitur TF-IDF**: Representasi vektor numerik dari data teks diperoleh menggunakan **TF-IDF** (_Term Frequency-Inverse Document Frequency_) dengan maksimal 5.000 fitur dan kombinasi _n-gram_ (unigram + bigram), menghasilkan matriks fitur berdimensi 745 dari data latih.
5. **Pelatihan Model**: Model SVM dengan kernel _linear_ dilatih menggunakan 940 sampel data latih (80%) yang telah diseimbangkan kelasnya melalui _Random Oversampling_ dan pembobotan kelas (`class_weight='balanced'`). Hyperparameter optimal (`C=1.0`, `gamma='scale'`) diperoleh melalui pencarian otomatis **GridSearchCV** dengan skema validasi silang 5-lipat (_5-fold cross-validation_).
6. **Evaluasi**: Model SVM dievaluasi menggunakan 235 sampel data uji (20%) yang belum pernah dilihat selama pelatihan, menggunakan metrik _Confusion Matrix_ (Akurasi, Presisi, Recall, dan F1-Score).

### 5.1.2 Kinerja Model SVM dalam Mengenali Pola Sentimen

Model SVM yang dibangun menunjukkan kinerja yang **baik dan kompetitif** dalam mengklasifikasikan sentimen teks komentar tidak terstruktur berbahasa Indonesia informal. Hasil evaluasi kuantitatif menunjukkan:

| Metrik Evaluasi                     |   Nilai    |
| :---------------------------------- | :--------: |
| **Akurasi (_Accuracy_)**            | **80,85%** |
| **Presisi (_Precision_, Weighted)** | **80,55%** |
| **Recall (Weighted)**               | **80,85%** |
| **F1-Score (Weighted)**             | **80,66%** |

Nilai akurasi **80,85%** berarti model berhasil mengklasifikasikan sekitar **190 dari 235 komentar** data uji dengan benar. Secara per kelas, model memiliki performa terbaik pada kelas **Netral** (F1-Score 0,87) karena dominasi data pada kelas ini, diikuti kelas **Positif** (F1-Score 0,71), dan kelas **Negatif** (F1-Score 0,62) yang tetap stabil berkat penerapan _Random Oversampling_.

Sebagai validasi tambahan, model SVM yang diusulkan berhasil **mengungguli model pembanding Logistic Regression** di seluruh metrik evaluasi — dengan selisih Akurasi sebesar **+1,70%** dan F1-Score sebesar **+1,52%**. Hal ini membuktikan bahwa prinsip _Maximal Margin Classifier_ pada SVM lebih tangguh dalam menghadapi _noise_ linguistik dari teks informal komentar YouTube dibandingkan pendekatan probabilistik.

Nilai akurasi 80,85% pada penelitian ini terbukti **kompetitif secara ilmiah** mengingat tingkat kesulitannya jauh lebih tinggi dibandingkan mayoritas penelitian terdahulu, yakni: menggunakan **tiga kelas sentimen** (bukan dua), domain data berupa komentar YouTube yang sangat informal dan penuh bahasa gaul, serta ukuran dataset yang lebih terbatas.

### 5.1.3 Kecenderungan Dominasi Sentimen Pendengar Lagu "Arteri"

Berdasarkan hasil klasifikasi model SVM terhadap 1.175 komentar, terdapat kecenderungan dominasi sentimen yang jelas dan memiliki makna sosiologis yang mendalam:

- **Sentimen Netral mendominasi** dengan proporsi **64,2% (754 komentar)**. Temuan ini tidak mencerminkan ketidakpedulian pendengar, melainkan menggambarkan pola komunikasi khas di YouTube — komentar pendengar didominasi oleh curahan pengalaman hidup _quarter-life crisis_ yang bersifat naratif informatif, diskusi komparatif musikal dengan band _My Chemical Romance_ (MCR), serta kutipan dan interpretasi makna lirik tanpa penilaian evaluatif yang eksplisit. Fenomena ini juga diperkuat oleh **bias entitas nama tokoh**: nama vokalis _"Baskara"_ yang sering disebut dalam konteks apresiasi tidak terdeteksi sebagai kata sentimen positif oleh kamus leksikon, sehingga komentar apresiasi implisit ini terkategorikan sebagai Netral.

- **Sentimen Positif berada di posisi kedua** dengan proporsi **26,2% (308 komentar)**. Kata-kata dominan seperti _"bagus"_, _"keren"_, _"relate"_, _"indah"_, dan _"masterpiece"_ menunjukkan apresiasi pendengar yang tinggi terhadap kualitas musikalitas, kedalaman lirik, dan relevansi personal narasi _quarter-life crisis_ yang diangkat lagu "Arteri".

- **Sentimen Negatif memiliki proporsi paling kecil** yakni **9,6% (113 komentar)**. Kritik yang ada umumnya berupa ketidakcocokan selera musikal atau ekspresi emosi negatif yang dipicu oleh kenangan personal yang menyakitkan akibat mendengarkan lagu ini, bukan sebuah penolakan terhadap kualitas lagu secara objektif.

Secara keseluruhan, temuan ini menunjukkan bahwa lagu _"Arteri"_ oleh **.FEAST** **diterima secara sangat positif dan memberi dampak emosional yang signifikan** oleh mayoritas pendengar di YouTube. Respons publik yang didominasi oleh narasi personal dan apresiasi mencerminkan keberhasilan lagu ini dalam membangun koneksi emosional yang autentik dengan fenomena _quarter-life crisis_ yang dialami oleh generasi muda Indonesia.

---

## 5.2 Saran

Berdasarkan keterbatasan penelitian yang telah diidentifikasi pada Bab IV, saran yang direkomendasikan adalah sebagai berikut.

### 5.2.1 Saran Pengembangan Teknis dan Ilmiah

1. **Pelabelan Manual oleh Pakar (_Human Annotation_)**: Melengkapi pelabelan leksikon dengan anotasi manual oleh 2–3 penilai manusia untuk meningkatkan kualitas label sentimen, terutama pada komentar yang mengandung sarkasme atau ironi implisit.

2. **Ekspansi Kamus Alay dan Leksikon Sentimen**: Kamus alay dan leksikon sentimen perlu diperbarui secara berkala untuk mengakomodasi perkembangan kata slang baru di media sosial yang belum tercakup dalam kamus yang digunakan.

3. **Eksplorasi Model Berbasis Representasi Kontekstual**: Mengeksplorasi model bahasa pra-latih berbasis Transformer seperti **IndoBERT** atau **mBERT** yang mampu menangkap relasi kontekstual antar kata, sehingga meningkatkan akurasi deteksi sarkasme dan sentimen ambigu.

4. **Pengembangan _Named Entity Recognition_ (NER)**: Mengintegrasikan NER untuk mendeteksi dan mengecualikan nama tokoh dan judul karya dari perhitungan skor leksikon, guna mengurangi bias entitas yang teridentifikasi pada penelitian ini.

5. **Perluasan Dataset**: Memperluas dataset dengan mengumpulkan komentar dari video lain milik .FEAST atau artis indie Indonesia lainnya untuk meningkatkan kemampuan generalisasi model.

6. **Aspect-Based Sentiment Analysis (ABSA)**: Mengembangkan analisis sentimen pada level aspek tertentu (vokal, lirik, aransemen) untuk menghasilkan wawasan yang lebih granular dan bernilai bagi industri musik.

### 5.2.2 Saran Praktis bagi Pemangku Kepentingan

1. **Bagi Grup Musik .FEAST**: Temuan riset ini mengonfirmasi bahwa audiens sangat responsif terhadap narasi jujur dan emosional seputar _quarter-life crisis_. Mempertahankan gaya penulisan lirik yang autentik dapat memperkuat loyalitas komunitas pendengar.

2. **Bagi Industri Musik Digital Indonesia**: _Pipeline_ NLP yang dikembangkan dalam penelitian ini dapat diadaptasi sebagai kerangka pemantauan persepsi publik terhadap karya musik secara _real-time_, sebagai alternatif survei manual yang mahal.

3. **Bagi Peneliti Selanjutnya**: Dataset bersih berlabel (1.175 komentar), model SVM terlatih (`.pkl`), dan _pipeline preprocessing_ yang terdokumentasi dalam penelitian ini tersedia sebagai _baseline_ perbandingan untuk penelitian analisis sentimen Bahasa Indonesia domain musik selanjutnya.
