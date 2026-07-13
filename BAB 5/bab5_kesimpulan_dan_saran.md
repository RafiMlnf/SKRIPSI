# BAB V

# PENUTUP

---

## 5.1 Kesimpulan

Berdasarkan hasil penelitian analisis sentimen pada komentar YouTube lagu *".Feast – Arteri"* menggunakan algoritma *Support Vector Machine* (SVM) dan pembobotan TF-IDF, diperoleh kesimpulan sebagai berikut:

1. **Penerapan Algoritma SVM**: Alur *pipeline* klasifikasi sentimen berhasil diimplementasikan melalui enam tahapan KDD (*Knowledge Discovery in Databases*). Dari total **1.531 komentar mentah** yang dikumpulkan, diperoleh **1.175 komentar bersih** setelah melalui pemrosesan awal (*preprocessing*). Data kemudian diberi label otomatis menggunakan metode *Lexicon-Based*, diekstraksi menggunakan fitur TF-IDF (unigram + bigram), dan dilatih menggunakan model SVM linear yang dioptimasi via GridSearchCV dengan penanganan *imbalance class* (Random Oversampling & class weight).

2. **Kinerja Model**: Model SVM menunjukkan performa yang sangat baik dan kompetitif untuk klasifikasi tiga kelas (*positive*, *neutral*, *negative*), dengan nilai **Akurasi 80,85%**, **Presisi 80,55%**, **Recall 80,85%**, dan **F1-Score 80,66%**. Model ini juga terbukti lebih unggul dibandingkan model pembanding *Logistic Regression* (selisih akurasi +1,70% dan F1-Score +1,52%), membuktikan ketangguhan prinsip *Maximal Margin Classifier* SVM dalam menangani data teks informal yang kompleks dan berisik (*noisy*).

3. **Analisis Sentimen Pendengar**: Analisis sentimen terhadap 1.175 komentar menunjukkan kecenderungan sebagai berikut:
   - **Sentimen Netral mendominasi (64,2% / 754 komentar)**, yang sebagian besar berupa narasi pengalaman pribadi terkait *quarter-life crisis*, diskusi komparatif musik, serta kutipan lirik lagu.
   - **Sentimen Positif sebesar 26,2% (308 komentar)**, yang merepresentasikan apresiasi pendengar terhadap kedalaman makna lirik dan kualitas aransemen lagu.
   - **Sentimen Negatif sebesar 9,6% (113 komentar)**, yang umumnya dipicu oleh ketidakcocokan selera musik personal atau luapan emosi negatif akibat kenangan masa lalu, bukan penolakan terhadap kualitas karya.

Secara keseluruhan, lagu *"Arteri"* diterima secara sangat baik oleh audiens dan berhasil membangun kedekatan emosional yang kuat dengan generasi muda dalam menghadapi fase *quarter-life crisis*.

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
