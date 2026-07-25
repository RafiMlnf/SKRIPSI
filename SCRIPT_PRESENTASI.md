# SCRIPT PRESENTASI SKRIPSI
## Rafi Maulana Firdaus — 312210382

---

## SLIDE 1 — PEMBUKAAN

Assalamu'alaikum warahmatullahi wabarakatuh.

Selamat pagi, Bapak/Ibu dosen penguji yang saya hormati.

Perkenalkan, saya **Rafi Maulana Firdaus**, NIM 312210382, dari Program Studi Teknik Informatika, Universitas Pelita Bangsa. Pada kesempatan ini, saya akan mempresentasikan skripsi saya yang berjudul **"Analisis Sentimen Pendengar Lagu Arteri oleh .Feast Menggunakan Algoritma Support Vector Machine."**

---

## SLIDE 2 — LATAR BELAKANG

Jadi, penelitian ini berawal dari fenomena yang cukup menarik. Lagu **"Arteri"** oleh **.Feast** viral di YouTube — dan ribuan orang meninggalkan komentar yang menceritakan pengalaman *quarter-life crisis* mereka.

Nah, komentar-komentar ini tidak terstruktur. Bahasanya campur: ada slang, singkatan, bahkan sarkasme. Analisis manual tentu tidak efisien dan rawan bias.

Maka dari itu, saya mengusulkan solusi otomatisasi menggunakan **Machine Learning** — tepatnya algoritma **SVM** — untuk mengklasifikasikan sentimen komentar tersebut secara akurat.

---

## SLIDE 3 — RUMUSAN & TUJUAN MASALAH

Ada tiga rumusan masalah yang saya angkat dalam penelitian ini.

**Pertama**, bagaimana penerapan SVM dan TF-IDF untuk klasifikasi sentimen? **Kedua**, seberapa baik kinerjanya jika diukur dengan Confusion Matrix? Dan **ketiga**, bagaimana kecenderungan dominasi sentimen pendengar — apakah positif, negatif, atau netral?

Ketiga pertanyaan inilah yang akan saya jawab di akhir presentasi.

---

## SLIDE 4 — LANDASAN TEORI

Nah, di sini saya jelaskan dua teknologi inti yang digunakan.

**SVM** — atau *Support Vector Machine* — bekerja dengan mencari *hyperplane* optimal yang memisahkan kelas sentimen. Metode ini unggul pada data berdimensi tinggi seperti teks.

**TF-IDF** bertugas mengubah teks mentah menjadi vektor numerik. Ia memberi bobot lebih tinggi pada kata-kata yang spesifik dan bermakna, sekaligus menekan kata-kata umum yang tidak informatif.

Jadi, keduanya bekerja bersama: TF-IDF mengubah teks menjadi angka, lalu SVM mengklasifikasikannya.

---

## SLIDE 5 — METODOLOGI

Metodologi yang saya gunakan mengacu pada kerangka **KDD** — *Knowledge Discovery in Databases* — yang terdiri dari enam tahap.

Dimulai dari pengumpulan data melalui *web scraping*, lalu **preprocessing** teks dalam tujuh sub-tahap, kemudian ekstraksi fitur dengan TF-IDF, pembagian data, pelatihan model SVM, dan terakhir evaluasi performa.

Setiap tahap dirancang sistematis agar hasilnya dapat direproduksi.

---

## SLIDE 6 — DATA & PREPROCESSING

Nah, berikut ini saya jelaskan statistik dataset yang saya gunakan.

Data awal yang berhasil di-scraping berjumlah **1.531 komentar**. Setelah proses pembersihan, yang tersisa adalah **1.175 komentar valid**.

Untuk distribusi sentimennya — seperti yang terlihat di tabel ini — kelas **Netral mendominasi dengan 64,2%**, diikuti Positif 26,2%, dan Negatif 9,6%. Ketimpangan inilah yang nantinya menjadi tantangan dalam pelatihan model.

---

## SLIDE 7 — HASIL PELATIHAN MODEL

Jadi, untuk menemukan hyperparameter terbaik, saya menggunakan **GridSearchCV dengan 5-Fold Cross-Validation**.

Hasilnya: kernel **Linear** dengan nilai **C = 1.0**, menggunakan strategi **One-vs-One**, menghasilkan *CV Score* terbaik sekitar **80%**.

Nah, karena kelas Negatif hanya 9,6% dari data, model awalnya cenderung mengabaikan kelas ini. Saya tangani dengan **Random Oversampling**, dan hasilnya *Recall* kelas Negatif naik signifikan — dari **31% menjadi 65%**.

---

## SLIDE 8 — HASIL EVALUASI

Di sini saya paparkan hasil evaluasi akhir model pada **235 data uji**.

Model berhasil mengklasifikasikan **190 dari 235 komentar** dengan benar. Secara metrik: **Akurasi 80,85%**, Presisi 80,55%, Recall 80,85%, dan F1-Score 80,66%.

Nah, pada *Confusion Matrix*, kesalahan terbesar terjadi pada kelas **Positif yang salah diprediksi sebagai Netral** — sebanyak 16 kasus. Ini karena batas semantik antara apresiasi dan komentar informatif memang sangat tipis dalam teks informal.

Sebagai perbandingan, SVM unggul **+1,70%** dibanding Logistic Regression — yang membuktikan keunggulan prinsip *maximal margin* pada data teks.

---

## SLIDE 9 — ANALISIS TREN SENTIMEN

Jadi, berdasarkan hasil klasifikasi, sentimen **Netral mendominasi dengan 64,2%**. Ini bukan berarti pendengar tidak peduli — melainkan karena mereka lebih banyak berbagi narasi dan cerita pribadi tanpa memberi penilaian eksplisit.

Sentimen **Positif 26,2%** didominasi kata-kata seperti *relate*, *bagus*, dan *suka* — menunjukkan resonansi emosional yang tinggi terhadap lagu ini.

Sementara **Negatif 9,6%** lebih banyak berasal dari luapan emosi pribadi terkait kenangan menyakitkan — bukan penolakan terhadap kualitas karya.

---

## SLIDE 10 — KESIMPULAN

Baik, izinkan saya menyimpulkan.

**Pertama**, pipeline KDD enam tahap berhasil diimplementasikan secara penuh — mulai dari *scraping* hingga klasifikasi SVM.

**Kedua**, model mencapai **akurasi 80,85%** dengan F1-Score 80,66% — kompetitif untuk klasifikasi tiga kelas pada teks informal bahasa Indonesia.

**Ketiga**, lagu "Arteri" diterima secara positif-netral oleh pendengar, dengan resonansi emosional yang kuat terhadap tema *quarter-life crisis*.

Nah, untuk pengembangan ke depan, saya merekomendasikan penggunaan **IndoBERT**, anotasi manual oleh anotator ahli, serta implementasi **Aspect-Based Sentiment Analysis** agar analisis bisa lebih mendalam per aspek — vokal, lirik, maupun aransemen.

Demikian presentasi dari saya. Terima kasih atas perhatian Bapak/Ibu, dan saya siap menerima pertanyaan.

*Wassalamu'alaikum warahmatullahi wabarakatuh.*

---

> **Rafi Maulana Firdaus | 312210382 | Teknik Informatika, Universitas Pelita Bangsa**
