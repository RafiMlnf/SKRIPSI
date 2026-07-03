# JAWABAN REVISI SIDANG SKRIPSI
**Judul:** Analisis Sentimen Pendengar Lagu "Arteri" oleh .FEAST Menggunakan Algoritma Support Vector Machine (SVM)
**Disusun oleh:** Rafi Maulana Firdaus — NIM 312210382
**Program Studi:** Teknik Informatika, Universitas Pelita Bangsa

---

## PERTANYAAN 1: Urgensi Penelitian

> *"Pelajari dan dalami urgensi dari penelitian yang dilakukan — apa urgensinya penelitian ini, siapa yang akan terbantu atau mendapatkan manfaat dari penelitian ini?"*

---

### Jawaban dengan Pendekatan Psikologis

Pertanyaan ini menyentuh inti dari "mengapa penelitian ini penting dilakukan sekarang?" Untuk menjawabnya secara menyeluruh, perlu dihubungkan tiga lapisan: **krisis psikologis yang sedang terjadi**, **kebutuhan akan data empiris**, dan **peran teknologi sebagai solusinya**.

---

#### A. Urgensi dari Sisi Psikologis: Fenomena Quarter-Life Crisis (QLC) pada Gen Z

Generasi Z (lahir 1997–2012) saat ini sedang berada pada fase paling rentan dalam siklus kehidupan mereka: fase dewasa awal (usia 20–29 tahun). Pada fase ini, mereka menghadapi tekanan psikologis berlapis yang dikenal sebagai **Quarter-Life Crisis (QLC)** — sebuah periode kebingungan, kecemasan eksistensial, dan krisis identitas yang ditandai oleh:

- **Tekanan karier dan finansial:** Tuntutan untuk "sukses" sebelum usia 30 tahun
- **Perbandingan sosial (Social Comparison Theory - Festinger, 1954):** Paparan media sosial yang terus-menerus menciptakan persepsi bahwa orang lain lebih sukses, lebih bahagia, atau lebih terarah
- **Disorientasi identitas:** Kebingungan antara ekspektasi keluarga, sosial, dan keinginan pribadi
- **Tekanan ekonomi:** Ketidakpastian lapangan kerja, biaya hidup tinggi, dan persaingan global

Fenomena ini bukan sekadar "drama remaja". Secara klinis, Quarter-Life Crisis diakui oleh psikolog Oliver Robinson (2011) sebagai fase perkembangan yang nyata dan berdampak serius pada kesehatan mental. Lebih dari **50% Gen Z Indonesia** dilaporkan mengalami kecemasan tinggi sebagai dampak langsung dari tekanan ini.

**Di sinilah lagu "Arteri" oleh .FEAST menjadi relevan secara psikologis.**

Lagu "Arteri" menggunakan metafora pembuluh darah yang tersumbat sebagai representasi dari seseorang yang terjebak dalam trauma masa lalu dan kesulitan untuk melanjutkan hidup — sebuah narasi yang sangat berresonansi dengan pengalaman batin Gen Z. Ribuan komentar yang muncul di kolom YouTube bukan sekadar pujian terhadap lagu; mereka adalah **ekspresi terbuka dari kondisi psikologis nyata** — tanda bahwa pendengar merasa divalidasi, dipahami, dan tidak sendirian dalam perjuangan mereka.

Dalam perspektif psikologi, ini adalah mekanisme koping berbasis **Emotional Regulation Through Music (ERTM)** — penggunaan musik sebagai alat untuk memproses emosi, mengurangi kortisol (hormon stres), dan meningkatkan dopamin. Namun, selama ini tidak ada yang membaca ribuan ekspresi tersebut secara sistematis dan terukur.

---

#### B. Urgensi dari Sisi Akademis dan Teknologi

**Masalah utama:** Terdapat lebih dari **1.449 komentar** (data mentah) dari pendengar lagu "Arteri" yang tersebar di YouTube. Ini adalah lautan data emosional yang belum pernah dianalisis secara empiris dan terstruktur. Metode analisis manual memiliki tiga kelemahan fatal:

1. **Tidak efisien** — tidak mungkin satu orang membaca dan mengklasifikasikan ribuan komentar secara akurat
2. **Bias subjektif** — penilaian manual dipengaruhi oleh persepsi peneliti, bukan data objektif
3. **Tidak skalabel** — tidak dapat diterapkan pada dataset yang lebih besar

Oleh karena itu, diperlukan metode komputasi otomatis: **Analisis Sentimen menggunakan Machine Learning (SVM)**.

---

#### C. Siapa yang Akan Terbantu?

| Pihak | Manfaat yang Diperoleh |
|-------|----------------------|
| **Grup Musik .FEAST** | Mendapatkan insights berbasis data mengenai bagaimana karya mereka diterima secara emosional oleh pendengar |
| **Industri Musik Digital** | Memperoleh referensi strategis tentang tren persepsi audiens terhadap isu sensitif seperti quarter-life crisis |
| **Praktisi Kesehatan Mental** | Mendapatkan gambaran kuantitatif tentang bagaimana Gen Z mengekspresikan distress emosional melalui komentar |
| **Peneliti Akademis** | Memperoleh kerangka kerja teknis dan dataset terstruktur sebagai baseline untuk penelitian NLP bahasa Indonesia |
| **Gen Z Secara Kolektif** | Penelitian ini mengakui dan memvalidasi secara akademis bahwa ekspresi emosional mereka di ruang digital adalah sinyal nyata yang layak diteliti |

---

#### D. Mengapa Saat Ini? (Urgensi Temporal)

1. **Lagu "Arteri" sedang dalam puncak relevansinya** — data komentar yang diambil per 18 Januari 2026 menangkap opini pada periode paling aktif
2. **Krisis literasi kesehatan mental** di Indonesia masih tinggi — penelitian yang menggunakan medium yang dekat dengan Gen Z (musik dan YouTube) dapat menjadi jembatan diskusi yang lebih luas
3. **Gap penelitian:** Belum ada penelitian yang secara spesifik menganalisis sentimen komentar pendengar musik indie Indonesia terhadap tema QLC menggunakan SVM

> **Kesimpulan:** Urgensi penelitian ini terletak pada persimpangan antara krisis psikologis nyata yang sedang dialami Gen Z, kebutuhan akan data empiris yang terukur, dan potensi teknologi machine learning untuk mengungkap pola emosi tersembunyi dalam lautan komentar digital. Penelitian ini bukan hanya soal teknis komputasi — ini adalah upaya untuk "mendengarkan" Gen Z melalui data.

---
---

## PERTANYAAN 2: Perbedaan Deep Learning dan Machine Learning

> *"Pelajari mengenai deep learning dan machine learning — berikan jawaban sesuai laporan yang ada"*

> PERHATIAN: Dosen penguji menggunakan pendekatan psikologis untuk menguji pemahaman materi. Perhatikan bahwa di Keywords Abstract Bahasa Inggris tertulis: "Deep Learning, Support Vector Machine (SVM)" — padahal SVM adalah Machine Learning, BUKAN Deep Learning. Ini kemungkinan adalah jebakan yang disengaja penguji untuk melihat apakah Anda menyadari inkonsistensi dalam laporan Anda sendiri.

---

### A. Hierarki yang Benar

```
Kecerdasan Buatan (Artificial Intelligence / AI)
    |
    +-- Machine Learning (ML)
            |
            +-- Supervised Learning   <-- Termasuk SVM
            +-- Unsupervised Learning
            +-- Reinforcement Learning
            +-- Deep Learning (DL)    <-- Subset dari ML
                    |
                    +-- CNN
                    +-- RNN / LSTM
                    +-- Transformer (BERT, GPT)
```

> **Deep Learning adalah BAGIAN dari Machine Learning, bukan sesuatu yang terpisah.**

---

### B. Perbedaan Fundamental

| Aspek | Machine Learning Tradisional | Deep Learning |
|-------|------------------------------|---------------|
| **Definisi** | Algoritma yang belajar dari data dengan fitur yang diekstraksi secara manual | Subset ML yang menggunakan jaringan saraf berlapis-lapis untuk belajar fitur otomatis |
| **Representasi Fitur** | Harus diekstraksi manual (contoh: TF-IDF) | Dipelajari otomatis dari data mentah |
| **Skala Data** | Efektif pada dataset kecil-menengah (ratusan–ribuan sampel) | Membutuhkan data sangat besar (>10.000–jutaan sampel) |
| **Kebutuhan Komputasi** | Ringan; bisa di laptop standar | Berat; memerlukan GPU/TPU khusus |
| **Interpretabilitas** | Relatif mudah dijelaskan | "Black box" — sulit dijelaskan intuitif |
| **Contoh Algoritma** | SVM, Naive Bayes, Decision Tree, Random Forest | LSTM, CNN, BERT, GPT, Transformer |
| **Waktu Pelatihan** | Cepat | Jauh lebih lama |

---

### C. Posisi SVM dalam Penelitian Ini

Sesuai yang tertulis dalam laporan (BAB II, Sub-bab 2.2.9), penelitian ini menggunakan **Support Vector Machine (SVM)**, yang merupakan algoritma **Machine Learning tradisional** (bukan Deep Learning). Pemilihan SVM didasarkan pada:

**1. Ukuran Dataset Terbatas**
Dataset penelitian ini berjumlah 1.146 komentar mentah (menjadi 821 setelah preprocessing). Pada skala ini, SVM terbukti lebih efektif dibanding Deep Learning yang memerlukan data puluhan ribu sampel untuk optimal.

**2. Kemampuan pada Data Berdimensi Tinggi**
Setelah ekstraksi fitur TF-IDF, setiap komentar direpresentasikan sebagai vektor berdimensi tinggi. SVM sangat tangguh dalam kondisi ini — "SVM memiliki performa klasifikasi yang robust, seringkali menghasilkan nilai error yang rendah pada data pelatihan dan pengujian bila disetel dengan parameter yang optimal" (referensi [6] dalam laporan).

**3. Efisiensi Komputasi**
Sesuai spesifikasi hardware penelitian (AMD Ryzen 5 2600, RAM 16GB — Sub-bab 3.7.1), SVM dapat berjalan optimal tanpa GPU khusus yang menjadi syarat mutlak Deep Learning.

**4. Dukungan Penelitian Terdahulu**
Dari 8 penelitian terdahulu (Tabel 2.1), semuanya menggunakan SVM pada dataset skala serupa dan menghasilkan akurasi 62%–98%. Ini menjadi baseline yang kuat.

---

### D. Koreksi Inkonsistensi dalam Laporan

> PERLU DIKOREKSI: Terdapat kesalahan pada bagian Keywords Abstract Bahasa Inggris yang mencantumkan "Deep Learning" sebagai keyword, padahal metode yang digunakan adalah Machine Learning (SVM).

Koreksi yang tepat:
```
SALAH: Keywords: Sentiment Analysis, Deep Learning, SVM, NLP, YouTube, .Feast
BENAR: Keywords: Sentiment Analysis, Machine Learning, SVM, TF-IDF, NLP, YouTube, .Feast
```

---
---

## PERTANYAAN 3: Apakah Platform YouTube Didominasi oleh Gen Z?

> *"Cari tau apakah platform YouTube didominasi oleh Gen Z — berikan jawaban sesuai laporan yang ada"*

---

### A. Posisi dalam Laporan

Dalam laporan (BAB II, Sub-bab 2.2.2 tentang YouTube), disebutkan bahwa YouTube adalah platform yang memungkinkan komentar publik dan menjadi sumber data tekstual yang sangat kaya. Namun laporan tidak secara eksplisit menyatakan YouTube "didominasi" oleh Gen Z. Pertanyaan penguji ini mengarahkan Anda untuk memverifikasi asumsi yang mungkin tersirat dalam pemilihan sumber data.

---

### B. Fakta Berdasarkan Data Statistik (2024–2025)

**Data Umum YouTube di Indonesia:**
- Total pengguna YouTube di Indonesia: ±151 juta orang (±53% dari total populasi) — Digital Report 2025
- YouTube adalah platform digital nomor satu di Indonesia dari segi jangkauan

**Data Spesifik Berdasarkan Generasi:**
- Tingkat penggunaan YouTube di kalangan Gen Z: **±78%**
- Tingkat penggunaan YouTube pada generasi Milenial, Gen X, dan Baby Boomer: **±84%** (YouGov Indonesia Media Consumption Report 2025)

---

### C. Analisis: YouTube = Lintas Generasi, Bukan Eksklusif Gen Z

YouTube bukan platform yang eksklusif didominasi Gen Z. YouTube justru memiliki ciri khas lintas generasi:

| Platform | Dominasi Usia |
|----------|--------------|
| TikTok | Gen Z (sangat dominan) |
| Instagram | Gen Z + Milenial |
| X (Twitter) | Gen Z + Milenial |
| YouTube | **Lintas generasi** (Milenial dan Gen X bahkan memiliki penetrasi lebih tinggi) |
| Facebook | Gen X + Baby Boomer |

---

### D. Mengapa Batasan Gen Z Tetap Relevan untuk Penelitian Ini?

Meskipun YouTube lintas generasi, relevansi Gen Z untuk penelitian ini tetap kuat karena:

**1. Tema Lagu Sangat Spesifik untuk Gen Z**
Lagu "Arteri" berbicara tentang Quarter-Life Crisis — fenomena yang secara definitif dialami kelompok usia 20–29 tahun (mayoritas adalah Gen Z pada 2026). Liriknya tentang kebingungan identitas dan pelepasan trauma adalah narasi yang sangat spesifik pada fase perkembangan ini.

**2. Pola Komentar Lagu Indie Cenderung dari Gen Z**
Komentar pada video musik indie/alternatif Indonesia seperti .FEAST secara demografis cenderung berasal dari pendengar muda (18–28 tahun / Gen Z) — berbeda dari komentar pada konten tutorial atau berita umum.

**3. Ekspresi Komentar yang Sangat Khas Gen Z**
Data mentah komentar menunjukkan penggunaan bahasa yang sangat khas Gen Z: slang, singkatan, emoji, dan referensi budaya pop spesifik pada generasi ini. Hal ini tercantum dalam laporan: "data komentar YouTube yang penuh dengan noise linguistik seperti sarkasme dan bahasa gaul" (BAB I, Sub-bab 1.1).

**4. YouTube sebagai Ruang Ekspresi QLC bagi Gen Z**
Penelitian menunjukkan kolom komentar YouTube pada konten bertema kesehatan mental menjadi "ruang komunitas" bagi Gen Z untuk berbagi cerita, mencari validasi, dan mengekspresikan pengalaman QLC secara terbuka.

---

### E. Kesimpulan

> YouTube secara keseluruhan adalah platform lintas generasi. Namun, **konteks spesifik lagu "Arteri"** — dengan tema Quarter-Life Crisis yang sangat identik dengan Gen Z, gaya indie yang menjadi identitas generasi ini, dan pola bahasa informal khas Gen Z dalam komentar — menjustifikasi fokus pada segmen pendengar Gen Z. Batasan bukan didasarkan pada dominasi Gen Z di YouTube secara platform, melainkan pada **relevansi tematik dan demografis** antara pesan lagu dengan kelompok pendengar yang paling beresonansi.

---
---

## PERTANYAAN 4: Untuk Siapa Penelitian Ini? (Mengapa Scope-nya Gen Z?)

> *"Kenapa scope-nya Gen Z saja, padahal bisa saja Milenial atau Gen X?"*

---

### A. Definisi Generasi dan Fase Kehidupan

| Generasi | Tahun Lahir | Usia pada 2026 | Fase Hidup |
|----------|------------|----------------|-----------|
| Baby Boomer | 1946–1964 | 62–80 tahun | Pensiun / Lansia |
| Gen X | 1965–1980 | 46–61 tahun | Karier mapan, keluarga stabil |
| Milenial | 1981–1996 | 30–45 tahun | Karier aktif, keluarga muda |
| **Gen Z** | **1997–2012** | **14–29 tahun** | **Dewasa awal, transisi hidup** |

---

### B. Justifikasi Ilmiah: Mengapa Gen Z?

#### 1. Kesesuaian Tematik: QLC adalah Fenomena Usia 20-an

Quarter-Life Crisis secara definitif adalah krisis yang dialami pada usia 20–29 tahun. Psikolog Oliver Robinson (2011) mendefinisikan QLC sebagai periode ketidakpastian dan pencarian makna pada fase transisi dari remaja ke dewasa penuh. Pada 2026, mayoritas individu di rentang usia ini adalah **Gen Z**.

Lagu "Arteri" secara eksplisit berbicara tentang:
- Kebingungan identitas di usia muda
- Kesulitan melepaskan trauma masa lalu
- Pencarian arah hidup di tengah ekspektasi sosial

Ini adalah narasi yang paling relevan bagi Gen Z yang saat ini menjalani fase tersebut.

#### 2. Milenial: Mengapa Tidak Menjadi Fokus Utama?

Milenial (usia 30–45 tahun pada 2026) memang pernah mengalami QLC, namun sebagian besar sudah melewati fase tersebut:
- Mereka umumnya sudah memiliki stabilitas karier dan keluarga
- Respons mereka terhadap lagu "Arteri" mungkin lebih bersifat nostalgia daripada identifikasi aktif
- Pola ekspresi komentar Milenial cenderung lebih formal dan berbeda dari dataset penelitian

#### 3. Gen X dan Baby Boomer: Relevansi Sangat Rendah

Gen X (46–61 tahun) dan Baby Boomer (62+ tahun) memiliki jarak kultural dan temporal yang sangat jauh dari tema lagu. Kemungkinan mereka menjadi pendengar aktif musik indie alternatif berbahasa Indonesia dengan tema QLC sangat kecil.

#### 4. Konteks Fenomenologi .FEAST

.FEAST secara historis memiliki basis fanbase yang sangat dominan Gen Z. Grup ini dengan genre rock alternatif/indie berhasil mengonsolidasi pendengar muda yang mencari narasi jujur tentang realitas kehidupan mereka. Koneksi paling kuat terjalin dengan generasi yang sedang dalam proses pembentukan identitas.

#### 5. Relevansi Bahasa dan Data

Komentar dalam dataset dipenuhi bahasa gaul, singkatan, dan ekspresi yang sangat spesifik pada Gen Z. Ini adalah karakteristik aktual dari dataset yang digunakan dan menjadi tantangan utama dalam preprocessing (Sub-bab 3.8.1 dalam laporan).

---

### C. Pengakuan Jujur: Batasan Penelitian

Perlu diakui secara jujur bahwa:
- Penelitian ini **tidak secara teknis memfilter komentar berdasarkan usia** pengguna YouTube (karena data usia tidak tersedia dari web scraping)
- Istilah "Gen Z" lebih mengacu pada **konteks demografis pendengar yang paling dominan** secara tematik dan kultural

Hal ini menjadi **saran untuk penelitian selanjutnya**: penggunaan data demografis yang lebih spesifik dapat menghasilkan penelitian yang lebih targeted.

---

### D. Kesimpulan

> Cakupan Gen Z dalam penelitian ini bukan pilihan sembarangan. Justifikasi utamanya:
> 1. **QLC adalah fenomena yang secara definitif dialami di usia 20-an** — rentang usia yang pada 2026 didominasi Gen Z
> 2. **Lagu "Arteri" memiliki narasi yang berresonansi paling kuat dengan Gen Z** — tema identitas, trauma, dan kebingungan arah hidup
> 3. **Basis pendengar .FEAST secara historis didominasi Gen Z**
> 4. **Karakteristik bahasa dalam dataset mencerminkan pola komunikasi Gen Z** — slang, emoji, dan ekspresi informal yang menjadi tantangan utama preprocessing
>
> Meski demikian, perlu diakui bahwa penelitian ini tidak menggunakan filter usia secara teknis — ini menjadi peluang pengembangan penelitian berikutnya.

---

## REFERENSI TAMBAHAN PENDUKUNG JAWABAN

1. Robinson, O. (2011). The Causes, Prevalence and Resolution of the "Quarter-Life Crisis". University of Greenwich.
2. YouGov Indonesia Media Consumption Report (2025). Platform Usage Across Generations in Indonesia.
3. Festinger, L. (1954). A Theory of Social Comparison Processes. Human Relations, 7(2), 117-140.
4. Wicaksono, B. & Nastiti, V.R.S. (2024). Analisis Sentimen dalam Opini Publik di Channel YouTube. Jurnal Algoritma, 21(2), 241-251.
5. Malikah, M. (2024). Dinamika Pengaruh Musik pada Kesejahteraan Psikologis Peserta Didik. Journal of Educational Research, 5(4), 5109-5118.
6. Surya, R., Reni, F., Bakhtiar, R., & Zaidan, A.A.D. (2025). Musik Sebagai Media Komunikasi Bagi Kesehatan Mental. Vol. 5, No. 2.

---

*Dokumen ini disusun sebagai bahan persiapan revisi sidang skripsi.*
*Revisi terakhir: Juli 2026*
