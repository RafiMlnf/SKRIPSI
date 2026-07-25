# Penjelasan Alur: Preprocessing → Evaluasi
### Analisis Sentimen Lagu "Arteri" — Panduan Belajar Mandiri

---

## 🧭 Gambaran Besar

Bayangkan kamu punya **1.531 komentar YouTube** yang berantakan — ada emoji, singkatan, bahasa gaul, typo. Komputer tidak bisa langsung mengerti teks seperti itu. Jadi, kita harus "membersihkan dan menerjemahkan" teks itu ke dalam bahasa yang bisa dipahami mesin, baru kemudian kita ajarkan mesin untuk mengenali sentimennya.

Inilah alurnya secara keseluruhan:

```
Komentar Mentah (1.531)
        ↓
  [PREPROCESSING]  ← "Bersihkan & rapikan teks"
        ↓
  Komentar Bersih (1.175)
        ↓
  [TF-IDF]         ← "Ubah teks → angka"
        ↓
  Matriks Angka (1.175 baris × 745 kolom)
        ↓
  [SVM Training]   ← "Latih model dari data"
        ↓
  Model Terlatih
        ↓
  [EVALUASI]       ← "Seberapa akurat?"
        ↓
  Akurasi 80,85%
```

---

## TAHAP 1 — Preprocessing Teks

> **Tujuan:** Mengubah teks kotor dan tidak terstruktur menjadi teks bersih yang bisa diolah komputer.

Ada **7 sub-tahap** preprocessing. Kita ikuti satu komentar nyata dari awal sampai akhir:

---

### Komentar awal (mentah):
```
"dulu sering denger lagu ini sampai suka bangett... 😭❤️ https://t.co/abc123"
```

---

### Sub-tahap 1 — Cleaning (Pembersihan)

**Apa yang dilakukan:**
Hapus semua hal yang tidak bermakna bagi analisis: URL, emoji, tanda baca berlebih, angka, simbol.

**Kenapa?**
Emoji `😭❤️` dan URL `https://...` tidak punya nilai linguistik. Kalau dibiarkan, mereka justru jadi "noise" yang membingungkan model.

```
Sebelum : "dulu sering denger lagu ini sampai suka bangett... 😭❤️ https://t.co/abc123"
Sesudah  : "dulu sering denger lagu ini sampai suka bangett"
```

---

### Sub-tahap 2 — Case Folding

**Apa yang dilakukan:**
Ubah semua huruf menjadi **huruf kecil (lowercase)**.

**Kenapa?**
Komputer menganggap `"Suka"`, `"SUKA"`, dan `"suka"` adalah tiga kata berbeda. Padahal artinya sama. Dengan menyeragamkan ke huruf kecil, komputer mengenalinya sebagai satu kata.

```
Sebelum : "dulu sering denger lagu ini sampai suka bangett"
Sesudah  : "dulu sering denger lagu ini sampai suka bangett"
           (di contoh ini sudah lowercase, tidak ada perubahan)
```

---

### Sub-tahap 3 — Tokenizing

**Apa yang dilakukan:**
Pecah kalimat menjadi **daftar kata-kata individual** (disebut *token*).

**Kenapa?**
Komputer tidak bekerja dengan kalimat utuh. Ia perlu memproses kata per kata.

```
Sebelum : "dulu sering denger lagu ini sampai suka bangett"
Sesudah  : ["dulu", "sering", "denger", "lagu", "ini", "sampai", "suka", "bangett"]
```

---

### Sub-tahap 4 — Normalisasi

**Apa yang dilakukan:**
Koreksi kata-kata slang/informal ke bentuk baku menggunakan **kamus slang**.

**Kenapa?**
Kata `"denger"` dan `"mendengar"` artinya sama, tapi komputer menganggapnya berbeda. Normalisasi menyatukan keduanya ke bentuk standar.

```
Sebelum : ["dulu", "sering", "denger", "lagu", "ini", "sampai", "suka", "bangett"]
Sesudah  : ["dulu", "sering", "dengar", "lagu", "ini", "sampai", "suka", "banget"]

Contoh mapping kamus slang:
  "denger"  → "dengar"
  "bangett" → "banget"
  "gak"     → "tidak"
  "bgt"     → "banget"
  "hrs"     → "harus"
```

---

### Sub-tahap 5 — Stopword Removal

**Apa yang dilakukan:**
Hapus kata-kata yang **tidak memiliki makna penting** dalam konteks analisis sentimen.

**Kenapa?**
Kata seperti `"ini"`, `"sampai"`, `"yang"`, `"di"` muncul di hampir semua kalimat dan tidak membantu model membedakan sentimen. Kalau dibiarkan, mereka justru "mengotori" fitur.

```
Sebelum : ["dulu", "sering", "dengar", "lagu", "ini", "sampai", "suka", "banget"]
Sesudah  : ["dulu", "sering", "dengar", "lagu", "suka", "banget"]

Kata yang dihapus: "ini", "sampai"
```

---

### Sub-tahap 6 — Stemming

**Apa yang dilakukan:**
Kembalikan setiap kata ke **bentuk dasarnya** menggunakan library **Sastrawi** (khusus bahasa Indonesia).

**Kenapa?**
`"mendengar"`, `"didengar"`, `"pendengar"` semuanya berakar dari `"dengar"`. Tanpa stemming, model memperlakukan ketiganya sebagai tiga fitur berbeda — padahal maknanya serumpun.

```
Sebelum : ["dulu", "sering", "dengar", "lagu", "suka", "banget"]
Sesudah  : ["dulu", "sering", "dengar", "lagu", "suka", "banget"]
           (di contoh ini sudah dalam bentuk dasar)

Contoh stemming lain:
  "menyukai"   → "suka"
  "mendengar"  → "dengar"
  "keindahan"  → "indah"
  "perasaan"   → "rasa"
```

---

### Sub-tahap 7 — Pelabelan Leksikon

**Apa yang dilakukan:**
Beri **label sentimen otomatis** (Positif / Netral / Negatif) pada setiap komentar menggunakan **kamus kata sentimen**.

**Kenapa?**
Model SVM adalah *supervised learning* — ia butuh data latihan yang sudah dilabeli. Kita tidak bisa melabeli 1.175 komentar satu per satu secara manual, jadi kita gunakan kamus leksikon sebagai acuan otomatis.

**Cara kerjanya:**
1. Setiap kata dalam komentar dicek di kamus leksikon
2. Kata positif → skor +1, kata negatif → skor -1, netral → 0
3. Skor dijumlahkan → jika total > 0: **Positif**, < 0: **Negatif**, = 0: **Netral**

```
Komentar  : ["dulu", "sering", "dengar", "lagu", "suka", "banget"]
Skor kata : dulu=0, sering=0, dengar=0, lagu=0, suka=+1, banget=0
Total skor: +1  →  Label: POSITIF ✅
```

**Hasil akhir preprocessing komentar ini:**

| | |
|---|---|
| **Teks awal** | `"dulu sering denger lagu ini sampai suka bangett... 😭❤️ https://..."` |
| **Teks bersih** | `dengar suka lagu` |
| **Label** | **Positif** |

---

## TAHAP 2 — Ekstraksi Fitur dengan TF-IDF

> **Tujuan:** Mengubah teks (yang sudah bersih) menjadi angka, karena algoritma Machine Learning hanya bisa bekerja dengan angka.

### Apa itu TF-IDF?

**TF-IDF** = *Term Frequency – Inverse Document Frequency*

Ini adalah cara menghitung **seberapa penting sebuah kata** dalam sebuah komentar, relatif terhadap seluruh dataset.

Ada dua komponen:

| Komponen | Arti | Logika |
|---|---|---|
| **TF** (Term Frequency) | Seberapa sering kata muncul di komentar ini | Kata yang sering muncul → lebih penting untuk komentar itu |
| **IDF** (Inverse Document Freq.) | Seberapa jarang kata muncul di seluruh dataset | Kata yang jarang di seluruh dataset → lebih unik & informatif |

**Skor TF-IDF = TF × IDF**

### Ilustrasi Sederhana:

Misalkan dataset kita hanya punya 4 komentar:

```
Komentar 1: "lagu ini bagus banget"
Komentar 2: "lagu ini biasa saja"
Komentar 3: "suka banget musiknya"
Komentar 4: "tidak suka lagunya"
```

Kata **"lagu"** muncul di banyak komentar → **IDF rendah** → skor kecil (kurang informatif)  
Kata **"bagus"** hanya muncul di 1 komentar → **IDF tinggi** → skor besar (sangat informatif untuk membedakan)

### Hasil TF-IDF dalam Penelitian Ini:

Dari 1.175 komentar yang sudah bersih, TF-IDF menghasilkan:

```
┌─────────────────────────────────────────────────────────────────┐
│  MATRIKS TF-IDF                                                 │
│                                                                 │
│  Setiap baris   = 1 komentar                                    │
│  Setiap kolom   = 1 kata unik dari seluruh dataset              │
│  Setiap nilai   = skor TF-IDF kata tersebut di komentar itu     │
│                                                                 │
│  Ukuran matriks: 1.175 baris × 745 kolom                        │
└─────────────────────────────────────────────────────────────────┘
```

**Artinya:** Setiap komentar kini direpresentasikan sebagai **deretan 745 angka**. Inilah "bahasa" yang bisa dipahami oleh SVM.

---

## TAHAP 3 — Pembagian Data

> **Tujuan:** Memisahkan data menjadi dua bagian — untuk belajar dan untuk ujian.

```
Total data bersih: 1.175 komentar
      │
      ├── 80% → Training Set (940 komentar)  ← SVM "belajar" dari ini
      │
      └── 20% → Testing Set  (235 komentar)  ← SVM "diuji" dari ini
```

**Kenapa dipisah?**
Kalau kita menguji model dengan data yang sama yang ia gunakan untuk belajar, hasilnya tidak valid — seperti ujian yang soalnya sudah bocor. Kita perlu data yang benar-benar "belum pernah dilihat" model untuk mengukur kemampuan aslinya.

---

## TAHAP 4 — Pelatihan Model SVM

> **Tujuan:** Mengajarkan mesin cara membedakan sentimen Positif, Netral, dan Negatif.

### Cara Kerja SVM (Versi Sederhana)

Bayangkan kamu punya dua kelompok titik di kertas — kelompok Positif (biru) dan kelompok Negatif (merah). SVM mencari **garis pemisah terbaik** antara dua kelompok itu.

```
  POSITIF  |    NEGATIF
    ●  ●   |   ○  ○
       ●   |  ○
    ●      |     ○  ○
           |
    ←——— GARIS PEMISAH (Hyperplane) ———→
```

Yang membuat SVM istimewa: ia tidak asal mencari garis, tapi mencari garis dengan **jarak terlebar** ke kedua kelompok (disebut *Maximal Margin*). Garis seperti ini lebih tahan terhadap data baru yang sedikit melenceng.

Dalam penelitian ini datanya bukan 2D, tapi **745 dimensi** (745 fitur TF-IDF). Konsepnya sama, hanya di ruang dimensi yang sangat tinggi.

### Mencari Pengaturan Terbaik — GridSearchCV

Sebelum melatih, kita perlu menentukan **hyperparameter** — pengaturan-pengaturan yang menentukan perilaku SVM. Di sini saya gunakan **GridSearchCV** untuk mencoba semua kombinasi secara otomatis:

| Parameter | Pilihan yang Dicoba | Terpilih |
|---|---|---|
| **Kernel** | Linear, RBF, Polynomial | **Linear** |
| **C (Regularisasi)** | 0.1, 1.0, 10.0 | **1.0** |
| **Strategi multi-kelas** | OvO, OvR | **OvO (One-vs-One)** |

**5-Fold Cross-Validation** artinya: data training dibagi 5 bagian, lalu dilatih 5 kali dengan bagian yang berbeda sebagai validasi. Ini memastikan hasil tidak kebetulan.

### Masalah: Kelas Negatif Terlalu Sedikit

```
Distribusi kelas Negatif:
  Netral   : 754 (64,2%)  ████████████████████████████████
  Positif  : 308 (26,2%)  █████████████
  Negatif  : 113  (9,6%)  █████
```

Dengan data seperi ini, model cenderung **"malas"** mengenali Negatif — karena jika asal prediksi semua sebagai Netral pun akurasinya sudah 64%. Model seperti ini tidak adil dan tidak berguna.

### Solusi: Random Oversampling

Kita **duplikasi secara acak** data kelas Negatif sampai jumlahnya seimbang dengan kelas lain. Ini dilakukan **hanya di data training**, bukan testing.

```
Sebelum Oversampling    Sesudah Oversampling
  Netral  : 754           Netral  : 754
  Positif : 308           Positif : 308
  Negatif : 113     →     Negatif : ~700+ (hasil duplikasi)
```

**Dampaknya:** Model tidak lagi bias. Recall kelas Negatif naik dari **31% → 65%**.

---

## TAHAP 5 — Evaluasi Model

> **Tujuan:** Mengukur seberapa baik model yang sudah dilatih dalam mengklasifikasikan data baru yang belum pernah ia lihat.

Model diuji menggunakan **235 data testing** yang tidak pernah disentuh selama pelatihan.

### Apa itu Confusion Matrix?

Ini adalah tabel yang menunjukkan **seberapa sering prediksi model benar atau salah**, dikelompokkan per kelas.

```
                 PREDIKSI MODEL
                 Negatif  Netral  Positif
               ┌─────────────────────────┐
AKTUAL Negatif │   14   │   6   │   3   │  → 23 data negatif asli
       Netral  │    5   │  134  │  12   │  → 151 data netral asli
       Positif │    3   │   16  │  42   │  → 61 data positif asli
               └─────────────────────────┘
```

**Cara membacanya:**
- Angka di **diagonal utama** (14, 134, 42) = prediksi **BENAR** ✅
- Angka di luar diagonal = prediksi **SALAH** ❌

**Kesalahan terbesar:** 16 komentar yang sebenarnya Positif diprediksi sebagai Netral.  
Ini masuk akal — komentar seperti *"lagunya relate banget sama hidupku"* bisa terkesan netral karena tidak ada kata sifat evaluatif eksplisit seperti "bagus" atau "keren".

---

### Metrik Evaluasi — Apa Artinya?

Dari confusion matrix di atas, kita hitung empat metrik utama:

---

#### Akurasi

> Dari semua komentar yang diuji, berapa persen yang diprediksi benar?

```
Akurasi = Jumlah prediksi benar / Total data uji
        = (14 + 134 + 42) / 235
        = 190 / 235
        = 80,85% ✅
```

---

#### Presisi

> Dari semua yang diprediksi "Positif", berapa persen yang memang benar Positif?

Ini mengukur **ketepatan** — jangan sampai model terlalu banyak salah tuduh.

```
Contoh untuk kelas Positif:
  Model memprediksi Positif sebanyak: 3 + 12 + 42 = 57 kali
  Yang benar-benar Positif          : 42
  Presisi Positif = 42 / 57 = 73,7%
```

---

#### Recall

> Dari semua komentar yang **sebenarnya** Positif, berapa persen yang berhasil ditemukan model?

Ini mengukur **kelengkapan** — jangan sampai model melewatkan terlalu banyak.

```
Contoh untuk kelas Positif:
  Total data yang sebenarnya Positif : 61
  Yang berhasil diprediksi benar     : 42
  Recall Positif = 42 / 61 = 68,9%
```

---

#### F1-Score

> Gabungan antara Presisi dan Recall — keseimbangan antara ketepatan dan kelengkapan.

```
F1-Score = 2 × (Presisi × Recall) / (Presisi + Recall)
```

Ini metrik yang paling sering dipakai karena tidak bisa "diakali" hanya dengan mengorbankan salah satu.

---

### Ringkasan Hasil Akhir

| Metrik | Nilai |
|---|:---:|
| **Akurasi** | **80,85%** |
| **Presisi (Weighted)** | **80,55%** |
| **Recall (Weighted)** | **80,85%** |
| **F1-Score (Weighted)** | **80,66%** |

**Per kelas:**

| Kelas | Precision | Recall | F1-Score | Jumlah Data Uji |
|---|:---:|:---:|:---:|:---:|
| Netral | 0,86 | 0,89 | **0,87** | 151 |
| Positif | 0,74 | 0,69 | **0,71** | 61 |
| Negatif | 0,64 | 0,61 | **0,62** | 23 |

**Mengapa Negatif paling rendah?**
Karena datanya paling sedikit (23 dari 235 data uji = hanya 9,8%). Meskipun sudah di-oversampling saat training, model masih kesulitan karena memang contoh komentar negatif yang beragam terbatas.

---

## Ringkasan: Alur Lengkap dalam Satu Pandangan

```
1.531 komentar mentah
    ↓  [Cleaning]        → buang URL, emoji, simbol
    ↓  [Case Folding]    → semua huruf kecil
    ↓  [Tokenizing]      → pecah jadi kata-kata
    ↓  [Normalisasi]     → koreksi slang ke baku
    ↓  [Stopword]        → buang kata tidak bermakna
    ↓  [Stemming]        → kembalikan ke kata dasar
    ↓  [Pelabelan]       → beri label Positif/Netral/Negatif
1.175 komentar bersih + berlabel
    ↓  [TF-IDF]          → ubah teks → matriks 1.175 × 745 angka
    ↓  [Split Data]      → 940 training / 235 testing
    ↓  [Oversampling]    → seimbangkan kelas Negatif
    ↓  [GridSearchCV]    → cari hyperparameter terbaik
    ↓  [SVM Training]    → latih model dengan 940 data
    ↓  [Evaluasi]        → uji dengan 235 data → Akurasi 80,85%
```

---

> 📌 **Catatan Belajar**
> 
> Bagian yang paling penting untuk dipahami saat presentasi adalah:
> - **Kenapa preprocessing perlu 7 tahap?** → Karena teks informal sangat kotor dan bervariasi.
> - **Kenapa TF-IDF bukan hanya menghitung frekuensi kata?** → Karena kata umum yang sering muncul justru tidak informatif.
> - **Kenapa perlu oversampling?** → Karena model akan bias ke kelas mayoritas jika data tidak seimbang.
> - **Kenapa F1-Score lebih representatif dari Akurasi saja?** → Karena akurasi bisa menipu jika data tidak seimbang.
