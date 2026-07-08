# -*- coding: utf-8 -*-
"""
=============================================================================
LABELING SENTIMEN KOMENTAR YOUTUBE
Untuk Analisis Sentimen Lagu "Arteri" - FEAST menggunakan LSTM
=============================================================================
Author: [Nama Anda]
Tanggal: Januari 2026
Keperluan: Seminar Proposal (SEMPRO)

Metode Labeling:
1. Lexicon-Based (Kamus Sentimen Bahasa Indonesia)
2. Rule-Based Enhancement
3. Manual Review untuk kasus ambigu

Output:
- Dataset berlabel dalam format CSV
- Laporan distribusi sentimen
=============================================================================
"""

import pandas as pd
import numpy as np
import re
import os
from collections import Counter
from datetime import datetime

# ============================================================================
# KONFIGURASI
# ============================================================================

class Config:
    """Konfigurasi labeling"""
    
    # File paths
    INPUT_FILE = "Data/comments_preprocessed.csv"
    OUTPUT_FILE = "Data/comments_labeled.csv"
    REPORT_FILE = "Data/labeling_report.txt"
    
    # Threshold untuk klasifikasi
    POSITIVE_THRESHOLD = 1      # Score >= 1 = Positif
    NEGATIVE_THRESHOLD = -1     # Score <= -1 = Negatif
    # Selain itu = Netral


# ============================================================================
# KAMUS SENTIMEN BAHASA INDONESIA
# ============================================================================

# Kata-kata positif dengan bobot
POSITIVE_WORDS = {
    # Ekspresi positif umum (bobot tinggi: 2-3)
    'bagus': 2, 'keren': 2, 'mantap': 2, 'gokil': 2, 'amazing': 2,
    'luar biasa': 3, 'sempurna': 3, 'terbaik': 3, 'best': 3, 'perfect': 3,
    'hebat': 2, 'dahsyat': 2, 'spektakuler': 3, 'fenomenal': 3,
    'top': 2, 'apik': 2, 'mantep': 2,  # slang positif

    # Ekspresi suka/cinta (bobot tinggi: 2-3)
    'suka': 2, 'cinta': 2, 'sayang': 2, 'love': 2, 'adore': 2,
    'favorit': 2, 'fav': 2, 'favorite': 2, 'favourite': 2,
    'bangga': 2, 'proud': 2,

    # Ekspresi kagum/apresiasi (bobot sedang: 1-2)
    'salut': 2, 'respect': 2, 'apresiasi': 2,
    'kagum': 2, 'takjub': 2, 'memukau': 2, 'terpesona': 2,  # ekspresi kekaguman
    'terima kasih': 2, 'makasih': 2, 'thanks': 2, 'thank': 2,
    'terimakasih': 2, 'trimakasih': 2,

    # Kualitas positif (bobot sedang: 1-2)
    'enak': 2, 'nikmat': 2, 'asyik': 2, 'asik': 2, 'seru': 2,
    'indah': 2, 'cantik': 2, 'cakep': 2, 'ganteng': 1,
    'manis': 1, 'lucu': 1, 'unik': 1, 'kreatif': 2,
    'nyaman': 2, 'adem': 2,   # nyaman & adem = positif untuk musik
    'baik': 1, 'okey': 1,     # afirmasi ringan
    'friendly': 2,            # dari data frekuensi

    # Ekspresi antusias (bobot sedang: 1-2)
    'semangat': 2, 'excited': 2, 'antusias': 2,
    'kece': 2, 'cadas': 2,
    'josss': 2, 'jos': 2, 'mantul': 2,

    # Ekspresi puas/senang (bobot sedang: 1-2)
    'puas': 2, 'senang': 2, 'bahagia': 2, 'happy': 2,
    'gembira': 2, 'riang': 1, 'ceria': 1,
    'selamat': 1,  # ucapan selamat = positif ringan

    # Musik spesifik positif (bobot sedang-tinggi: 2-3)
    'merinding': 2, 'goosebumps': 2, 'eargasm': 3,
    'catchy': 2, 'addictive': 2, 'candu': 2,
    'relate': 2, 'relatable': 2, 'ngena': 2, 'mengena': 2,
    'deep': 2, 'dalem': 2, 'bermakna': 2,
    'epic': 2, 'masterpiece': 3, 'karya': 1,
    'merdu': 2, 'syahdu': 2,
    'terharu': 2, 'haru': 2,          # terharu / haru = apresiasi emosional
    'menyentuh': 2, 'nyentuh': 2,    # menyentuh hati = apresiasi
    'vibes': 2,                       # vibes bagus = positif musik
    'mood': 1,                        # mood konteks positif di komentar musik
    'hits': 2, 'nonstop': 1,          # popular + diputar terus
    'nostalgia': 1,                   # kenangan = positif ringan
    'on repeat': 2,                   # diputar berulang = apresiasi

    # Slang positif (bobot sedang: 1-2)
    'nice': 2, 'cool': 2, 'lit': 2, 'fire': 2,
    'dope': 2, 'insane': 2,
    'underrated': 1,  # konteks positif: "underrated banget padahal bagus"
    'pecah': 3, 'goks': 2,            # slang penonton konser / live
    'nagih': 2, 'nyandu': 2, 'kecanduan': 2, # adiksi positif lagu
    'amazed': 2, 'deserve': 2,
    'gairah': 1,                      # tak ada gairah akan ternegasi
    'gaada obat': 2, 'ga ada obat': 2, # slang "tidak ada obat" = sangat bagus

    # Ekspresi dukungan
    'sukses': 2, 'berhasil': 2, 'jaya': 1,

    # Afirmasi
    'setuju': 2, 'bener': 1, 'benar': 1,
}

# Kata-kata negatif dengan bobot
NEGATIVE_WORDS = {
    # Ekspresi negatif umum (bobot tinggi: -2 sampai -3)
    'jelek': -2, 'buruk': -2, 'busuk': -2, 'sampah': -2, 'trash': -2,
    'hancur': -2, 'rusak': -2,
    'gagal': -2, 'fail': -2, 'failed': -2,

    # Ekspresi tidak suka (bobot tinggi: -2 sampai -3)
    'benci': -2, 'hate': -2, 'muak': -2, 'jijik': -2,
    'bosan': -2, 'boring': -2, 'membosankan': -2,

    # Ekspresi kecewa (bobot sedang: -1 sampai -2)
    'kecewa': -2, 'disappointed': -2, 'disappointing': -2,
    'menyesal': -2, 'nyesal': -2,

    # Kualitas negatif (bobot sedang: -1 sampai -2)
    'norak': -2, 'lebay': -1, 'alay': -1,
    'cringe': -2, 'awkward': -1,
    'sakit': -2, 'sakit hati': -3,   # jelas negatif
    'hambar': -1, 'datar': -1,       # kritik musik: hambar/datar = tidak berasa
    'monoton': -2,                   # monoton = membosankan

    # Ekspresi marah/kesal (bobot sedang: -1 sampai -2)
    'kesal': -1, 'marah': -1, 'sebel': -1,
    'nyebelin': -2, 'menyebalkan': -2,

    # Musik spesifik negatif
    'berisik': -2, 'fals': -2,
    'jiplak': -2, 'plagiat': -2,

    # Kritik negatif eksplisit (BUKAN kata ambigu)
    'overrated': -2,

    # Tambahan hasil analisis komentar Arteri
    'selingkuh': -2, 'selingkuhannya': -2,
    'mati rasa': -2,
    'celaka': -2, 'kecelakaan': -2,
    'fatherless': -1, 'problematik': -2,
    'stress': -1,
    'bodoh': -2, 'sebodoh': -2,
    'nihil': -1,
    'sehancur': -2,
    'gocek': -1, 'digocek': -2,
    'kena mental': -2,
    'pilu': -2,
    'capek': -2, 'cape': -2,
    'nyakitin': -2, 'tersakiti': -2, 'disakiti': -2,
    'sial': -2, 'sialnya': -2,
    'sok': -1, 'sokan': -1,
    'redflag': -2,
    'bangke': -2, 'bangsat': -3, 'bgst': -3, 'anjim': -2, 'kepharat': -3, 'keparat': -3,
    'pusing': -1,
    'gawat': -1, 'ghosting': -2, 'broken': -1, 'terendah': -2, 'rendah': -1, 'muntah': -2,
}

# -------------------------------------------------------------------
# KATA AMBIGU — konteks sangat menentukan
# -------------------------------------------------------------------

# Kata-kata yang DIMODIFIKASI oleh negasi
NEGATION_WORDS = {
    'tidak', 'bukan', 'jangan', 'belum', 'tanpa',
    'gak', 'ga', 'nggak', 'ngga', 'enggak', 'kagak',
    'tak', 'tiada', 'never', 'no', 'non',
    'kurang', 'gk', 'g', 'tdk',
}

# Kata intensifier yang memperkuat sentimen
INTENSIFIERS = {
    'sangat': 1.5, 'amat': 1.5, 'sekali': 1.5, 'banget': 1.5,
    'bgt': 1.5, 'bngt': 1.5,
    'paling': 1.5, 'super': 1.5, 'ultra': 1.5,
    'very': 1.5, 'really': 1.5,
    'extremely': 2.0, 'absolutely': 2.0,
    'bener': 1.3, 'benar': 1.3,
    # Slang intensifiers khas komentar YouTube / internet
    'anjir': 1.2, 'anjing': 1.2, 'bjir': 1.2, 'jir': 1.2,
}

# Aturan konteks MUSIK INDONESIA
MUSIC_CONTEXT_RULES = [
    # Pola sedih/haru/nangis sebagai APRESIASI musik → Positif
    (r'\b(bikin|buat|sampe|sampai|membuatku?)\b.{0,20}\b(sedih|nangis|menangis|baper|haru|terharu|galau)\b', +2),
    (r'\b(sedih|nangis|baper|haru|terharu|galau)\b.{0,20}\b(banget|bgt|bener|amat|sekali)\b', +2),
    # Pola menyentuh hati → Positif (apresiasi mendalam)
    (r'\b(menyayat|nyayat|nyentuh|menyentuh|menembus|ngena)\b.{0,20}\b(hati|jiwa|perasaan|kalbu|dada)\b', +2),
    # Pola vibes/mood positif → Positif
    (r'\b(vibes|mood)\b.{0,15}\b(bagus|keren|mantap|enak|positif|baik|sedap)\b', +2),
    # Pola "on repeat" / diputar terus → Positif kuat (indikator apresiasi nyata)
    (r'\b(on repeat|diulang|ulang ulang|ulang2|nonstop|loop)\b', +2),
    (r'\b(putar|play|dengerin)\b.{0,20}\b(terus|lagi|ulang|nonstop|mulu|melulu)\b', +2),
    # Pola "parah/gila + [kata positif]" → Positif (slang hiperbola)
    (r'\b(parah|gila|gilak)\b.{0,15}\b(bagus|keren|mantap|indah|merdu|enak|seru|kece)\b', +2),
    # Pola "parah/gila + [kata negatif]" → Negatif
    (r'\b(parah|gila)\b.{0,15}\b(buruk|jelek|ancur|hancur|busuk|hambar|datar)\b', -2),
    # Pola negasi langsung → Negatif
    (r'\b(tidak|ga|gak|nggak|enggak|tak|bukan)\s+(suka|cinta|mau|bagus|enak|keren|mantap|indah|merdu|nyaman)\b', -3),
    # Pola double negation → Positif
    (r'\b(tidak|ga|gak|nggak|enggak|tak)\s+(buruk|jelek|kecewa|bosan|boring|membosankan|hambar)\b', +2),
    # Kata aneh/unik dalam konteks positif
    (r'\b(aneh|unik)\b.{0,20}\b(tapi|namun|tetapi|tetap).{0,20}\b(bagus|keren|menarik|mantap|enak|suka|love)\b', +1),
    # Rindu/kangen terhadap lagu/artis → Positif ringan
    (r'\b(rindu|kangen|savefav|saved|bookmark|nostalgia)\b', +1),
    # Pola selamat + [pencapaian] → Positif
    (r'\b(selamat)\b.{0,20}\b(sukses|berhasil|rilis|debut|launch|album|lagu)\b', +2),
]

# Emotikon teks dan maknanya
EMOTICON_SENTIMENT = {
    # Positif
    ':)': 1, ':D': 2, 'xD': 2, ':p': 1, ';)': 1,
    '<3': 2, '^^': 1, ':3': 1,
    # Negatif
    ':(': -1, 'T_T': -1, '>_<': -1, '-_-': -1,
}


# ============================================================================
# KELAS SENTIMENT LABELER
# ============================================================================

class SentimentLabeler:
    """
    Kelas untuk melabeli sentimen komentar
    """
    
    def __init__(self, config=Config):
        self.config = config
        self.positive_words = POSITIVE_WORDS
        self.negative_words = NEGATIVE_WORDS
        self.negation_words = NEGATION_WORDS
        self.intensifiers = INTENSIFIERS
        self.stats = {
            'total': 0,
            'positive': 0,
            'negative': 0,
            'neutral': 0,
        }
    
    def _word_in_text(self, word, tokens):
        """Exact word match."""
        if ' ' in word:
            return word in ' '.join(tokens)
        return word in tokens

    def _find_positions(self, word, tokens):
        """Kembalikan semua indeks posisi kemunculan kata dalam tokens."""
        positions = []
        if ' ' in word:
            text_joined = ' '.join(tokens)
            start = 0
            while True:
                idx = text_joined.find(word, start)
                if idx == -1:
                    break
                positions.append(idx)
                start = idx + 1
            return positions
        return [i for i, t in enumerate(tokens) if t == word]

    def calculate_sentiment_score(self, text):
        """
        Menghitung skor sentimen dari teks dengan:
        1. Exact word-boundary matching (bukan substring)
        2. Negasi POSITIONAL — hanya membalik 1-3 kata berikutnya
        3. Intensifier lokal — hanya memperkuat kata terdekat
        4. Aturan konteks khusus musik Indonesia
        """
        if pd.isna(text) or not isinstance(text, str):
            return 0, 0, 0, []

        text_lower = text.lower()
        tokens = text_lower.split()
        n = len(tokens)

        score = 0.0
        positive_count = 0
        negative_count = 0
        matched_words = []

        # ------------------------------------------------------------------
        # STEP 1: Terapkan aturan konteks khusus musik lebih dulu
        # ------------------------------------------------------------------
        context_bonus = 0.0
        for pattern, bonus in MUSIC_CONTEXT_RULES:
            if re.search(pattern, text_lower):
                context_bonus += bonus
                matched_words.append((f'[RULE] {pattern[:40]}', bonus, 'context'))

        # ------------------------------------------------------------------
        # STEP 2: Hitung skor token per-token dengan negasi POSITIONAL
        # ------------------------------------------------------------------
        # Tandai token yang berada di bawah pengaruh negasi (window 3 kata setelah negasi)
        negated = [False] * n
        for i, token in enumerate(tokens):
            if token in self.negation_words:
                for j in range(i + 1, min(i + 4, n)):
                    negated[j] = True

        # Hitung intensifier per-posisi: memperkuat token berikutnya (1-2 posisi)
        intensifier_at = [1.0] * n
        for i, token in enumerate(tokens):
            if token in self.intensifiers:
                mult = self.intensifiers[token]
                for j in range(i + 1, min(i + 3, n)):
                    intensifier_at[j] = max(intensifier_at[j], mult)

        processed_positions = set()

        def score_word(word, base_weight, word_type):
            nonlocal score, positive_count, negative_count
            positions = self._find_positions(word, tokens)
            for pos in positions:
                if pos in processed_positions:
                    continue
                processed_positions.add(pos)

                mult = intensifier_at[pos] if pos < n else 1.0
                word_score = base_weight * mult

                is_neg = negated[pos] if pos < n else False
                if is_neg:
                    word_score = -word_score * 0.8  # negasi membalik dan melemahkan

                score += word_score
                if word_score > 0:
                    positive_count += 1
                else:
                    negative_count += 1
                matched_words.append((word, round(word_score, 2), word_type))

        for word, weight in self.positive_words.items():
            if self._word_in_text(word, tokens):
                score_word(word, weight, 'positive')

        for word, weight in self.negative_words.items():
            if self._word_in_text(word, tokens):
                score_word(word, weight, 'negative')

        # Tambahkan context bonus
        score += context_bonus

        return score, positive_count, negative_count, matched_words
    
    def classify_sentiment(self, score):
        """
        Mengklasifikasikan sentimen berdasarkan skor
        """
        if score >= self.config.POSITIVE_THRESHOLD:
            return 'positif'
        elif score <= self.config.NEGATIVE_THRESHOLD:
            return 'negatif'
        else:
            return 'netral'
    
    def label_single(self, text):
        """
        Melabeli satu teks
        """
        score, pos_count, neg_count, matched = self.calculate_sentiment_score(text)
        label = self.classify_sentiment(score)
        return label, score, pos_count, neg_count, matched
    
    def label_dataframe(self, df, text_column='text_processed'):
        """
        Melabeli seluruh DataFrame
        """
        print("\n" + "="*70)
        print("LABELING SENTIMEN KOMENTAR")
        print("="*70)
        
        self.stats['total'] = len(df)
        print(f"\n[INFO] Total komentar: {len(df)}")
        
        # Proses labeling
        print("\n[PROSES] Melabeli komentar...")
        
        results = df[text_column].apply(self.label_single)
        
        df['sentiment'] = results.apply(lambda x: x[0])
        df['sentiment_score'] = results.apply(lambda x: x[1])
        df['positive_word_count'] = results.apply(lambda x: x[2])
        df['negative_word_count'] = results.apply(lambda x: x[3])
        
        # Hitung statistik
        self.stats['positive'] = len(df[df['sentiment'] == 'positif'])
        self.stats['negative'] = len(df[df['sentiment'] == 'negatif'])
        self.stats['neutral'] = len(df[df['sentiment'] == 'netral'])
        
        # Konversi label ke numerik untuk LSTM
        label_map = {'positif': 2, 'netral': 1, 'negatif': 0}
        df['sentiment_label'] = df['sentiment'].map(label_map)
        
        # Print summary
        self._print_summary()
        
        return df
    
    def _print_summary(self):
        """Print ringkasan labeling"""
        print("\n" + "="*70)
        print("RINGKASAN LABELING")
        print("="*70)
        print(f"Total komentar       : {self.stats['total']}")
        print("-"*70)
        print(f"Positif              : {self.stats['positive']} ({self.stats['positive']/self.stats['total']*100:.1f}%)")
        print(f"Negatif              : {self.stats['negative']} ({self.stats['negative']/self.stats['total']*100:.1f}%)")
        print(f"Netral               : {self.stats['neutral']} ({self.stats['neutral']/self.stats['total']*100:.1f}%)")
        print("="*70)
    
    def generate_report(self, df, output_path):
        """Generate laporan labeling"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = []
        report.append("="*70)
        report.append("LAPORAN LABELING SENTIMEN")
        report.append("Analisis Sentimen Lagu 'Arteri' - FEAST")
        report.append("="*70)
        report.append(f"\nTanggal/Waktu: {timestamp}")
        report.append(f"File Input   : {self.config.INPUT_FILE}")
        report.append(f"File Output  : {self.config.OUTPUT_FILE}")
        
        report.append("\n" + "-"*70)
        report.append("METODE LABELING")
        report.append("-"*70)
        report.append("1. Lexicon-Based Sentiment Analysis")
        report.append("2. Kamus Sentimen Bahasa Indonesia (custom)")
        report.append("3. Negation Handling")
        report.append("4. Intensifier Recognition")
        
        report.append("\n" + "-"*70)
        report.append("KONFIGURASI THRESHOLD")
        report.append("-"*70)
        report.append(f"Positive Threshold   : >= {self.config.POSITIVE_THRESHOLD}")
        report.append(f"Negative Threshold   : <= {self.config.NEGATIVE_THRESHOLD}")
        report.append("Selain itu           : Netral")
        
        report.append("\n" + "-"*70)
        report.append("DISTRIBUSI SENTIMEN")
        report.append("-"*70)
        report.append(f"Total komentar       : {self.stats['total']}")
        report.append(f"Positif              : {self.stats['positive']} ({self.stats['positive']/self.stats['total']*100:.1f}%)")
        report.append(f"Negatif              : {self.stats['negative']} ({self.stats['negative']/self.stats['total']*100:.1f}%)")
        report.append(f"Netral               : {self.stats['neutral']} ({self.stats['neutral']/self.stats['total']*100:.1f}%)")
        
        report.append("\n" + "-"*70)
        report.append("STATISTIK SKOR SENTIMEN")
        report.append("-"*70)
        report.append(f"Rata-rata skor       : {df['sentiment_score'].mean():.2f}")
        report.append(f"Minimum skor         : {df['sentiment_score'].min():.2f}")
        report.append(f"Maximum skor         : {df['sentiment_score'].max():.2f}")
        report.append(f"Standar deviasi      : {df['sentiment_score'].std():.2f}")
        
        report.append("\n" + "-"*70)
        report.append("SAMPLE HASIL LABELING")
        report.append("-"*70)
        
        # Sample positif
        report.append("\n[POSITIF] Contoh:")
        positif_samples = df[df['sentiment'] == 'positif'].head(5)
        for idx, row in positif_samples.iterrows():
            report.append(f"  - {row['text_processed'][:60]}... (score: {row['sentiment_score']:.1f})")
        
        # Sample negatif
        report.append("\n[NEGATIF] Contoh:")
        negatif_samples = df[df['sentiment'] == 'negatif'].head(5)
        for idx, row in negatif_samples.iterrows():
            report.append(f"  - {row['text_processed'][:60]}... (score: {row['sentiment_score']:.1f})")
        
        # Sample netral
        report.append("\n[NETRAL] Contoh:")
        netral_samples = df[df['sentiment'] == 'netral'].head(5)
        for idx, row in netral_samples.iterrows():
            report.append(f"  - {row['text_processed'][:60]}... (score: {row['sentiment_score']:.1f})")
        
        report.append("\n" + "-"*70)
        report.append("LABEL ENCODING UNTUK LSTM")
        report.append("-"*70)
        report.append("positif = 2")
        report.append("netral  = 1")
        report.append("negatif = 0")
        
        report.append("\n" + "="*70)
        report.append("END OF REPORT")
        report.append("="*70)
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        print(f"\n[INFO] Laporan tersimpan: {output_path}")


# ============================================================================
# VISUALISASI (OPSIONAL)
# ============================================================================

def visualize_distribution(df, output_path='Data/sentiment_distribution.png'):
    """Visualisasi distribusi sentimen"""
    try:
        import matplotlib.pyplot as plt
        
        # Hitung distribusi
        sentiment_counts = df['sentiment'].value_counts()
        
        # Warna
        colors = {'positif': '#4CAF50', 'netral': '#FFC107', 'negatif': '#F44336'}
        
        # Buat pie chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Pie chart
        ax1.pie(
            sentiment_counts.values,
            labels=sentiment_counts.index,
            autopct='%1.1f%%',
            colors=[colors[s] for s in sentiment_counts.index],
            explode=[0.05] * len(sentiment_counts),
            shadow=True
        )
        ax1.set_title('Distribusi Sentimen\nKomentar "Arteri" - FEAST', fontsize=14, fontweight='bold')
        
        # Bar chart
        bars = ax2.bar(
            sentiment_counts.index,
            sentiment_counts.values,
            color=[colors[s] for s in sentiment_counts.index],
            edgecolor='black',
            linewidth=1.2
        )
        
        # Tambah label di atas bar
        for bar, count in zip(bars, sentiment_counts.values):
            ax2.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + 5,
                str(count),
                ha='center',
                fontsize=12,
                fontweight='bold'
            )
        
        ax2.set_xlabel('Sentimen', fontsize=12)
        ax2.set_ylabel('Jumlah Komentar', fontsize=12)
        ax2.set_title('Jumlah Komentar per Sentimen', fontsize=14, fontweight='bold')
        ax2.set_ylim(0, max(sentiment_counts.values) * 1.15)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"[INFO] Visualisasi tersimpan: {output_path}")
        
    except ImportError:
        print("[WARNING] matplotlib tidak terinstall. Visualisasi dilewati.")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main function"""
    print("""
    ╔═══════════════════════════════════════════════════════════════════╗
    ║     LABELING SENTIMEN KOMENTAR YOUTUBE                            ║
    ║     Analisis Sentimen Lagu "Arteri" - FEAST                       ║
    ║     Metode: Lexicon-Based Sentiment Analysis                      ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """)
    
    # Check input file
    if not os.path.exists(Config.INPUT_FILE):
        print(f"[ERROR] File tidak ditemukan: {Config.INPUT_FILE}")
        print("        Jalankan step_02_preprocessing.py terlebih dahulu!")
        return
    
    # Load data
    print(f"[INFO] Memuat data dari: {Config.INPUT_FILE}")
    df = pd.read_csv(Config.INPUT_FILE)
    print(f"[INFO] Jumlah komentar: {len(df)}")
    
    # Initialize labeler
    labeler = SentimentLabeler(Config)
    
    # Run labeling
    df_labeled = labeler.label_dataframe(df, text_column='text_processed')
    
    # Save labeled data
    output_columns = ['no', 'author', 'text', 'text_processed', 
                      'sentiment', 'sentiment_score', 'sentiment_label',
                      'positive_word_count', 'negative_word_count']
    output_columns = [col for col in output_columns if col in df_labeled.columns]
    
    df_labeled[output_columns].to_csv(Config.OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f"\n[INFO] Data berlabel tersimpan: {Config.OUTPUT_FILE}")
    
    # Generate report
    labeler.generate_report(df_labeled, Config.REPORT_FILE)
    
    # Visualize (optional)
    visualize_distribution(df_labeled)
    
    print("\n[SELESAI] Labeling sentimen berhasil!")
    print(f"          - Data berlabel : {Config.OUTPUT_FILE}")
    print(f"          - Laporan       : {Config.REPORT_FILE}")


if __name__ == "__main__":
    main()
