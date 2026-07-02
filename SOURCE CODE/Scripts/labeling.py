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
    
    # Ekspresi suka/cinta (bobot tinggi: 2-3)
    'suka': 2, 'cinta': 2, 'sayang': 2, 'love': 2, 'adore': 2,
    'favorit': 2, 'fav': 2, 'favorite': 2, 'favourite': 2,
    'bangga': 2, 'proud': 2,
    
    # Ekspresi kagum/apresiasi (bobot sedang: 1-2)
    'salut': 2, 'respect': 2, 'apresiasi': 2,
    'terima kasih': 2, 'makasih': 2, 'thanks': 2, 'thank': 2,
    'terimakasih': 2, 'trimakasih': 2,
    
    # Kualitas positif (bobot sedang: 1-2)
    'enak': 2, 'nikmat': 2, 'asyik': 2, 'asik': 2, 'seru': 2,
    'indah': 2, 'cantik': 2, 'cakep': 2, 'ganteng': 1,
    'manis': 1, 'lucu': 1, 'unik': 1, 'kreatif': 2,
    
    # Ekspresi antusias (bobot sedang: 1-2)
    'semangat': 2, 'excited': 2, 'antusias': 2,
    'wow': 1, 'wah': 1, 'kece': 2, 'cadas': 2,
    'josss': 2, 'jos': 2, 'gas': 1, 'mantul': 2,
    
    # Ekspresi puas/senang (bobot sedang: 1-2)
    'puas': 2, 'senang': 2, 'bahagia': 2, 'happy': 2,
    'gembira': 2, 'riang': 1, 'ceria': 1,
    
    # Musik spesifik positif (bobot tinggi: 2)
    'merinding': 2, 'goosebumps': 2, 'eargasm': 3,
    'catchy': 2, 'addictive': 2, 'candu': 2,
    'relate': 2, 'relatable': 2, 'ngena': 2, 'kena': 1,
    'deep': 2, 'dalem': 2, 'dalam': 1, 'bermakna': 2,
    'epic': 2, 'masterpiece': 3, 'karya': 1,
    
    # Intensifier positif
    'banget': 1, 'sekali': 1, 'sangat': 1, 'amat': 1,
    'bgt': 1, 'bngt': 1,
    
    # Ekspresi dukungan
    'sukses': 2, 'berhasil': 2, 'jaya': 1,
    'maju': 1, 'terus': 1, 'lanjut': 1,
    
    # Slang positif
    'sabi': 1, 'ok': 1, 'oke': 1, 'okey': 1,
    'nice': 2, 'cool': 2, 'lit': 2, 'fire': 2,
    'dope': 2, 'sick': 2, 'insane': 2,
    'underrated': 1,  # biasanya konteks positif
}

# Kata-kata negatif dengan bobot
NEGATIVE_WORDS = {
    # Ekspresi negatif umum (bobot tinggi: -2 sampai -3)
    'jelek': -2, 'buruk': -2, 'busuk': -2, 'sampah': -2, 'trash': -2,
    'parah': -1, 'hancur': -2, 'rusak': -2,
    'gagal': -2, 'fail': -2, 'failed': -2,
    
    # Ekspresi tidak suka (bobot tinggi: -2 sampai -3)
    'benci': -2, 'hate': -2, 'muak': -2, 'jijik': -2,
    'bosan': -1, 'boring': -1, 'membosankan': -2,
    
    # Ekspresi kecewa (bobot sedang: -1 sampai -2)
    'kecewa': -2, 'disappointed': -2, 'disappointing': -2,
    'menyesal': -2, 'nyesel': -2, 'sesal': -1,
    
    # Kualitas negatif (bobot sedang: -1 sampai -2)
    'aneh': -1, 'norak': -2, 'lebay': -1, 'alay': -1,
    'cringe': -2, 'awkward': -1,
    
    # Ekspresi marah/kesal (bobot sedang: -1 sampai -2)
    'kesal': -1, 'marah': -1, 'sebel': -1,
    'nyebelin': -2, 'menyebalkan': -2,
    
    # Ekspresi sedih (bobot rendah: -1, kadang netral)
    'sedih': -1, 'sad': -1, 'galau': -1,
    'nangis': -1, 'menangis': -1, 'crying': -1,
    
    # Musik spesifik negatif
    'berisik': -2, 'noise': -1, 'fals': -2,
    'jiplak': -2, 'plagiat': -2, 'copy': -1, 'tiruan': -1,
    
    # Kritik
    'overrated': -1, 'biasa': -1, 'standar': -1, 
    'mainstream': -1,
    
    # Kata kasar (bobot tinggi negatif, tapi kadang positif dalam slang)
    # Tidak dimasukkan karena ambigu dalam konteks Indonesia
    
    # Penolakan
    'tidak': -1, 'bukan': -1, 'jangan': -1,
    'gak': -1, 'ga': -1, 'enggak': -1,
}

# Kata-kata yang mengubah sentimen (negasi)
NEGATION_WORDS = {
    'tidak', 'bukan', 'jangan', 'belum', 'tanpa',
    'gak', 'ga', 'nggak', 'ngga', 'enggak', 'kagak',
    'tak', 'tiada', 'never', 'no', 'non',
}

# Kata intensifier yang memperkuat sentimen
INTENSIFIERS = {
    'sangat': 1.5, 'amat': 1.5, 'sekali': 1.5, 'banget': 1.5,
    'bgt': 1.5, 'bngt': 1.5, 'bener': 1.3, 'benar': 1.3,
    'paling': 1.5, 'ter': 1.3, 'super': 1.5, 'ultra': 1.5,
    'very': 1.5, 'really': 1.5, 'so': 1.3, 'too': 1.3,
    'extremely': 2.0, 'absolutely': 2.0,
}

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
    
    def calculate_sentiment_score(self, text):
        """
        Menghitung skor sentimen dari teks
        
        Returns:
            score (float): Skor sentimen
            positive_count (int): Jumlah kata positif
            negative_count (int): Jumlah kata negatif
            matched_words (list): Kata-kata yang cocok
        """
        if pd.isna(text) or not isinstance(text, str):
            return 0, 0, 0, []
        
        text = text.lower()
        words = text.split()
        
        score = 0
        positive_count = 0
        negative_count = 0
        matched_words = []
        
        # Cek negasi
        has_negation = any(word in self.negation_words for word in words)
        
        # Cek intensifier
        intensifier_mult = 1.0
        for word in words:
            if word in self.intensifiers:
                intensifier_mult = max(intensifier_mult, self.intensifiers[word])
        
        # Hitung skor dari kata positif
        for word, weight in self.positive_words.items():
            if word in text:
                count = text.count(word)
                word_score = weight * count * intensifier_mult
                
                # Jika ada negasi, balik sentimennya
                if has_negation:
                    word_score = -word_score * 0.5  # Negasi melemahkan
                
                score += word_score
                positive_count += count
                matched_words.append((word, weight, 'positive'))
        
        # Hitung skor dari kata negatif
        for word, weight in self.negative_words.items():
            if word in text:
                count = text.count(word)
                word_score = weight * count * intensifier_mult
                
                # Jika ada negasi, balik sentimennya
                if has_negation:
                    word_score = -word_score * 0.5
                
                score += word_score
                negative_count += count
                matched_words.append((word, weight, 'negative'))
        
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
        print("        Jalankan preprocessing.py terlebih dahulu!")
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
