# -*- coding: utf-8 -*-
"""
=============================================================================
PREPROCESSING DATA KOMENTAR YOUTUBE
Untuk Analisis Sentimen Lagu "Arteri" - .FEAST menggunakan SVM
=============================================================================
Author: Rafi Maulana Firdaus (312210382)
Tanggal: Februari 2026
Keperluan: Seminar Proposal (SEMPRO)

Tahapan Preprocessing:
1. Case Folding (lowercase conversion)
2. Data Cleaning (URL, mention, hashtag, angka berlebih)
3. Emoji & Emoticon Removal
4. Duplicate Detection & Removal
5. Spam/Lyrics Detection & Removal
6. Punctuation Normalization
7. Slang Word Normalization (Bahasa Indonesia)
8. Stopwords Removal (Bahasa Indonesia)
9. Stemming (Sastrawi - Bahasa Indonesia)
10. Tokenization
11. Quality Filtering (min/max length)

Output:
- Dataset bersih dalam format CSV
- Laporan preprocessing dalam format TXT
=============================================================================
"""

import pandas as pd
import numpy as np
import re
import string
import os
from collections import Counter
from datetime import datetime

# Import modul Kamus Alay & NER Indonesia
try:
    from kamus_ner import (
        get_extended_slang_dict, 
        should_preserve_word, 
        is_named_entity,
        NAMA_TEMPAT_INDONESIA,
        PRESERVE_WORDS_EXTENDED
    )
    KAMUS_ALAY_AVAILABLE = True
    print("[INFO] Kamus Alay & NER Indonesia loaded successfully")
except ImportError:
    KAMUS_ALAY_AVAILABLE = False
    print("[WARNING] Modul kamus_ner tidak ditemukan, menggunakan kamus bawaan")

# ============================================================================
# KONFIGURASI
# ============================================================================

class Config:
    """Konfigurasi preprocessing"""
    
    # File paths - Dataset utama (full = MV + Lyric + Replies)
    INPUT_FILE = "Data/comments_arteri_full.csv"
    OUTPUT_FILE = "Data/comments_preprocessed.csv"
    REPORT_FILE = "Data/preprocessing_report.txt"
    
    # Quality thresholds
    MIN_WORDS = 2           # Minimum kata dalam komentar
    MAX_WORDS = 100         # Maximum kata (filter lirik/spam panjang)
    MIN_CHARS = 5           # Minimum karakter
    MAX_CHARS = 500         # Maximum karakter
    
    # Duplicate detection
    SIMILARITY_THRESHOLD = 0.85  # Threshold kemiripan untuk deteksi duplikat
    
    # Spam patterns (regex)
    SPAM_PATTERNS = [
        r'^\.+$',                    # Hanya titik
        r'^[^\w\s]+$',               # Hanya simbol
        r'^\d+$',                    # Hanya angka
        r'^.{1,3}$',                 # Terlalu pendek (1-3 karakter)
        r'^(p|pp|ppp|y|yy|yyy|ok|oke|okey|iy|iya)$',  # Spam singkat
    ]
    
    # Kata-kata sensitif yang akan dihapus dari teks
    SENSITIVE_WORDS = [
        'narkoba', 'narkotika', 'drugs', 'drug', 'ganja', 'sabu',
    ]


# ============================================================================
# KAMUS SLANG BAHASA INDONESIA
# ============================================================================


# ============================================================================
# KAMUS KHUSUS KONTEKS .FEAST / ARTERI
# ============================================================================
# Kata-kata yang TIDAK boleh diubah karena memiliki makna khusus dalam konteks ini

CONTEXT_SPECIFIC_WORDS = {
    # Personil .FEAST - JANGAN diubah
    'bas': 'baskara',           # Baskara Putra (vokalis), BUKAN alat musik bass
    'baskara': 'baskara',
    'feast': 'feast',
    '.feast': 'feast',
    
    # Referensi budaya populer yang sering muncul di komentar
    'mcr': 'my chemical romance',
    'my chemical romance': 'my chemical romance',
    
    # Film yang sering direferensikan (500 Days of Summer)
    # Tidak perlu diubah, biarkan apa adanya untuk analisis sentimen
    
    # Kata-kata terkait lagu Arteri
    'arteri': 'arteri',
    'quarter life crisis': 'quarter life crisis',
    'pondok indah': 'pondok indah',
}

# Kata-kata yang harus dipertahankan (tidak di-stem atau dihapus)
PRESERVE_WORDS = {
    'bas', 'baskara', 'feast', 'mcr', 'arteri',
    '500', 'days', 'summer',  # 500 Days of Summer (film)
    'my', 'chemical', 'romance',  # My Chemical Romance (band)
    'pondok', 'indah',  # Lokasi dalam lagu
    'quarter', 'life', 'crisis',  # Tema lagu
}

SLANG_DICT = {
    # Kata ganti & sapaan
    'gw': 'saya', 'gue': 'saya', 'gwa': 'saya', 'aku': 'saya', 'w': 'saya',
    'lo': 'kamu', 'lu': 'kamu', 'elu': 'kamu', 'loe': 'kamu',
    'dia': 'dia', 'dy': 'dia',
    'org': 'orang', 'orng': 'orang',
    'cwe': 'cewek', 'cwek': 'cewek', 'cewe': 'cewek',
    'cwo': 'cowok', 'cwok': 'cowok', 'cowo': 'cowok', 'co': 'cowok',
    
    # Kata kerja
    'gak': 'tidak', 'ga': 'tidak', 'g': 'tidak', 'nggak': 'tidak', 
    'ngga': 'tidak', 'gk': 'tidak', 'kagak': 'tidak', 'enggak': 'tidak',
    'tdk': 'tidak', 'tak': 'tidak',
    'udh': 'sudah', 'udah': 'sudah', 'dah': 'sudah', 'sdh': 'sudah',
    'blm': 'belum', 'blom': 'belum', 'belom': 'belum',
    'bgt': 'banget', 'bngt': 'banget', 'bngtt': 'banget', 'bgtt': 'banget',
    'bkn': 'bukan', 'bukn': 'bukan',
    'jd': 'jadi', 'jdi': 'jadi',
    'jgn': 'jangan', 'jngn': 'jangan',
    'krn': 'karena', 'krna': 'karena', 'karna': 'karena', 'gara2': 'gara-gara',
    'gmn': 'gimana', 'gmna': 'gimana',
    'dmn': 'dimana', 'dmna': 'dimana',
    'knp': 'kenapa', 'knpa': 'kenapa', 'napa': 'kenapa',
    'kyk': 'kayak', 'kyak': 'kayak', 'kek': 'kayak', 'kya': 'kayak',
    'yg': 'yang', 'yng': 'yang',
    'dgn': 'dengan', 'dg': 'dengan', 'sm': 'sama', 'ama': 'sama',
    'tp': 'tapi', 'tpi': 'tapi',
    'emg': 'memang', 'emng': 'memang', 'mmg': 'memang',
    'lgsg': 'langsung', 'lngsung': 'langsung',
    'bru': 'baru', 'br': 'baru',
    'skrg': 'sekarang', 'skrang': 'sekarang', 'skarang': 'sekarang',
    'td': 'tadi',
    'bs': 'bisa', 'bsa': 'bisa',
    'hrs': 'harus', 'hrus': 'harus',
    'msh': 'masih', 'msih': 'masih',
    'bnr': 'benar', 'bner': 'benar', 'bener': 'benar',
    'slh': 'salah',
    'trs': 'terus', 'trus': 'terus', 'truss': 'terus',
    'dlu': 'dulu', 'dl': 'dulu',
    'aj': 'saja', 'aja': 'saja', 'ajh': 'saja', 'ajaa': 'saja',
    'doang': 'saja', 'doank': 'saja', 'dong': 'saja',
    'sih': 'sih', 'si': 'sih', 'seh': 'sih',
    'bkl': 'bakal', 'bakl': 'bakal',
    'pny': 'punya', 'puny': 'punya',
    'ntn': 'nonton', 'nntn': 'nonton',
    'dger': 'dengar', 'denger': 'dengar', 'dgr': 'dengar', 'dengerin': 'dengar',
    'liat': 'lihat', 'lht': 'lihat',
    'nyoba': 'coba', 'cba': 'coba',
    'mna': 'mana', 'mn': 'mana',
    'cb': 'coba', 'cob': 'coba',
    'btw': 'ngomong-ngomong',
    'otw': 'dalam perjalanan',
    
    # Ekspresi
    'wkwk': 'haha', 'wkwkwk': 'haha', 'wkwkwkwk': 'haha', 'wkkwkw': 'haha',
    'haha': 'haha', 'hahaha': 'haha', 'hahahaha': 'haha',
    'hehe': 'haha', 'hehehe': 'haha',
    'xixi': 'haha', 'xixix': 'haha',
    'kwkw': 'haha', 'awkwkw': 'haha',
    'njir': 'gila', 'njay': 'gila', 'anjir': 'gila', 'anjay': 'gila',
    'anjing': 'gila', 'anying': 'gila', 'anjg': 'gila', 'ajg': 'gila',
    'asli': 'sungguh', 'asliii': 'sungguh',
    'parah': 'parah', 'prah': 'parah',
    'mantap': 'bagus', 'mantep': 'bagus', 'mntap': 'bagus', 'mantab': 'bagus',
    'keren': 'bagus', 'krn': 'bagus', 'kren': 'bagus',
    'gokil': 'bagus', 'goks': 'bagus',
    'cakep': 'bagus', 'ckep': 'bagus',
    'jelek': 'buruk', 'jlek': 'buruk', 'jlk': 'buruk',
    'ancur': 'hancur', 'ancr': 'hancur',
    'bgus': 'bagus', 'bgs': 'bagus',
    'enak': 'enak', 'enk': 'enak',
    
    # Kata sifat & lainnya
    'byk': 'banyak', 'bnyk': 'banyak',
    'skit': 'sakit', 'skt': 'sakit',
    'sedih': 'sedih', 'sdh': 'sedih',
    'sneng': 'senang', 'seneng': 'senang',
    'cpek': 'capek', 'cpk': 'capek', 'cape': 'capek',
    'sbr': 'sabar', 'sbr': 'sabar',
    'smngat': 'semangat', 'semangatt': 'semangat', 'smangat': 'semangat',
    
    # Kata benda
    'lgu': 'lagu', 'lg': 'lagu',
    'vdeo': 'video', 'vid': 'video', 'vidio': 'video',
    'mv': 'musik video', 'mvi': 'musik video',
    'tmen': 'teman', 'tmn': 'teman', 'temen': 'teman',
    'abg': 'remaja',
    'org': 'orang', 'orng': 'orang',
    'msik': 'musik', 'msk': 'musik',
    'bnd': 'band',
    
    # Angka slang
    'stgh': 'setengah', 'stngah': 'setengah',
    '1': 'satu', '2': 'dua', '3': 'tiga', '4': 'empat', '5': 'lima',
    '6': 'enam', '7': 'tujuh', '8': 'delapan', '9': 'sembilan', '10': 'sepuluh',
    
    # Intensifier
    'bgt': 'sangat', 'sangatt': 'sangat', 'sgt': 'sangat',
    'amat': 'sangat', 'bngt': 'sangat',
    'plg': 'paling', 'palng': 'paling',
    
    # Afirmasi
    'iy': 'iya', 'iyh': 'iya', 'iyo': 'iya', 'iyaa': 'iya',
    'yoi': 'iya', 'yoii': 'iya',
    'oke': 'baik', 'ok': 'baik', 'okey': 'baik', 'okee': 'baik',
    'sip': 'baik', 'siap': 'baik', 'siapp': 'baik',
    
    # Musik spesifik
    'reff': 'reff', 'ref': 'reff',
    'intro': 'intro', 'intr': 'intro',
    'outro': 'outro',
    'lirik': 'lirik', 'lrik': 'lirik', 'lyric': 'lirik', 'lyrics': 'lirik',
    'vokal': 'vokal', 'vocal': 'vokal',
    
    # Sentimen spesifik
    'suka': 'suka', 'suk': 'suka', 'sukak': 'suka',
    'benci': 'benci', 'bnci': 'benci',
    'cinta': 'cinta', 'cnta': 'cinta',
    'sayang': 'sayang', 'syg': 'sayang', 'syng': 'sayang',
    'galau': 'galau', 'glau': 'galau',
    'baper': 'baper', 'bper': 'baper',
    'relate': 'relate', 'relatable': 'relate',
    'ngena': 'mengena', 'kena': 'mengena',
}

# ============================================================================
# STOPWORDS BAHASA INDONESIA
# ============================================================================

STOPWORDS_ID = {
    # Kata sambung
    'dan', 'atau', 'tetapi', 'tapi', 'namun', 'melainkan', 'padahal', 'sedangkan',
    'sementara', 'sambil', 'seraya', 'serta', 'lalu', 'kemudian', 'selanjutnya',
    'sebelum', 'sesudah', 'ketika', 'waktu', 'saat', 'tatkala', 'sewaktu',
    'karena', 'sebab', 'oleh', 'akibat', 'sehingga', 'sampai', 'hingga',
    'supaya', 'agar', 'untuk', 'guna', 'demi', 'buat',
    'jika', 'kalau', 'bila', 'apabila', 'andai', 'seandainya', 'jikalau',
    'meskipun', 'walaupun', 'biarpun', 'kendatipun', 'sungguhpun', 'sekalipun',
    'bahwa', 'yang', 'apa', 'siapa', 'mana', 'kapan', 'bagaimana', 'mengapa',
    
    # Kata ganti
    'saya', 'aku', 'kamu', 'engkau', 'kau', 'dia', 'ia', 'beliau',
    'kami', 'kita', 'mereka', 'kalian', 'para',
    'ini', 'itu', 'tersebut', 'begini', 'begitu',
    '-nya', 'nya', 'ku', 'mu',
    
    # Kata depan
    'di', 'ke', 'dari', 'pada', 'daripada', 'kepada', 'oleh',
    'tentang', 'mengenai', 'akan', 'terhadap', 'dengan', 'tanpa',
    'antara', 'dalam', 'luar', 'atas', 'bawah', 'depan', 'belakang',
    'samping', 'kanan', 'kiri', 'sekitar', 'sekeliling',
    
    # Kata keterangan
    'sangat', 'amat', 'sekali', 'paling', 'lebih', 'kurang', 'agak', 'cukup',
    'hampir', 'nyaris', 'kira-kira', 'seolah-olah', 'seperti', 'bagaikan',
    'sudah', 'telah', 'pernah', 'belum', 'sedang', 'masih', 'akan', 'bakal',
    'harus', 'wajib', 'perlu', 'mesti', 'boleh', 'dapat', 'bisa', 'mampu',
    'mau', 'ingin', 'hendak',
    'tidak', 'bukan', 'jangan', 'belum', 'tanpa',
    'ya', 'tidak', 'bukan', 'iya', 'enggak',
    'saja', 'hanya', 'cuma', 'melulu', 'doang',
    'juga', 'pun', 'pula', 'bahkan', 'malah', 'malahan',
    'lagi', 'masih', 'terus', 'selalu', 'senantiasa', 'kerap', 'sering', 'kadang',
    'sekarang', 'kini', 'tadi', 'nanti', 'kemarin', 'besok', 'lusa',
    'dulu', 'dahulu', 'lampau', 'silam',
    'sini', 'situ', 'sana',
    
    # Kata bilangan
    'satu', 'dua', 'tiga', 'empat', 'lima', 'enam', 'tujuh', 'delapan', 'sembilan', 'sepuluh',
    'pertama', 'kedua', 'ketiga',
    'semua', 'seluruh', 'segenap', 'segala',
    'beberapa', 'banyak', 'sedikit', 'sebagian',
    
    # Kata tanya
    'apa', 'siapa', 'mana', 'kapan', 'bagaimana', 'mengapa', 'kenapa', 'berapa',
    
    # Partikel
    'lah', 'kah', 'tah', 'pun', 'per',
    'si', 'sang', 'sri', 'hang', 'dang',
    
    # Kata umum lainnya
    'ada', 'adalah', 'ialah', 'yaitu', 'yakni', 'merupakan',
    'bisa', 'dapat', 'mampu', 'sanggup',
    'seperti', 'sebagai', 'layaknya', 'ibarat',
    'mungkin', 'barangkali', 'boleh jadi',
    'tentu', 'pasti', 'niscaya',
    'memang', 'sungguh', 'benar', 'betul',
    'sebenarnya', 'sesungguhnya', 'sejatinya',
    'hal', 'perkara', 'soal', 'masalah',
    'cara', 'jalan', 'langkah',
    'tempat', 'lokasi', 'posisi',
    'waktu', 'masa', 'periode', 'zaman',
    'orang', 'manusia', 'individu',
    'barang', 'benda', 'objek',
    'uang', 'rupiah',
    
    # Filler words komentar YouTube
    'bang', 'mas', 'pak', 'bu', 'mbak', 'kak', 'min', 'gan', 'bro', 'sis', 'guys',
    'eh', 'ah', 'oh', 'ih', 'uh', 'wah', 'nah', 'duh', 'aduh', 'waduh',
    'dong', 'deh', 'sih', 'nih', 'tuh', 'gitu', 'gini',
    'hehe', 'haha', 'hihi', 'hoho', 'huhu', 'wkwk',
    'amin', 'aminn',
}


# ============================================================================
# KELAS PREPROCESSOR
# ============================================================================

class TextPreprocessor:
    """
    Kelas untuk preprocessing teks komentar YouTube
    """
    
    def __init__(self, config=Config):
        self.config = config
        self.stats = {
            'total_original': 0,
            'removed_empty': 0,
            'removed_duplicate': 0,
            'removed_lyrics': 0,
            'removed_spam': 0,
            'removed_too_short': 0,
            'removed_too_long': 0,
            'removed_non_indonesian': 0,
            'total_clean': 0,
        }
        
        # Load kamus slang - prioritaskan Kamus Alay (15.000+ kata) jika tersedia
        if KAMUS_ALAY_AVAILABLE:
            self.slang_dict = get_extended_slang_dict("Data/kamus_alay.csv")
            print(f"[INFO] Menggunakan Kamus Alay Extended: {len(self.slang_dict)} kata")
        else:
            self.slang_dict = SLANG_DICT  # Fallback ke kamus bawaan
            print(f"[INFO] Menggunakan kamus slang bawaan: {len(self.slang_dict)} kata")
        
        self.stopwords = STOPWORDS_ID
        self.stemmer = None
        
    def _init_stemmer(self):
        """Initialize Sastrawi stemmer"""
        try:
            from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
            factory = StemmerFactory()
            # Try both API versions
            try:
                self.stemmer = factory.createStemmer()
            except AttributeError:
                self.stemmer = factory.create_stemmer()
            return True
        except ImportError:
            print("[WARNING] Sastrawi tidak terinstall. Stemming dilewati.")
            print("          Install dengan: pip install Sastrawi")
            return False
    
    # ========================================================================
    # STEP 1: CASE FOLDING
    # ========================================================================
    
    def case_folding(self, text):
        """Mengubah semua huruf menjadi lowercase"""
        if pd.isna(text) or not isinstance(text, str):
            return ""
        return text.lower().strip()
    
    # ========================================================================
    # STEP 2: DATA CLEANING
    # ========================================================================
    
    def remove_urls(self, text):
        """Menghapus URL"""
        # HTTP/HTTPS URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        # Shortened URLs
        text = re.sub(r'bit\.ly/\S+|goo\.gl/\S+|t\.co/\S+', '', text)
        return text
    
    def remove_mentions(self, text):
        """Menghapus mentions (@username)"""
        return re.sub(r'@[\w]+', '', text)
    
    def remove_hashtags(self, text):
        """Menghapus hashtags (#hashtag)"""
        return re.sub(r'#[\w]+', '', text)
    
    def remove_timestamps(self, text):
        """Menghapus timestamp (1:23, 01:23:45)"""
        return re.sub(r'\b\d{1,2}:\d{2}(:\d{2})?\b', '', text)
    
    def remove_excessive_numbers(self, text):
        """Menghapus angka yang terlalu panjang (view count, dates, dll)"""
        # Hapus format view count: 6,634,732 atau 6.634.732
        text = re.sub(r'\b\d{1,3}([.,]\d{3})+\b', '', text)
        # Hapus tanggal format: 15/01/2026 atau 15-01-2026
        text = re.sub(r'\b\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b', '', text)
        return text
    
    def clean_data(self, text):
        """Gabungan semua data cleaning"""
        text = self.remove_urls(text)
        text = self.remove_mentions(text)
        text = self.remove_hashtags(text)
        text = self.remove_timestamps(text)
        text = self.remove_excessive_numbers(text)
        return text
    
    # ========================================================================
    # STEP 3: EMOJI & EMOTICON REMOVAL
    # ========================================================================
    
    def remove_emojis(self, text):
        """Menghapus emoji dan emoticon"""
        # Unicode emoji ranges
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F700-\U0001F77F"  # alchemical symbols
            "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
            "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
            "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
            "\U0001FA00-\U0001FA6F"  # Chess Symbols
            "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
            "\U00002702-\U000027B0"  # Dingbats
            "\U000024C2-\U0001F251"  # Enclosed characters
            "\U0001F1E0-\U0001F1FF"  # Flags
            "\U00002300-\U000023FF"  # Misc Technical
            "\U00002600-\U000026FF"  # Misc symbols
            "\U00002700-\U000027BF"  # Dingbats
            "\U0000FE00-\U0000FE0F"  # Variation Selectors
            "\U0001F000-\U0001F02F"  # Mahjong Tiles
            "\U0001F0A0-\U0001F0FF"  # Playing Cards
            "]+", 
            flags=re.UNICODE
        )
        text = emoji_pattern.sub('', text)
        
        # Text emoticons
        emoticons = [
            r':\)', r':\(', r':D', r':P', r':p', r';D', r';\)', r':O', r':o',
            r'<3', r'</3', r':3', r'xD', r'XD', r':-\)', r':-\(',
            r'\^\^', r'-_-', r'T_T', r'>_<', r'O_O', r'o_o',
            r'uwu', r'owo', r'UwU', r'OwO',
        ]
        for emot in emoticons:
            text = re.sub(emot, '', text, flags=re.IGNORECASE)
        
        return text
    
    # ========================================================================
    # STEP 4: SPAM & LYRICS DETECTION
    # ========================================================================
    
    def is_lyrics(self, text):
        """Deteksi apakah komentar adalah lirik lagu"""
        # Keywords lirik
        lyrics_keywords = [
            'verse', 'chorus', 'reff', 'bridge', 'outro', 'intro',
            'pre-chorus', 'interlude', 'bait', 'lirik'
        ]
        
        text_lower = text.lower()
        
        # Cek keywords
        keyword_count = sum(1 for kw in lyrics_keywords if kw in text_lower)
        if keyword_count >= 2:
            return True
        
        # Cek pola lirik (banyak newline atau baris berulang)
        lines = text.split('\n')
        if len(lines) > 10:
            return True
        
        # Cek frase dari lagu Arteri
        arteri_phrases = [
            'telanjang ku telanjang',
            'menyicipi dunia',
            'selamat datang di dua puluh',
            'menangisku di pundakmu',
            'muntahkan semua pilu',
            'kau hanya trauma',
            'meluncur di arteri',
            'setetes bahagia',
        ]
        
        phrase_count = sum(1 for phrase in arteri_phrases if phrase in text_lower)
        if phrase_count >= 3:
            return True
        
        return False
    
    def is_spam(self, text):
        """Deteksi apakah komentar adalah spam"""
        text_clean = text.strip()
        
        # Cek pola spam
        for pattern in self.config.SPAM_PATTERNS:
            if re.match(pattern, text_clean, re.IGNORECASE):
                return True
        
        # Cek karakter berulang berlebihan
        if re.search(r'(.)\1{4,}', text_clean):  # 5+ karakter sama berturut-turut
            return True
        
        return False
    
    # ========================================================================
    # STEP 5: PUNCTUATION NORMALIZATION
    # ========================================================================
    
    def normalize_punctuation(self, text):
        """Normalisasi tanda baca"""
        # Hapus tanda baca berulang
        text = re.sub(r'([!?.])\1+', r'\1', text)
        
        # Hapus semua tanda baca kecuali yang diperlukan
        allowed_punct = set('.,!?\'"-')
        text = ''.join(
            char if char not in string.punctuation or char in allowed_punct 
            else ' ' 
            for char in text
        )
        
        # Hapus whitespace berlebih
        text = ' '.join(text.split())
        
        return text
    
    # ========================================================================
    # STEP 6: SLANG WORD NORMALIZATION
    # ========================================================================
    
    def normalize_slang(self, text):
        """Normalisasi kata slang ke bentuk baku"""
        words = text.split()
        normalized = []
        
        for word in words:
            word_lower = word.lower()
            
            # Prioritas 1: Cek kata-kata konteks khusus (bas, mcr, dll)
            if word_lower in CONTEXT_SPECIFIC_WORDS:
                normalized.append(CONTEXT_SPECIFIC_WORDS[word_lower])
            # Prioritas 2: Cek di kamus slang
            elif word_lower in self.slang_dict:
                normalized.append(self.slang_dict[word_lower])
            else:
                normalized.append(word)
        
        return ' '.join(normalized)
    
    # ========================================================================
    # STEP 7: STOPWORDS REMOVAL
    # ========================================================================
    
    def remove_stopwords(self, text):
        """Menghapus stopwords (tapi pertahankan kata konteks khusus dan nama tempat)"""
        words = text.split()
        filtered = []
        for word in words:
            word_lower = word.lower()
            # Pertahankan kata-kata penting (konteks .FEAST/Arteri + nama tempat Indonesia)
            if KAMUS_ALAY_AVAILABLE and should_preserve_word(word_lower):
                filtered.append(word)
            elif word_lower in PRESERVE_WORDS:
                filtered.append(word)
            elif word_lower not in self.stopwords:
                filtered.append(word)
        return ' '.join(filtered)
    
    # ========================================================================
    # STEP 8: STEMMING
    # ========================================================================
    
    def stem_text(self, text):
        """Stemming menggunakan Sastrawi (tapi pertahankan kata konteks khusus dan nama tempat)"""
        if self.stemmer is None:
            return text
        
        words = text.split()
        stemmed = []
        for word in words:
            word_lower = word.lower()
            # Jangan stem: kata konteks .FEAST/Arteri + nama tempat Indonesia
            if KAMUS_ALAY_AVAILABLE and should_preserve_word(word_lower):
                stemmed.append(word)
            elif word_lower in PRESERVE_WORDS:
                stemmed.append(word)
            else:
                stemmed.append(self.stemmer.stem(word))
        return ' '.join(stemmed)
    
    # ========================================================================
    # STEP 9: FINAL CLEANING
    # ========================================================================
    
    def final_clean(self, text):
        """Pembersihan akhir"""
        # Hapus karakter non-alfanumerik kecuali spasi
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        
        # Hapus single characters
        text = ' '.join([word for word in text.split() if len(word) > 1])
        
        # Hapus kata-kata sensitif
        words = text.split()
        words = [w for w in words if w.lower() not in self.config.SENSITIVE_WORDS]
        text = ' '.join(words)
        
        # Hapus whitespace berlebih
        text = ' '.join(text.split())
        
        return text.strip()
    
    # ========================================================================
    # DUPLICATE DETECTION
    # ========================================================================
    
    def detect_duplicates(self, df, column='text_processed'):
        """Deteksi dan tandai duplikat"""
        # Exact duplicates
        df['is_duplicate'] = df.duplicated(subset=[column], keep='first')
        
        return df
    
    # ========================================================================
    # QUALITY FILTERING
    # ========================================================================
    
    def quality_filter(self, text):
        """Filter berdasarkan kualitas teks"""
        if not text or len(text.strip()) == 0:
            return False, 'empty'
        
        words = text.split()
        word_count = len(words)
        char_count = len(text)
        
        if word_count < self.config.MIN_WORDS:
            return False, 'too_short'
        
        if word_count > self.config.MAX_WORDS:
            return False, 'too_long'
        
        if char_count < self.config.MIN_CHARS:
            return False, 'too_short'
        
        if char_count > self.config.MAX_CHARS:
            return False, 'too_long'
        
        return True, 'ok'
    
    # ========================================================================
    # MAIN PREPROCESSING PIPELINE
    # ========================================================================
    
    def preprocess_single(self, text, apply_stemming=True, apply_stopwords=True):
        """
        Pipeline preprocessing untuk satu teks
        """
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # Step 1: Case folding
        text = self.case_folding(text)
        
        # Step 2: Data cleaning
        text = self.clean_data(text)
        
        # Step 3: Emoji removal
        text = self.remove_emojis(text)
        
        # Step 4: Punctuation normalization
        text = self.normalize_punctuation(text)
        
        # Step 5: Slang normalization
        text = self.normalize_slang(text)
        
        # Step 6: Stopwords removal (optional)
        if apply_stopwords:
            text = self.remove_stopwords(text)
        
        # Step 7: Stemming (optional)
        if apply_stemming and self.stemmer:
            text = self.stem_text(text)
        
        # Step 8: Final cleaning
        text = self.final_clean(text)
        
        return text
    
    def preprocess_dataframe(self, df, text_column='text', apply_stemming=True, apply_stopwords=True):
        """
        Pipeline preprocessing untuk DataFrame
        """
        print("\n" + "="*70)
        print("PREPROCESSING DATA KOMENTAR YOUTUBE")
        print("="*70)
        
        self.stats['total_original'] = len(df)
        print(f"\n[INFO] Total komentar awal: {len(df)}")
        
        # Initialize stemmer
        if apply_stemming:
            self._init_stemmer()
        
        # Create processed dataframe
        df_processed = df.copy()
        
        # Step 1-3: Basic preprocessing
        print("\n[STEP 1-3] Case folding, cleaning, emoji removal...")
        df_processed['text_clean'] = df_processed[text_column].apply(
            lambda x: self.remove_emojis(self.clean_data(self.case_folding(x)))
        )
        
        # Step 4: Remove empty
        print("[STEP 4] Menghapus komentar kosong...")
        mask_empty = df_processed['text_clean'].str.strip().str.len() == 0
        self.stats['removed_empty'] = mask_empty.sum()
        df_processed = df_processed[~mask_empty]
        print(f"         Dihapus: {self.stats['removed_empty']} komentar kosong")
        
        # Step 5: Remove lyrics
        print("[STEP 5] Mendeteksi dan menghapus lirik...")
        mask_lyrics = df_processed['text_clean'].apply(self.is_lyrics)
        self.stats['removed_lyrics'] = mask_lyrics.sum()
        df_processed = df_processed[~mask_lyrics]
        print(f"         Dihapus: {self.stats['removed_lyrics']} komentar lirik")
        
        # Step 6: Remove spam
        print("[STEP 6] Mendeteksi dan menghapus spam...")
        mask_spam = df_processed['text_clean'].apply(self.is_spam)
        self.stats['removed_spam'] = mask_spam.sum()
        df_processed = df_processed[~mask_spam]
        print(f"         Dihapus: {self.stats['removed_spam']} komentar spam")
        
        # Step 7: Full preprocessing
        print("[STEP 7] Menjalankan preprocessing lengkap...")
        print("         - Normalisasi tanda baca")
        print("         - Normalisasi kata slang")
        if apply_stopwords:
            print("         - Penghapusan stopwords")
        if apply_stemming and self.stemmer:
            print("         - Stemming (Sastrawi)")
        
        df_processed['text_processed'] = df_processed['text_clean'].apply(
            lambda x: self.preprocess_single(x, apply_stemming, apply_stopwords)
        )
        
        # Step 8: Quality filter
        print("[STEP 8] Quality filtering...")
        quality_results = df_processed['text_processed'].apply(self.quality_filter)
        df_processed['quality_ok'] = quality_results.apply(lambda x: x[0])
        df_processed['quality_reason'] = quality_results.apply(lambda x: x[1])
        
        # Count removals by reason
        too_short = (df_processed['quality_reason'] == 'too_short').sum()
        too_long = (df_processed['quality_reason'] == 'too_long').sum()
        self.stats['removed_too_short'] = too_short
        self.stats['removed_too_long'] = too_long
        
        df_processed = df_processed[df_processed['quality_ok']]
        print(f"         Dihapus: {too_short} komentar terlalu pendek")
        print(f"         Dihapus: {too_long} komentar terlalu panjang")
        
        # Step 9: Remove duplicates
        print("[STEP 9] Menghapus duplikat...")
        df_processed = self.detect_duplicates(df_processed, 'text_processed')
        self.stats['removed_duplicate'] = df_processed['is_duplicate'].sum()
        df_processed = df_processed[~df_processed['is_duplicate']]
        print(f"         Dihapus: {self.stats['removed_duplicate']} komentar duplikat")
        
        # Final cleanup
        df_processed = df_processed.drop(columns=['quality_ok', 'quality_reason', 'is_duplicate'], errors='ignore')
        
        # Reset index
        df_processed = df_processed.reset_index(drop=True)
        df_processed['no'] = range(1, len(df_processed) + 1)
        
        self.stats['total_clean'] = len(df_processed)
        
        # Print summary
        self._print_summary()
        
        return df_processed
    
    def _print_summary(self):
        """Print ringkasan preprocessing"""
        print("\n" + "="*70)
        print("RINGKASAN PREPROCESSING")
        print("="*70)
        print(f"Total komentar awal      : {self.stats['total_original']}")
        print(f"Dihapus - Kosong         : {self.stats['removed_empty']}")
        print(f"Dihapus - Lirik          : {self.stats['removed_lyrics']}")
        print(f"Dihapus - Spam           : {self.stats['removed_spam']}")
        print(f"Dihapus - Terlalu pendek : {self.stats['removed_too_short']}")
        print(f"Dihapus - Terlalu panjang: {self.stats['removed_too_long']}")
        print(f"Dihapus - Duplikat       : {self.stats['removed_duplicate']}")
        print("-"*70)
        total_removed = (
            self.stats['removed_empty'] + 
            self.stats['removed_lyrics'] + 
            self.stats['removed_spam'] +
            self.stats['removed_too_short'] + 
            self.stats['removed_too_long'] +
            self.stats['removed_duplicate']
        )
        print(f"Total dihapus            : {total_removed}")
        print(f"Total komentar bersih    : {self.stats['total_clean']}")
        print(f"Persentase tersisa       : {self.stats['total_clean']/self.stats['total_original']*100:.1f}%")
        print("="*70)
    
    def generate_report(self, df_original, df_processed, output_path):
        """Generate laporan preprocessing dalam format TXT"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = []
        report.append("="*70)
        report.append("LAPORAN PREPROCESSING DATA KOMENTAR YOUTUBE")
        report.append("Analisis Sentimen Lagu 'Arteri' - FEAST")
        report.append("="*70)
        report.append(f"\nTanggal/Waktu: {timestamp}")
        report.append(f"File Input   : {self.config.INPUT_FILE}")
        report.append(f"File Output  : {self.config.OUTPUT_FILE}")
        
        report.append("\n" + "-"*70)
        report.append("KONFIGURASI PREPROCESSING")
        report.append("-"*70)
        report.append(f"Minimum kata per komentar    : {self.config.MIN_WORDS}")
        report.append(f"Maximum kata per komentar    : {self.config.MAX_WORDS}")
        report.append(f"Minimum karakter per komentar: {self.config.MIN_CHARS}")
        report.append(f"Maximum karakter per komentar: {self.config.MAX_CHARS}")
        
        report.append("\n" + "-"*70)
        report.append("TAHAPAN PREPROCESSING")
        report.append("-"*70)
        report.append("1. Case Folding (lowercase conversion)")
        report.append("2. Data Cleaning (URL, mention, hashtag, timestamp, angka)")
        report.append("3. Emoji & Emoticon Removal")
        report.append("4. Spam Detection & Removal")
        report.append("5. Lyrics Detection & Removal")
        report.append("6. Punctuation Normalization")
        report.append("7. Slang Word Normalization (Bahasa Indonesia)")
        report.append("8. Stopwords Removal (Bahasa Indonesia)")
        report.append("9. Stemming (Sastrawi)")
        report.append("10. Quality Filtering (min/max length)")
        report.append("11. Duplicate Removal")
        
        report.append("\n" + "-"*70)
        report.append("STATISTIK PREPROCESSING")
        report.append("-"*70)
        report.append(f"Total komentar awal      : {self.stats['total_original']}")
        report.append(f"Dihapus - Kosong         : {self.stats['removed_empty']}")
        report.append(f"Dihapus - Lirik          : {self.stats['removed_lyrics']}")
        report.append(f"Dihapus - Spam           : {self.stats['removed_spam']}")
        report.append(f"Dihapus - Terlalu pendek : {self.stats['removed_too_short']}")
        report.append(f"Dihapus - Terlalu panjang: {self.stats['removed_too_long']}")
        report.append(f"Dihapus - Duplikat       : {self.stats['removed_duplicate']}")
        
        total_removed = sum([
            self.stats['removed_empty'],
            self.stats['removed_lyrics'],
            self.stats['removed_spam'],
            self.stats['removed_too_short'],
            self.stats['removed_too_long'],
            self.stats['removed_duplicate']
        ])
        
        report.append("-"*70)
        report.append(f"Total dihapus            : {total_removed}")
        report.append(f"Total komentar bersih    : {self.stats['total_clean']}")
        report.append(f"Persentase tersisa       : {self.stats['total_clean']/self.stats['total_original']*100:.1f}%")
        
        report.append("\n" + "-"*70)
        report.append("STATISTIK TEKS HASIL PREPROCESSING")
        report.append("-"*70)
        
        # Word count statistics
        df_processed['word_count'] = df_processed['text_processed'].apply(lambda x: len(x.split()))
        df_processed['char_count'] = df_processed['text_processed'].apply(len)
        
        report.append(f"Rata-rata jumlah kata    : {df_processed['word_count'].mean():.1f}")
        report.append(f"Minimum jumlah kata      : {df_processed['word_count'].min()}")
        report.append(f"Maximum jumlah kata      : {df_processed['word_count'].max()}")
        report.append(f"Rata-rata jumlah karakter: {df_processed['char_count'].mean():.1f}")
        
        report.append("\n" + "-"*70)
        report.append("SAMPLE HASIL PREPROCESSING (10 PERTAMA)")
        report.append("-"*70)
        
        for idx, row in df_processed.head(10).iterrows():
            report.append(f"\n[{row['no']}] Original:")
            report.append(f"    {row['text'][:100]}..." if len(row['text']) > 100 else f"    {row['text']}")
            report.append(f"    Processed:")
            report.append(f"    {row['text_processed']}")
        
        report.append("\n" + "="*70)
        report.append("END OF REPORT")
        report.append("="*70)
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        print(f"\n[INFO] Laporan tersimpan: {output_path}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main function"""
    print("""
    ╔═══════════════════════════════════════════════════════════════════╗
    ║     PREPROCESSING DATA KOMENTAR YOUTUBE                           ║
    ║     Analisis Sentimen Lagu "Arteri" - .FEAST                      ║
    ║     Metode: SVM dengan TF-IDF                                     ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """)
    
    # Check input file
    if not os.path.exists(Config.INPUT_FILE):
        print(f"[ERROR] File tidak ditemukan: {Config.INPUT_FILE}")
        return
    
    # Load data
    print(f"[INFO] Memuat data dari: {Config.INPUT_FILE}")
    df = pd.read_csv(Config.INPUT_FILE)
    print(f"[INFO] Jumlah komentar: {len(df)}")
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor(Config)
    
    # Run preprocessing
    df_processed = preprocessor.preprocess_dataframe(
        df, 
        text_column='text',
        apply_stemming=True,
        apply_stopwords=True
    )
    
    # Save processed data
    output_columns = ['no', 'author', 'text', 'text_clean', 'text_processed', 'likes', 'time']
    output_columns = [col for col in output_columns if col in df_processed.columns]
    
    df_processed[output_columns].to_csv(Config.OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f"\n[INFO] Data tersimpan: {Config.OUTPUT_FILE}")
    
    # Generate report
    preprocessor.generate_report(df, df_processed, Config.REPORT_FILE)
    
    print("\n[SELESAI] Preprocessing berhasil!")
    print(f"          - Data bersih : {Config.OUTPUT_FILE}")
    print(f"          - Laporan     : {Config.REPORT_FILE}")


if __name__ == "__main__":
    main()
