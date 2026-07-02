# -*- coding: utf-8 -*-
"""
=============================================================================
KAMUS ALAY & NAMED ENTITY RECOGNIZER (NER) INDONESIA
Untuk Preprocessing Analisis Sentimen Lagu "Arteri" - .FEAST
=============================================================================
Author: Rafi Maulana Firdaus (312210382)
Tanggal: Februari 2026
Keperluan: Skripsi Bab 4

Fitur:
1. Kamus Alay (15.000+ kata slang → baku)
2. Named Entity Recognition untuk nama tempat Indonesia
3. Integrasi dengan preprocessing.py
=============================================================================
"""

import pandas as pd
import os

# ============================================================================
# LOAD KAMUS ALAY DARI CSV
# ============================================================================

def load_kamus_alay(csv_path="Data/kamus_alay.csv"):
    """
    Load kamus alay dari file CSV
    Format CSV: slang,formal,In-dictionary,context,category1,category2,category3
    
    Returns:
        dict: {slang: formal}
    """
    try:
        df = pd.read_csv(csv_path)
        # Hanya ambil kolom slang dan formal
        kamus = {}
        for _, row in df.iterrows():
            slang = str(row['slang']).lower().strip()
            formal = str(row['formal']).lower().strip()
            # Skip jika slang == formal atau keduanya kosong
            if slang and formal and slang != formal and slang != 'nan' and formal != 'nan':
                kamus[slang] = formal
        
        print(f"[INFO] Loaded {len(kamus)} kata slang dari kamus alay")
        return kamus
    except Exception as e:
        print(f"[WARNING] Gagal load kamus alay: {e}")
        return {}


# ============================================================================
# NAMED ENTITY RECOGNITION (NER) - NAMA TEMPAT INDONESIA
# ============================================================================

# Daftar Provinsi Indonesia
PROVINSI_INDONESIA = {
    'aceh', 'sumatera utara', 'sumut', 'sumatera barat', 'sumbar',
    'riau', 'jambi', 'sumatera selatan', 'sumsel', 'bengkulu',
    'lampung', 'bangka belitung', 'babel', 'kepulauan riau', 'kepri',
    'jakarta', 'dki jakarta', 'jawa barat', 'jabar', 'jawa tengah', 'jateng',
    'yogyakarta', 'jogja', 'jogjakarta', 'di yogyakarta', 'diy',
    'jawa timur', 'jatim', 'banten',
    'bali', 'nusa tenggara barat', 'ntb', 'nusa tenggara timur', 'ntt',
    'kalimantan barat', 'kalbar', 'kalimantan tengah', 'kalteng',
    'kalimantan selatan', 'kalsel', 'kalimantan timur', 'kaltim',
    'kalimantan utara', 'kaltara',
    'sulawesi utara', 'sulut', 'sulawesi tengah', 'sulteng',
    'sulawesi selatan', 'sulsel', 'sulawesi tenggara', 'sultra',
    'gorontalo', 'sulawesi barat', 'sulbar',
    'maluku', 'maluku utara', 'malut',
    'papua', 'papua barat', 'papbar', 'papua tengah', 'papua pegunungan',
    'papua selatan',
}

# Daftar Kota Besar Indonesia
KOTA_INDONESIA = {
    # Jabodetabek
    'jakarta', 'bogor', 'depok', 'tangerang', 'bekasi',
    'tangerang selatan', 'tangsel', 'jakarta selatan', 'jaksel',
    'jakarta barat', 'jakbar', 'jakarta timur', 'jaktim',
    'jakarta pusat', 'jakpus', 'jakarta utara', 'jakut',
    
    # Jawa Barat
    'bandung', 'cirebon', 'tasikmalaya', 'sukabumi', 'karawang',
    'cimahi', 'garut', 'subang', 'purwakarta',
    
    # Jawa Tengah
    'semarang', 'solo', 'surakarta', 'magelang', 'pekalongan',
    'tegal', 'salatiga', 'kudus', 'demak', 'purwokerto', 'klaten',
    
    # Yogyakarta
    'yogyakarta', 'jogja', 'jogjakarta', 'sleman', 'bantul',
    
    # Jawa Timur
    'surabaya', 'malang', 'sidoarjo', 'gresik', 'mojokerto',
    'kediri', 'blitar', 'madiun', 'jember', 'banyuwangi', 'lamongan',
    
    # Sumatera
    'medan', 'palembang', 'pekanbaru', 'padang', 'bandar lampung',
    'batam', 'binjai', 'dumai', 'jambi', 'bengkulu',
    
    # Kalimantan
    'balikpapan', 'samarinda', 'banjarmasin', 'pontianak', 'palangkaraya',
    'bontang', 'tarakan',
    
    # Sulawesi
    'makassar', 'manado', 'palu', 'kendari', 'gorontalo', 'mamuju',
    
    # Bali & Nusa Tenggara
    'denpasar', 'bali', 'mataram', 'kupang', 'lombok',
    
    # Papua & Maluku
    'jayapura', 'ambon', 'ternate', 'sorong', 'merauke', 'timika',
}

# Kecamatan & Desa di Jabodetabek dan sekitarnya
KECAMATAN_DESA = {
    # === TANGERANG & BANTEN ===
    'cisauk', 'kresek', 'tenjo', 'rajeg', 'mauk', 'sepatan', 'teluknaga',
    'kronjo', 'kemiri', 'sukamulya', 'pasar kemis', 'pasarkemis', 'cikupa',
    'curug', 'legok', 'pagedangan', 'kelapa dua', 'kelapadua', 'tigaraksa',
    'jambe', 'cisoka', 'solear', 'jayanti', 'balaraja', 'sindang jaya',
    'cipondoh', 'pinang', 'karawaci', 'cibodas', 'periuk', 'neglasari',
    'batuceper', 'benda', 'jatiuwung', 'tangerang', 'ciledug', 'larangan',
    'karang tengah', 'pondok aren', 'ciputat', 'ciputat timur', 'pamulang',
    'serpong', 'serpong utara', 'setu',
    'serang', 'cilegon', 'pandeglang', 'lebak', 'rangkasbitung',
    
    # === BOGOR ===
    'cibinong', 'cileungsi', 'gunung putri', 'gunungputri', 'citeureup',
    'parung', 'sawangan', 'bojong gede', 'bojonggede', 'tajur halang',
    'sukaraja', 'babakan madang', 'jonggol', 'cariu', 'tanjungsari',
    'ciawi', 'cisarua', 'megamendung', 'puncak', 'gadog',
    'ciomas', 'dramaga', 'ciampea', 'cibungbulang', 'pamijahan',
    'leuwiliang', 'nanggung', 'rumpin', 'parung panjang', 'tenjo',
    'jasinga', 'cigudeg', 'sukajaya',
    
    # === DEPOK ===
    'beji', 'pancoran mas', 'cipayung', 'sukmajaya', 'cilodong',
    'cimanggis', 'tapos', 'sawangan', 'bojong sari', 'cinere', 'limo',
    
    # === BEKASI ===
    'cikarang', 'cikarang barat', 'cikarang timur', 'cikarang utara',
    'cikarang selatan', 'cikarang pusat', 'cibitung', 'tambun',
    'tambun selatan', 'tambun utara', 'babelan', 'tarumajaya',
    'cabangbungin', 'muaragembong', 'sukatani', 'sukawangi', 'serang baru',
    'setu', 'cibarusah', 'bojongmangu', 'kedungwaringin', 'pebayuran',
    'karangbahagia',
    'bekasi barat', 'bekasi utara', 'bekasi timur', 'bekasi selatan',
    'rawalumbu', 'mustikajaya', 'bantargebang', 'pondok gede', 'pondokgede',
    'jatiasih', 'jatisampurna', 'pondok melati',
    
    # === JAKARTA (Kecamatan) ===
    'menteng', 'gambir', 'tanah abang', 'senen', 'cempaka putih', 'johar baru',
    'kemayoran', 'sawah besar', 'tebet', 'setiabudi', 'mampang prapatan',
    'pasar minggu', 'kebayoran baru', 'kebayoran lama', 'pesanggrahan',
    'cilandak', 'jagakarsa', 'pancoran', 'kalibata',
    'grogol petamburan', 'taman sari', 'tambora', 'palmerah', 'kebon jeruk',
    'kembangan', 'cengkareng', 'kalideres',
    'matraman', 'pulo gadung', 'jatinegara', 'duren sawit', 'cakung',
    'kramat jati', 'makasar', 'cipayung', 'ciracas', 'pasar rebo',
    'koja', 'kelapa gading', 'tanjung priok', 'pademangan', 'penjaringan',
    'cilincing',
    'kepulauan seribu',
    
    # === LOKASI TERKENAL DI JAKARTA ===
    'pondok indah', 'kemang', 'senopati', 'sudirman', 'thamrin', 'kuningan',
    'senayan', 'scbd', 'blok m', 'mangga dua', 'glodok', 'kota tua',
    'ancol', 'pik', 'pantai indah kapuk', 'kelapa gading', 'sunter',
    'pluit', 'gading serpong', 'bsd', 'alam sutera', 'lippo karawaci',
    'bintaro', 'cinere', 'margonda', 'ui', 'universitas indonesia',
    
    # === BANDUNG & JAWA BARAT ===
    'dago', 'cihampelas', 'braga', 'lembang', 'ciwidey', 'pangalengan',
    'cimahi', 'padalarang', 'cianjur', 'sukabumi', 'tasikmalaya',
    'majalengka', 'kuningan', 'indramayu', 'sumedang',
}

# Lokasi yang disebutkan dalam konteks lagu Arteri
LOKASI_KONTEKS = {
    'pondok indah',  # Disebutkan dalam lagu
    'jakarta selatan',
    'kemang',
    'senopati',
}

# Gabungkan semua nama tempat
NAMA_TEMPAT_INDONESIA = PROVINSI_INDONESIA | KOTA_INDONESIA | KECAMATAN_DESA | LOKASI_KONTEKS


# ============================================================================
# KATA-KATA YANG TIDAK BOLEH DI-STEM (PRESERVE WORDS)
# ============================================================================

# Kata-kata konteks .FEAST dan budaya populer
PRESERVE_WORDS_EXTENDED = {
    # Personil .FEAST
    'baskara', 'bas', 'feast',
    
    # Referensi film
    '500', 'days', 'summer',  # 500 Days of Summer
    
    # Referensi band/musik
    'mcr', 'my', 'chemical', 'romance',
    
    # Tema lagu Arteri
    'quarter', 'life', 'crisis', 'arteri',
    
    # Semua nama tempat Indonesia
    *NAMA_TEMPAT_INDONESIA,
    
    # Nama-nama umum Indonesia yang sering muncul
    'rina', 'budi', 'andi', 'siti', 'dewi', 'putri', 'ahmad', 'muhammad',
}


# ============================================================================
# FUNGSI HELPER
# ============================================================================

def get_extended_slang_dict(csv_path="Data/kamus_alay.csv"):
    """
    Gabungkan kamus alay dari CSV dengan kamus manual
    
    Args:
        csv_path: Path ke file CSV kamus alay
    
    Returns:
        dict: Kamus slang lengkap
    """
    # Load dari CSV
    kamus_csv = load_kamus_alay(csv_path)
    
    # Kamus manual untuk kata-kata yang sering muncul di komentar YouTube musik
    kamus_manual = {
        # Ekspresi musik
        'lagunya': 'lagunya',
        'musiknya': 'musiknya',
        'liriknya': 'liriknya',
        'vokalnya': 'vokalnya',
        
        # Referensi spesifik
        'bas': 'baskara',  # BUKAN bass (alat musik)
        'mcr': 'my chemical romance',
        
        # Typo umum
        'bagaus': 'bagus',
        'kreen': 'keren',
        'mantulll': 'mantul',
    }
    
    # Gabungkan (kamus manual override CSV jika ada konflik)
    kamus_csv.update(kamus_manual)
    
    return kamus_csv


def is_named_entity(word):
    """
    Cek apakah kata adalah Named Entity (nama tempat)
    
    Args:
        word: Kata yang akan dicek
    
    Returns:
        bool: True jika kata adalah Named Entity
    """
    return word.lower() in NAMA_TEMPAT_INDONESIA


def should_preserve_word(word):
    """
    Cek apakah kata harus dipertahankan (tidak di-stem)
    
    Args:
        word: Kata yang akan dicek
    
    Returns:
        bool: True jika kata harus dipertahankan
    """
    return word.lower() in PRESERVE_WORDS_EXTENDED


# ============================================================================
# MAIN (untuk testing)
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("KAMUS ALAY & NER INDONESIA - TEST")
    print("="*70)
    
    # Test load kamus
    kamus = get_extended_slang_dict()
    print(f"\nTotal kata dalam kamus: {len(kamus)}")
    
    # Sample entries
    print("\n--- Sample Kamus Alay ---")
    sample_keys = list(kamus.keys())[:10]
    for key in sample_keys:
        print(f"  '{key}' -> '{kamus[key]}'")
    
    # Test NER
    print("\n--- Test Named Entity Recognition ---")
    test_words = ['jakarta', 'bandung', 'random', 'pondok indah', 'baskara']
    for word in test_words:
        is_ner = is_named_entity(word)
        is_preserve = should_preserve_word(word)
        print(f"  '{word}': NER={is_ner}, Preserve={is_preserve}")
    
    print("\n" + "="*70)
    print("Test selesai!")
