"""
YouTube Comment Scraper untuk Analisis Sentimen
================================================
Script ini digunakan untuk mengambil komentar dari video YouTube
untuk keperluan analisis sentimen lagu "Arteri" - FEAST

Metode: YouTube Data API v3 + youtube-comment-downloader (backup)
"""

import os
import csv
import json
import time
from datetime import datetime

# ============================================
# METODE 1: Menggunakan youtube-comment-downloader
# (Tidak memerlukan API Key - Recommended)
# ============================================

def install_dependencies():
    """Install required packages"""
    import subprocess
    import sys
    
    packages = ['youtube-comment-downloader', 'pandas', 'tqdm']
    for package in packages:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package, '-q'])
    print("✅ Dependencies installed successfully!")

def scrape_comments_no_api(video_url, output_file='comments.csv', max_comments=None):
    """
    Scrape YouTube comments tanpa API key
    Menggunakan youtube-comment-downloader
    
    Args:
        video_url: URL video YouTube
        output_file: Nama file output (CSV)
        max_comments: Jumlah maksimum komentar (None = semua)
    """
    try:
        from youtube_comment_downloader import YoutubeCommentDownloader, SORT_BY_POPULAR, SORT_BY_RECENT
        import pandas as pd
        from tqdm import tqdm
    except ImportError:
        print("📦 Installing dependencies...")
        install_dependencies()
        from youtube_comment_downloader import YoutubeCommentDownloader, SORT_BY_POPULAR, SORT_BY_RECENT
        import pandas as pd
        from tqdm import tqdm
    
    print(f"\n🎵 Scraping komentar dari: {video_url}")
    print("=" * 60)
    
    downloader = YoutubeCommentDownloader()
    comments_data = []
    
    try:
        # Get comments sorted by most recent
        comments = downloader.get_comments_from_url(video_url, sort_by=SORT_BY_RECENT)
        
        print("📥 Mengambil komentar...")
        
        for idx, comment in enumerate(tqdm(comments, desc="Scraping")):
            if max_comments and idx >= max_comments:
                break
                
            comments_data.append({
                'no': idx + 1,
                'author': comment.get('author', ''),
                'text': comment.get('text', ''),
                'time': comment.get('time', ''),
                'likes': comment.get('votes', 0),
                'replies': comment.get('replies', 0),
                'is_reply': comment.get('reply', False)
            })
        
        # Save to CSV
        df = pd.DataFrame(comments_data)
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        print(f"\n✅ Berhasil mengambil {len(comments_data)} komentar!")
        print(f"📁 File disimpan: {output_file}")
        
        # Show statistics
        print("\n📊 Statistik:")
        print(f"   - Total komentar: {len(comments_data)}")
        print(f"   - Komentar utama: {len(df[df['is_reply'] == False])}")
        print(f"   - Balasan: {len(df[df['is_reply'] == True])}")
        
        return df
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return None


# ============================================
# METODE 2: Menggunakan YouTube Data API v3
# (Memerlukan API Key dari Google Cloud Console)
# ============================================

def scrape_comments_with_api(video_id, api_key, output_file='comments_api.csv', max_results=100):
    """
    Scrape YouTube comments menggunakan YouTube Data API v3
    
    Args:
        video_id: ID video YouTube (bagian setelah v=)
        api_key: API Key dari Google Cloud Console
        output_file: Nama file output (CSV)
        max_results: Jumlah maksimum komentar
    """
    try:
        from googleapiclient.discovery import build
        import pandas as pd
    except ImportError:
        import subprocess
        import sys
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 
                              'google-api-python-client', 'pandas', '-q'])
        from googleapiclient.discovery import build
        import pandas as pd
    
    print(f"\n🔑 Menggunakan YouTube Data API v3")
    print(f"🎬 Video ID: {video_id}")
    print("=" * 60)
    
    youtube = build('youtube', 'v3', developerKey=api_key)
    
    comments_data = []
    next_page_token = None
    
    try:
        while len(comments_data) < max_results:
            request = youtube.commentThreads().list(
                part='snippet,replies',
                videoId=video_id,
                maxResults=min(100, max_results - len(comments_data)),
                pageToken=next_page_token,
                textFormat='plainText'
            )
            response = request.execute()
            
            for item in response.get('items', []):
                snippet = item['snippet']['topLevelComment']['snippet']
                comments_data.append({
                    'no': len(comments_data) + 1,
                    'author': snippet['authorDisplayName'],
                    'text': snippet['textDisplay'],
                    'time': snippet['publishedAt'],
                    'likes': snippet['likeCount'],
                    'replies': item['snippet']['totalReplyCount'],
                    'is_reply': False
                })
                
                # Get replies if any
                if 'replies' in item:
                    for reply in item['replies']['comments']:
                        reply_snippet = reply['snippet']
                        comments_data.append({
                            'no': len(comments_data) + 1,
                            'author': reply_snippet['authorDisplayName'],
                            'text': reply_snippet['textDisplay'],
                            'time': reply_snippet['publishedAt'],
                            'likes': reply_snippet['likeCount'],
                            'replies': 0,
                            'is_reply': True
                        })
            
            next_page_token = response.get('nextPageToken')
            if not next_page_token:
                break
                
            print(f"📥 Terkumpul: {len(comments_data)} komentar...")
        
        # Save to CSV
        df = pd.DataFrame(comments_data)
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        print(f"\n✅ Berhasil mengambil {len(comments_data)} komentar!")
        print(f"📁 File disimpan: {output_file}")
        
        return df
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return None


# ============================================
# FUNGSI UTILITAS
# ============================================

def extract_video_id(url):
    """Extract video ID from YouTube URL"""
    import re
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
        r'(?:embed\/)([0-9A-Za-z_-]{11})',
        r'(?:youtu\.be\/)([0-9A-Za-z_-]{11})'
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def clean_comments_for_sentiment(input_file, output_file='comments_cleaned.csv'):
    """
    Membersihkan komentar untuk analisis sentimen
    - Menghapus emoji
    - Menghapus URL
    - Menghapus karakter spesial
    """
    import pandas as pd
    import re
    
    df = pd.read_csv(input_file)
    
    def clean_text(text):
        if pd.isna(text):
            return ""
        text = str(text)
        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        # Remove emojis
        text = re.sub(r'[^\w\s.,!?]', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text.strip()
    
    df['text_cleaned'] = df['text'].apply(clean_text)
    df = df[df['text_cleaned'].str.len() > 0]  # Remove empty comments
    
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"✅ Komentar dibersihkan! Tersimpan di: {output_file}")
    print(f"   Total komentar bersih: {len(df)}")
    
    return df


def save_to_json(df, output_file='comments.json'):
    """Save DataFrame to JSON format"""
    df.to_json(output_file, orient='records', force_ascii=False, indent=2)
    print(f"📁 File JSON disimpan: {output_file}")


# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║     YouTube Comment Scraper - Analisis Sentimen           ║
    ║     Lagu: "Arteri" - FEAST                                ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    # ========================================
    # KONFIGURASI - EDIT SESUAI KEBUTUHAN
    # ========================================
    
    # URL video YouTube lagu "Arteri" - FEAST
    VIDEO_URL = "https://www.youtube.com/watch?v=GANTI_DENGAN_VIDEO_ID"
    
    # Jumlah maksimum komentar (None = ambil semua)
    MAX_COMMENTS = 1000
    
    # Nama file output
    OUTPUT_FILE = "comments_arteri_feast.csv"
    
    # ========================================
    
    print("📌 Pilih metode scraping:")
    print("   1. Tanpa API Key (Recommended)")
    print("   2. Dengan YouTube Data API v3")
    print()
    
    choice = input("Masukkan pilihan (1/2): ").strip()
    
    if choice == "1":
        # Metode tanpa API Key
        url = input("Masukkan URL video YouTube: ").strip() or VIDEO_URL
        max_comments = input(f"Jumlah maksimum komentar [{MAX_COMMENTS}]: ").strip()
        max_comments = int(max_comments) if max_comments else MAX_COMMENTS
        
        df = scrape_comments_no_api(url, OUTPUT_FILE, max_comments)
        
    elif choice == "2":
        # Metode dengan API Key
        url = input("Masukkan URL video YouTube: ").strip() or VIDEO_URL
        video_id = extract_video_id(url)
        api_key = input("Masukkan API Key: ").strip()
        
        if not api_key:
            print("❌ API Key diperlukan untuk metode ini!")
            print("   Dapatkan di: https://console.cloud.google.com/apis/credentials")
        else:
            df = scrape_comments_with_api(video_id, api_key, OUTPUT_FILE, MAX_COMMENTS)
    
    else:
        print("❌ Pilihan tidak valid!")
        exit(1)
    
    # Clean comments for sentiment analysis
    if df is not None and len(df) > 0:
        print("\n" + "=" * 60)
        clean_choice = input("Bersihkan komentar untuk analisis sentimen? (y/n): ").strip().lower()
        if clean_choice == 'y':
            clean_comments_for_sentiment(OUTPUT_FILE, 'comments_cleaned.csv')
        
        # Save to JSON
        json_choice = input("Simpan juga dalam format JSON? (y/n): ").strip().lower()
        if json_choice == 'y':
            save_to_json(df, OUTPUT_FILE.replace('.csv', '.json'))
    
    print("\n🎉 Selesai! Komentar siap untuk analisis sentimen LSTM.")
