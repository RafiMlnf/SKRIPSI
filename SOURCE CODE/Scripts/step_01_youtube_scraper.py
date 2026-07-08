# -*- coding: utf-8 -*-
"""
Scrape lebih banyak komentar untuk mencapai 1000+
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

from youtube_comment_downloader import YoutubeCommentDownloader, SORT_BY_RECENT
import pandas as pd
import time

def scrape_video(video_url, output_file, max_comments=None):
    print(f"\n[*] Scraping: {video_url}")
    if max_comments:
        print(f"[*] Target: {max_comments} komentar")
    else:
        print("[*] Target: Semua komentar (Tanpa Batas)")
    print("=" * 60)
    
    downloader = YoutubeCommentDownloader()
    comments_data = []
    
    try:
        comments = downloader.get_comments_from_url(video_url, sort_by=SORT_BY_RECENT)
        
        count = 0
        for comment in comments:
            if max_comments and count >= max_comments:
                break
            
            count += 1
            if count % 100 == 0:
                print(f"    -> Terkumpul: {count} komentar...")
                
            comments_data.append({
                'no': count,
                'author': comment.get('author', ''),
                'text': comment.get('text', ''),
                'time': comment.get('time', ''),
                'likes': comment.get('votes', 0),
                'replies': comment.get('replies', 0),
                'is_reply': comment.get('reply', False)
            })
        
        df = pd.DataFrame(comments_data)
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        print(f"\n[OK] Berhasil: {len(comments_data)} komentar")
        print(f"[FILE] Tersimpan: {output_file}")
        
        return df
        
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        return None

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("    Scraping Komentar dari 2 Video: ARTERI - FEAST")
    print("=" * 60)
    
    # Video 1: Official MV
    url1 = "https://www.youtube.com/watch?v=eulYTPXOaio"
    # Video 2: Lyric Video
    url2 = "https://www.youtube.com/watch?v=HLCLMuxV0bI"
    
    print("[*] Scraping Video 1 (Official MV)...")
    df1 = scrape_video(url1, "Data/comments_arteri_mv.csv", max_comments=None)
    
    print("\n[*] Scraping Video 2 (Lyric Video)...")
    df2 = scrape_video(url2, "Data/comments_arteri_lyric.csv", max_comments=None)
    
    if df1 is not None and df2 is not None:
        # Gabungkan dua dataset apa adanya
        combined_df = pd.concat([df1, df2], ignore_index=True)
        
        # Re-number the 'no' column
        combined_df['no'] = range(1, len(combined_df) + 1)
        
        # Save merged data
        combined_df.to_csv("Data/comments_arteri_full.csv", index=False, encoding='utf-8-sig')
        
        print("\n" + "=" * 60)
        print("[RINGKASAN GABUNGAN - DATA ASLI]")
        print(f"    - Video 1 (Official MV) : {len(df1)} komentar")
        print(f"    - Video 2 (Lyric Video) : {len(df2)} komentar")
        print(f"    - Total Gabungan        : {len(combined_df)} komentar")
        print(f"    - File Gabungan         : Data/comments_arteri_full.csv")
        print("=" * 60)
        print("\n[CATATAN] Ini adalah jumlah komentar ASLI dari kedua video.")
        print("          Tidak ada penambahan atau pengurangan data buatan.")
    
    print("\n[SELESAI]")

