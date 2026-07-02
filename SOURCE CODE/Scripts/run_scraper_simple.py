# -*- coding: utf-8 -*-
"""
Script untuk menjalankan scraping komentar video Arteri - FEAST
"""

from youtube_comment_downloader import YoutubeCommentDownloader, SORT_BY_RECENT
import pandas as pd
import time
import sys

# Fix encoding untuk Windows
sys.stdout.reconfigure(encoding='utf-8')

def scrape_video(video_url, output_file, max_comments=500):
    print(f"\n[*] Scraping: {video_url}")
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
            if count % 50 == 0:
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
    print("\n")
    print("=" * 60)
    print("    Scraping Komentar: ARTERI - FEAST")
    print("=" * 60)
    
    # Video 1: Official Music Video
    url1 = "https://www.youtube.com/watch?v=eulYTPXOaio"
    df1 = scrape_video(url1, "comments_arteri_mv.csv", max_comments=500)
    
    time.sleep(2)  # Delay antar request
    
    # Video 2: Lyric Video / Audio
    url2 = "https://www.youtube.com/watch?v=HLCLMuxV0bI"
    df2 = scrape_video(url2, "comments_arteri_lyric.csv", max_comments=500)
    
    # Gabungkan kedua dataset
    if df1 is not None and df2 is not None:
        combined = pd.concat([df1, df2], ignore_index=True)
        combined['no'] = range(1, len(combined) + 1)
        combined.to_csv("comments_arteri_combined.csv", index=False, encoding='utf-8-sig')
        
        print("\n" + "=" * 60)
        print("[RINGKASAN]")
        print(f"    - Video 1 (MV): {len(df1)} komentar")
        print(f"    - Video 2 (Lyric): {len(df2)} komentar")
        print(f"    - Total Gabungan: {len(combined)} komentar")
        print(f"\n[FILE GABUNGAN] comments_arteri_combined.csv")
    
    print("\n[SELESAI]")
