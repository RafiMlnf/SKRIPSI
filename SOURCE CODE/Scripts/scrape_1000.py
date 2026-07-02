# -*- coding: utf-8 -*-
"""
Scrape lebih banyak komentar untuk mencapai 1000+
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

from youtube_comment_downloader import YoutubeCommentDownloader, SORT_BY_RECENT
import pandas as pd
import time

def scrape_video(video_url, output_file, max_comments=1200):
    print(f"\n[*] Scraping: {video_url}")
    print(f"[*] Target: {max_comments} komentar")
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
    print("    Scraping 1000+ Komentar: ARTERI - FEAST")
    print("=" * 60)
    
    # Video MV - ambil maksimal 1200 komentar
    url = "https://www.youtube.com/watch?v=eulYTPXOaio"
    df = scrape_video(url, "Data/comments_arteri_full.csv", max_comments=1200)
    
    if df is not None:
        print("\n" + "=" * 60)
        print("[RINGKASAN]")
        print(f"    - Total komentar: {len(df)}")
        print(f"    - File: Data/comments_arteri_full.csv")
    
    print("\n[SELESAI]")
