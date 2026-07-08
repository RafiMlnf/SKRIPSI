import pandas as pd
import os
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Resolve paths relative to the script file location
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.abspath(os.path.join(script_dir, '..', 'Data'))
output_dir = os.path.abspath(os.path.join(script_dir, '..', 'Output'))

os.makedirs(output_dir, exist_ok=True)

input_file = os.path.join(data_dir, 'comments_labeled.csv')

if os.path.exists(input_file):
    df = pd.read_csv(input_file)
    print(f"[INFO] Dataset loaded successfully from {input_file}")
else:
    print(f"[ERROR] Dataset {input_file} not found!")
    exit(1)

wc_configs = [
    {
        'sentiment': 'positif',
        'color': '#10b981',
        'filename': 'wordcloud_positif.png'
    },
    {
        'sentiment': 'netral',
        'color': '#3b82f6',
        'filename': 'wordcloud_netral.png'
    },
    {
        'sentiment': 'negatif',
        'color': '#ef4444',
        'filename': 'wordcloud_negatif.png'
    }
]

for config in wc_configs:
    sentiment = config['sentiment']
    color = config['color']
    filename = config['filename']
    
    texts = df[df['sentiment'] == sentiment]['text_processed'].dropna().astype(str)
    all_text = ' '.join(texts.tolist())
    
    if all_text.strip():
        # Widescreen Word Cloud
        wc = WordCloud(
            width=1200, height=500,
            background_color='#ffffff',
            color_func=lambda *args, **kwargs: color,
            max_words=100,
            collocations=False
        ).generate(all_text)
        
        fig, ax = plt.subplots(figsize=(12, 5))
        fig.patch.set_facecolor('#ffffff')
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        plt.tight_layout(pad=0)
        
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[SUCCESS] Word Cloud saved to: {output_path}")
    else:
        print(f"[WARNING] No text available for sentiment: {sentiment}")
