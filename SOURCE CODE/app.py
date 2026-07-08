import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import sys
import re

# Tambahkan folder Scripts ke path agar bisa import modul
sys.path.append(os.path.join(os.path.dirname(__file__), 'Scripts'))

try:
    from step_02_preprocessing import TextPreprocessor
    from step_03_labeling import SentimentLabeler
except ImportError:
    st.error("Gagal mengimpor modul preprocessing atau labeling. Pastikan folder Scripts berisi file tersebut.")

# Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="Sentimen Arteri - .FEAST (SVM)",
    page_icon=":material/music_note:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk tampilan premium modern
st.markdown("""
<style>
    /* Styling font dan background */
    @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css');
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3, h4, h5, h6, .title-text {
        font-family: 'Outfit', sans-serif;
    }
    
    .main {
        background-color: #ffffff;
        color: #0f172a;
    }
    
    /* Card/Box Styling */
    .metric-card {
        background: rgba(255, 255, 255, 0.7);
        border: 1px solid rgba(15, 23, 42, 0.06);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 12px -2px rgba(0, 0, 0, 0.03);
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 24px -10px rgba(249, 115, 22, 0.20);
        border: 1px solid rgba(249, 115, 22, 0.25);
    }
    
    /* Sentiment badges */
    .badge-positive {
        background-color: rgba(16, 185, 129, 0.08);
        color: #047857;
        border: 1px solid rgba(16, 185, 129, 0.25);
        border-radius: 12px;
        padding: 0.75rem 1.25rem;
        font-weight: 600;
        font-size: 1.1rem;
        display: inline-flex;
        align-items: center;
        gap: 8px;
    }
    
    .badge-negative {
        background-color: rgba(239, 68, 68, 0.08);
        color: #b91c1c;
        border: 1px solid rgba(239, 68, 68, 0.25);
        border-radius: 12px;
        padding: 0.75rem 1.25rem;
        font-weight: 600;
        font-size: 1.1rem;
        display: inline-flex;
        align-items: center;
        gap: 8px;
    }
    
    .badge-neutral {
        background-color: rgba(59, 130, 246, 0.08);
        color: #1d4ed8;
        border: 1px solid rgba(59, 130, 246, 0.25);
        border-radius: 12px;
        padding: 0.75rem 1.25rem;
        font-weight: 600;
        font-size: 1.1rem;
        display: inline-flex;
        align-items: center;
        gap: 8px;
    }
    
    /* Preprocessing steps box */
    .step-box {
        background: rgba(15, 23, 42, 0.03);
        border: 1px solid rgba(15, 23, 42, 0.05);
        border-left: 4px solid #f97316;
        padding: 1rem;
        margin-bottom: 1rem;
        border-radius: 4px 12px 12px 4px;
        color: #334155;
    }
</style>
""", unsafe_allow_html=True)

# Helper untuk memuat model dan data
@st.cache_resource
def load_ml_assets():
    model_path = "Output/svm_sentiment_model.pkl"
    vectorizer_path = "Output/tfidf_vectorizer.pkl"
    
    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        return model, vectorizer
    return None, None

@st.cache_resource
def init_labeler():
    labeler = SentimentLabeler()
    return labeler

@st.cache_resource
def init_preprocessor():
    preprocessor = TextPreprocessor()
    preprocessor._init_stemmer()
    return preprocessor

@st.cache_data
def load_labeled_data():
    data_path = "Data/comments_labeled.csv"
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    return None

# Load Assets
model, vectorizer = load_ml_assets()
preprocessor = init_preprocessor()
labeler = init_labeler()
df_labeled = load_labeled_data()

# Sidebar
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1514525253161-7a46d19cd819?w=500&auto=format&fit=crop&q=60&ixlib=rb-4.0.3", width='stretch')
    st.markdown("### <i class=\"fa-solid fa-graduation-cap\"></i> INFO SKRIPSI", unsafe_allow_html=True)
    st.markdown("""
    * **<i class="fa-solid fa-user"></i> Peneliti**: Rafi Maulana Firdaus
    * **<i class="fa-solid fa-id-card"></i> NIM**: 312210382
    * **<i class="fa-solid fa-book"></i> Judul**: Analisis Sentimen Pendengar Lagu "Arteri" oleh .FEAST Menggunakan Algoritma Support Vector Machine (SVM)
    * **<i class="fa-solid fa-university"></i> Kampus**: Universitas Pelita Bangsa
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### <i class=\"fa-solid fa-sliders\"></i> Konfigurasi Model", unsafe_allow_html=True)
    if model:
        st.success("Model SVM & TF-IDF Loaded!")
        # Tampilkan parameter terbaik
        st.write("**Hyperparameters Terpilih:**")
        st.write("- Kernel: `linear`")
        st.write("- C: `1.0` (Regularisasi)")
        st.write("- Gamma: `scale`")
    else:
        st.error("Model & Vectorizer tidak ditemukan. Silakan jalankan `step_04_svm_model.py` terlebih dahulu.")

# Main Layout
st.markdown("""
<div class="header-container" style="display: flex; align-items: center; gap: 20px; margin-bottom: 30px; padding: 20px; background: rgba(15, 23, 42, 0.02); border: 1px solid rgba(15, 23, 42, 0.06); border-radius: 16px;">
    <div class="header-icon-box" style="display: flex; align-items: center; justify-content: center; width: 60px; height: 60px; border-radius: 12px; background: linear-gradient(135deg, rgba(249, 115, 22, 0.08), rgba(239, 68, 68, 0.08)); border: 1px solid rgba(249, 115, 22, 0.2);">
        <i class="fa-solid fa-compact-disc fa-spin fa-2xl" style="color: #f97316;"></i>
    </div>
    <div>
        <h1 class="title-text" style="margin: 0; font-family: 'Outfit', sans-serif; font-weight: 800; font-size: 2.2rem; background: linear-gradient(90deg, #ff7e5f, #feb47b); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Analisis Sentimen Lagu "Arteri" - .FEAST</h1>
        <p class="subtitle-text" style="margin: 5px 0 0 0; font-family: 'Inter', sans-serif; font-weight: 400; color: #475569; font-size: 1rem; line-height: 1.4;">Klasifikasi Komentar YouTube Menggunakan Algoritma Support Vector Machine (SVM) dan Pembobotan TF-IDF</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    ":material/online_prediction: Prediksi Sentimen", 
    ":material/dashboard: Dashboard Dataset", 
    ":material/cloud: Word Cloud", 
    ":material/legend_toggle: Performa Model"
])

# TAB 1: PREDIKSI SENTIMEN
with tab1:
    st.markdown("### **Uji Sentimen Komentar Secara Real-Time**")
    
    col_input, col_result = st.columns([3, 2])
    
    with col_input:
        user_input = st.text_area(
            "Ketik komentar Anda mengenai lagu 'Arteri' di sini:",
            placeholder="Contoh: vokal baskara di lagu ini keren banget, relate sama fase hidup gw sekarang...",
            height=150
        )
        
        btn_predict = st.button("Analisis Sentimen", type="primary", icon=":material/search:", width='stretch')
        
    with col_result:
        if btn_predict and user_input:
            if model and vectorizer and preprocessor:
                # 1. Jalankan langkah Preprocessing satu per satu untuk demonstrasi visual
                st.markdown("#### **Tahapan Preprocessing Teks:**")
                
                step1_cf = preprocessor.case_folding(user_input)
                st.markdown(f'<div class="step-box"><b>1. Case Folding</b><br><code style="color: #feb47b;">{step1_cf}</code></div>', unsafe_allow_html=True)
                
                step2_clean = preprocessor.clean_data(step1_cf)
                step3_emoji = preprocessor.remove_emojis(step2_clean)
                st.markdown(f'<div class="step-box"><b>2. Data Cleaning & Emoji Removal</b><br><code style="color: #feb47b;">{step3_emoji}</code></div>', unsafe_allow_html=True)
                
                step4_punc = preprocessor.normalize_punctuation(step3_emoji)
                st.markdown(f'<div class="step-box"><b>3. Normalisasi Tanda Baca</b><br><code style="color: #feb47b;">{step4_punc}</code></div>', unsafe_allow_html=True)
                
                step5_slang = preprocessor.normalize_slang(step4_punc)
                st.markdown(f'<div class="step-box"><b>4. Normalisasi Slang (Kamus Alay)</b><br><code style="color: #feb47b;">{step5_slang}</code></div>', unsafe_allow_html=True)
                
                step6_stop = preprocessor.remove_stopwords(step5_slang)
                st.markdown(f'<div class="step-box"><b>5. Stopwords Removal (Negasi Dipertahankan)</b><br><code style="color: #feb47b;">{step6_stop}</code></div>', unsafe_allow_html=True)
                
                step7_stem = preprocessor.stem_text(step6_stop)
                step8_final = preprocessor.final_clean(step7_stem)
                st.markdown(f'<div class="step-box"><b>6. Stemming (Sastrawi) & Final Clean</b><br><code style="color: #feb47b;">{step8_final}</code></div>', unsafe_allow_html=True)
                
                # 2. Prediksi model
                # Transform dengan TF-IDF
                tfidf_vec = vectorizer.transform([step8_final])
                
                # Predict
                pred_label = model.predict(tfidf_vec)[0]
                proba = model.predict_proba(tfidf_vec)[0]
                
                labels = ['negatif', 'netral', 'positif']
                result_sentiment = labels[pred_label]
                confidence = proba[pred_label]
                
                st.markdown("---")
                st.markdown("#### **Hasil Klasifikasi Model SVM:**")
                
                # Tampilkan Badge berdasarkan kelas
                if result_sentiment == 'positif':
                    st.markdown(f'<div class="badge-positive"><i class="fa-solid fa-circle-check"></i> POSITIF (Confidence: {confidence:.2%})</div>', unsafe_allow_html=True)
                elif result_sentiment == 'negatif':
                    st.markdown(f'<div class="badge-negative"><i class="fa-solid fa-circle-exclamation"></i> NEGATIF (Confidence: {confidence:.2%})</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="badge-neutral"><i class="fa-solid fa-circle-minus"></i> NETRAL (Confidence: {confidence:.2%})</div>', unsafe_allow_html=True)
                
                # Tampilkan Chart Probabilitas
                prob_df = pd.DataFrame({
                    'Sentimen': ['Negatif', 'Netral', 'Positif'],
                    'Probabilitas': [proba[0] * 100, proba[1] * 100, proba[2] * 100]
                })
                
                fig, ax = plt.subplots(figsize=(6, 2.5))
                fig.patch.set_facecolor('#ffffff')
                ax.set_facecolor('#ffffff')
                
                colors = ['#ef4444', '#3b82f6', '#10b981']
                bars = ax.barh(prob_df['Sentimen'], prob_df['Probabilitas'], color=colors, height=0.5)
                
                ax.tick_params(colors='#334155')
                ax.spines['bottom'].set_color('#cbd5e1')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_color('#cbd5e1')
                ax.set_xlabel('Probabilitas (%)', color='#334155', fontsize=9)
                ax.xaxis.grid(True, linestyle='--', alpha=0.3, color='#cbd5e1')
                
                for bar in bars:
                    width = bar.get_width()
                    ax.text(width + 1, bar.get_y() + bar.get_height()/2, 
                            f'{width:.1f}%', 
                            va='center', ha='left', color='#334155', fontweight='bold', fontsize=9)
                
                plt.tight_layout()
                st.pyplot(fig)
                
            else:
                st.warning("Model belum siap digunakan. Pastikan model dan vectorizer sudah dilatih.")
        elif btn_predict:
            st.warning("Silakan masukkan teks terlebih dahulu!")

# TAB 2: DASHBOARD DATASET
with tab2:
    st.markdown("### **Dashboard Analisis Statistik Dataset**")
    
    if df_labeled is not None:
        total_data = len(df_labeled)
        pos_data = len(df_labeled[df_labeled['sentiment'] == 'positif'])
        neg_data = len(df_labeled[df_labeled['sentiment'] == 'negatif'])
        net_data = len(df_labeled[df_labeled['sentiment'] == 'netral'])
        
        # Cards row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f'<div class="metric-card"><h4 style="margin:0;color:#94a3b8;"><i class="fa-solid fa-database"></i> TOTAL DATA</h4><h1 style="margin:10px 0 0 0;font-size:3rem;font-weight:800;color:#fff;">{total_data}</h1></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card"><h4 style="margin:0;color:#10b981;"><i class="fa-solid fa-circle-check"></i> POSITIF</h4><h1 style="margin:10px 0 0 0;font-size:3rem;font-weight:800;color:#10b981;">{pos_data}</h1><p style="margin:0;color:#94a3b8;">{pos_data/total_data*100:.1f}%</p></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="metric-card"><h4 style="margin:0;color:#3b82f6;"><i class="fa-solid fa-circle-minus"></i> NETRAL</h4><h1 style="margin:10px 0 0 0;font-size:3rem;font-weight:800;color:#3b82f6;">{net_data}</h1><p style="margin:0;color:#94a3b8;">{net_data/total_data*100:.1f}%</p></div>', unsafe_allow_html=True)
        with col4:
            st.markdown(f'<div class="metric-card"><h4 style="margin:0;color:#ef4444;"><i class="fa-solid fa-circle-exclamation"></i> NEGATIF</h4><h1 style="margin:10px 0 0 0;font-size:3rem;font-weight:800;color:#ef4444;">{neg_data}</h1><p style="margin:0;color:#94a3b8;">{neg_data/total_data*100:.1f}%</p></div>', unsafe_allow_html=True)
            
        st.markdown("---")
        
        # Grafik Distribusi & Tabel Komentar dengan Frame Outline Grid
        col_chart, col_table = st.columns([3, 2])
        
        with col_chart:
            with st.container(border=True):
                st.markdown("##### <i class='fa-solid fa-chart-bar'></i> **Visualisasi Distribusi Sentimen**", unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(8, 4.5))
                fig.patch.set_facecolor('#ffffff')
                ax.set_facecolor('#ffffff')
                
                categories = ['Negatif', 'Netral', 'Positif']
                counts = [neg_data, net_data, pos_data]
                colors = ['#ef4444', '#3b82f6', '#10b981']
                
                bars = ax.bar(categories, counts, color=colors, edgecolor='black', width=0.5)
                
                # Label
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2, height + 5,
                            f'{height} ({height/total_data*100:.1f}%)',
                            ha='center', va='bottom', color='#0f172a', fontweight='bold')
                
                ax.tick_params(colors='#334155')
                ax.spines['bottom'].set_color('#cbd5e1')
                ax.spines['left'].set_color('#cbd5e1')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.set_ylabel('Jumlah Komentar', color='#0f172a')
                ax.grid(axis='y', alpha=0.2, color='#cbd5e1')
                
                st.pyplot(fig)
                plt.close(fig)
            
        with col_table:
            with st.container(border=True):
                st.markdown("##### <i class='fa-solid fa-table-list'></i> **Dataset Komentar YouTube Labeled**", unsafe_allow_html=True)
                # Menampilkan seluruh dataset komentar secara compact & scrollable
                st.dataframe(
                    df_labeled[['text', 'sentiment', 'sentiment_score']],
                    width='stretch',
                    height=360
                )
                
        # 3. Detail Penjelasan dan Deskripsi Distribusi di Bagian Bawah
        st.markdown("---")
        st.markdown("### **Analisis & Deskripsi Detail Distribusi Dataset**")
        
        col_dist_desc1, col_dist_desc2 = st.columns(2)
        with col_dist_desc1:
            st.markdown("#### **Karakteristik & Distribusi Sentimen**")
            st.markdown(f"""
            Dataset ini mencakup total **{total_data}** komentar bersih setelah melalui tahap quality filtering, pembersihan spam, lirik, dan duplikat.
            
            - **Kelas Netral ({net_data} data / {net_data/total_data*100:.1f}%)**: Merupakan kelas mayoritas yang dominan. Hal ini wajar dalam interaksi video klip musik di YouTube, di mana sebagian besar penonton mendiskusikan elemen objektif seperti membandingkan band .FEAST dengan band lain (misalnya My Chemical Romance), menanyakan lirik, membicarakan instrumen gitar/bass, atau memberikan opini umum tentang rilis album tanpa mengekspresikan rasa suka/tidak suka yang ekstrem.
            - **Kelas Positif ({pos_data} data / {pos_data/total_data*100:.1f}%)**: Mewakili ulasan dari pendengar yang mengapresiasi lagu "Arteri". Mereka memuji keunikan vokal Baskara, aransemen rock alternatif yang segar, dan mengekspresikan betapa liriknya sangat berhubungan (*relate*) dengan kehidupan emosional pribadi mereka.
            - **Kelas Negatif ({neg_data} data / {neg_data/total_data*100:.1f}%)**: Merupakan kelas minoritas terkecil. Berisi keluhan atau kritik objektif dari pendengar mengenai tempo lagu yang dirasa lambat/membosankan, kekecewaan terhadap warna musik baru, atau luapan emosi negatif yang terpicu oleh tema patah hati/toxic relationship dari lirik lagu itu sendiri.
            """)
            
        with col_dist_desc2:
            st.markdown("#### **Implikasi Terhadap Pelatihan Model (SVM)**")
            st.markdown(f"""
            Ketimpangan distribusi kelas (imbalance class) yang mencolok ini (**{net_data/total_data*100:.1f}% Netral** vs **{neg_data/total_data*100:.1f}% Negatif**) memberikan dampak langsung pada performa algoritma Machine Learning:
            
            1. **Bias Kelas Mayoritas:** 
               Jika model SVM dilatih langsung pada data asli tanpa penyeimbangan, model akan cenderung memprediksi sebagian besar komentar sebagai kelas mayoritas (Netral). Ini menjelaskan mengapa pada model awal, komentar negatif seperti *'lagunya membosankan'* salah diklasifikasikan sebagai positif atau netral.
               
            2. **Pentingnya Penyeimbangan Data (Random Oversampling):** 
               Dengan menduplikasi sampel kelas minoritas (Negatif dan Positif) hingga mencapai jumlah seimbang (**{net_data}** data untuk masing-masing kelas), kita memaksa model SVM untuk mempelajari bobot kata-kata kunci sentimen negatif secara adil, meningkatkan recall negatif secara drastis tanpa menurunkan akurasi global secara signifikan.
            """)
            
    else:
        st.warning("File `Data/comments_labeled.csv` tidak ditemukan. Silakan jalankan script pelabelan.")
 
# TAB 3: WORD CLOUD
with tab3:
    st.markdown("### **Visualisasi & Analisis Kata Dominan (Word Cloud)**")
    st.markdown("Halaman ini menyajikan Word Cloud dalam format layar lebar (widescreen) lengkap dengan analisis statistik kata kunci teratas untuk setiap kategori sentimen.")
    
    df_labeled_wc = None
    if os.path.exists("Data/comments_labeled.csv"):
        df_labeled_wc = pd.read_csv("Data/comments_labeled.csv")
    
    if df_labeled_wc is not None:
        try:
            from wordcloud import WordCloud
            from collections import Counter
            
            wc_configs = [
                {
                    'sentiment': 'positif',
                    'color': '#10b981',
                    'title': '<i class="fa-solid fa-circle-check" style="color: #10b981;"></i> Sentimen POSITIF (Apresiasi & Koneksi)',
                    'desc': 'Komentar bernada positif didominasi oleh pujian terhadap karya terbaru .FEAST ini serta tingkat keterkaitan (relatability) pendengar yang sangat tinggi.',
                    'analysis': 'Pendengar mengekspresikan kekaguman mereka terhadap lagu "Arteri" menggunakan kata **"keren"**, **"bagus"**, dan **"cinta"**. Penggunaan kata **"gue"** dan **"relate"** menunjukkan bahwa pendengar merasa lirik lagu ini sangat dekat dengan realitas kehidupan pribadi mereka, menciptakan keterikatan emosional yang kuat dengan lagu tersebut.'
                },
                {
                    'sentiment': 'netral',
                    'color': '#3b82f6',
                    'title': '<i class="fa-solid fa-circle-minus" style="color: #3b82f6;"></i> Sentimen NETRAL (Diskusi & Perbandingan)',
                    'desc': 'Komentar netral didominasi oleh perbincangan objektif mengenai musik, tanggal rilis, dan kemiripan aransemen dengan band rock alternatif ternama.',
                    'analysis': 'Kumpulan kata **"my"**, **"chemical"**, **"romance"**, **"not"**, dan **"okay"** menunjukkan tren diskusi di mana banyak pendengar membandingkan tempo dan gaya lagu "Arteri" dengan lagu legendaris *"I\'m Not Okay"* milik band My Chemical Romance (MCR). Diskusi ini bersifat analitis/perbandingan musik tanpa kecenderungan emosional suka atau tidak suka.'
                },
                {
                    'sentiment': 'negatif',
                    'color': '#ef4444',
                    'title': '<i class="fa-solid fa-circle-exclamation" style="color: #ef4444;"></i> Sentimen NEGATIF (Kritik & Keluhan)',
                    'desc': 'Komentar negatif merekam kekecewaan pendengar terhadap aspek musikalitas atau respons emosional negatif terhadap tema lirik patah hati.',
                    'analysis': 'Kata **"sakit"**, **"jelek"**, dan kata makian seperti **"kepharat"** muncul di kategori ini. Munculnya kata **"sakit"** dan **"cewek"** mengindikasikan keluhan pendengar yang terhanyut oleh tema patah hati/relasi toxic dalam lirik lagu, sementara kata **"jelek"** atau **"bosan"** merepresentasikan kritik minor terhadap arah aransemen musik yang dirasa kurang pas bagi selera sebagian pendengar.'
                }
            ]
            
            for config in wc_configs:
                sentiment = config['sentiment']
                color = config['color']
                title = config['title']
                desc = config['desc']
                analysis = config['analysis']
                
                st.markdown(f"""
                <div style="margin-top: 30px; padding: 10px; border-bottom: 2px solid rgba(15, 23, 42, 0.08);">
                    <h3 style="margin: 0; font-family: 'Outfit', sans-serif; font-weight: 700; font-size: 1.5rem;">{title}</h3>
                </div>
                """, unsafe_allow_html=True)
                
                texts = df_labeled_wc[df_labeled_wc['sentiment'] == sentiment]['text_processed'].dropna().astype(str)
                all_text = ' '.join(texts.tolist())
                
                if all_text.strip():
                    # Hitung kata teratas secara dinamis
                    words_list = all_text.split()
                    top_words = Counter(words_list).most_common(8)
                    top_words_str = ", ".join([f"**{word}** ({count}x)" for word, count in top_words])
                    
                    # Widescreen Word Cloud dengan Frame Outline Grid
                    with st.container(border=True):
                        wc = WordCloud(
                            width=1200, height=500,
                            background_color='#ffffff',
                            color_func=lambda *args, **kwargs: color,
                            max_words=100,
                            collocations=False
                        ).generate(all_text)
                        
                        fig_wc, ax_wc = plt.subplots(figsize=(12, 5))
                        fig_wc.patch.set_facecolor('#ffffff')
                        ax_wc.imshow(wc, interpolation='bilinear')
                        ax_wc.axis('off')
                        plt.tight_layout(pad=0)
                        st.pyplot(fig_wc)
                        plt.close(fig_wc)
                    
                    # Detail penjelasan dibawah visual
                    st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
                    col_desc, col_words, col_anal = st.columns(3)
                    
                    with col_desc:
                        st.markdown("##### **Fokus Ulasan**")
                        st.write(desc)
                        
                    with col_words:
                        st.markdown("##### **Kata Kunci Teratas**")
                        st.markdown(top_words_str)
                        
                    with col_anal:
                        st.markdown("##### **Analisis & Interpretasi**")
                        st.write(analysis)
                else:
                    st.info("Tidak ada data ulasan untuk kelas ini.")
                st.markdown("<div style='margin-bottom: 25px;'></div>", unsafe_allow_html=True)
                
        except ImportError:
            st.warning("Library `wordcloud` tidak terinstall. Jalankan: `pip install wordcloud`")
    else:
        st.warning("File `Data/comments_labeled.csv` belum tersedia. Jalankan `step_03_labeling.py` terlebih dahulu.")
 
# TAB 4: PERFORMA MODEL SVM
with tab4:
    st.markdown("### **Metrik Evaluasi dan Performa Klasifikasi SVM**")
    
    # 1. Tampilkan Visual Performa Model di Bagian Atas dengan Frame Outline
    col_cm_frame, col_metrics_frame = st.columns(2)
    
    with col_cm_frame:
        with st.container(border=True):
            st.markdown("##### <i class='fa-solid fa-table-cells'></i> **Visual Confusion Matrix**", unsafe_allow_html=True)
            if os.path.exists("Output/confusion_matrix.png"):
                st.image("Output/confusion_matrix.png", width='stretch')
            else:
                st.info("Confusion matrix plot belum tersedia.")
                
    with col_metrics_frame:
        with st.container(border=True):
            st.markdown("##### <i class='fa-solid fa-chart-line'></i> **Visual Perbandingan Metrik**", unsafe_allow_html=True)
            if os.path.exists("Output/metrics_comparison.png"):
                st.image("Output/metrics_comparison.png", width='stretch')
            else:
                st.info("Grafik metrik perbandingan belum tersedia.")
                
    st.markdown("---")
    
    # 2. Detail Penjelasan dan Deskripsi di Bagian Bawah
    st.markdown("### **Analisis & Deskripsi Detail Performa Model**")
    
    # Ambil nilai metrik
    acc, prec, rec, f1w = "N/A", "N/A", "N/A", "N/A"
    if os.path.exists("Output/model_report.txt"):
        with open("Output/model_report.txt", "r", encoding="utf-8") as f:
            report_text = f.read()
        def extract_metric(text, key):
            match = re.search(rf"{key}\s*:\s*([\d.]+)%", text)
            return match.group(1) + "%" if match else "N/A"
        acc  = extract_metric(report_text, "Accuracy")
        prec = extract_metric(report_text, "Precision \\(Weighted\\)")
        rec  = extract_metric(report_text, "Recall \\(Weighted\\)")
        f1w  = extract_metric(report_text, "F1 Score \\(Weighted\\)")

    # Tampilkan kartu metrik kecil
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.markdown(f'<div class="metric-card"><h5 style="margin:0;color:#64748b;">AKURASI</h5><h2 style="margin:5px 0 0 0;font-weight:800;color:#0f172a;">{acc}</h2></div>', unsafe_allow_html=True)
    with col_m2:
        st.markdown(f'<div class="metric-card"><h5 style="margin:0;color:#64748b;">PRECISION</h5><h2 style="margin:5px 0 0 0;font-weight:800;color:#0f172a;">{prec}</h2></div>', unsafe_allow_html=True)
    with col_m3:
        st.markdown(f'<div class="metric-card"><h5 style="margin:0;color:#64748b;">RECALL</h5><h2 style="margin:5px 0 0 0;font-weight:800;color:#0f172a;">{rec}</h2></div>', unsafe_allow_html=True)
    with col_m4:
        st.markdown(f'<div class="metric-card"><h5 style="margin:0;color:#64748b;">F1-SCORE</h5><h2 style="margin:5px 0 0 0;font-weight:800;color:#0f172a;">{f1w}</h2></div>', unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 25px;'></div>", unsafe_allow_html=True)
    
    col_desc_left, col_desc_right = st.columns(2)
    
    with col_desc_left:
        st.markdown("#### **Analisis Metrik Evaluasi**")
        st.markdown(f"""
        - **Akurasi Global ({acc})**: Mengindikasikan bahwa model SVM secara keseluruhan berhasil memprediksi label sentimen dengan benar pada **{acc}** dari seluruh data uji komentar. Nilai ini sangat solid untuk klasifikasi 3-kelas (Positif, Netral, Negatif) yang memiliki tingkat subjektivitas tinggi.
        - **Precision (Weighted - {prec})**: Menunjukkan ketepatan prediksi model. Rata-rata tertimbang presisi sebesar **{prec}** berarti ketika model memprediksi suatu sentimen, tingkat kesalahan penuduhan (False Positive) sangat rendah.
        - **Recall (Weighted - {rec})**: Mengukur kelengkapan model. Nilai **{rec}** menunjukkan model mampu mengidentifikasi sebagian besar komentar aktual dari masing-masing kategori tanpa banyak yang terlewat (False Negative).
        - **F1-Score (Weighted - {f1w})**: Rata-rata harmonis yang seimbang antara precision dan recall. Nilai **{f1w}** yang tinggi membuktikan model memiliki stabilitas klasifikasi yang optimal di seluruh kelas.
        """)
        
    with col_desc_right:
        st.markdown("#### **Interpretasi Confusion Matrix**")
        st.markdown("""
        Confusion Matrix menggambarkan detail perbandingan klasifikasi aktual (baris) dengan prediksi model (kolom):
        
        - **Kelas Negatif (Aktual: 23 Komentar)**:
          - **14 komentar** berhasil dideteksi dengan benar sebagai **Negatif** (True Positive).
          - **6 komentar** salah diprediksi sebagai **Netral**, dan **3 komentar** salah diprediksi sebagai **Positif**. Ini menunjukkan model memiliki sensitivitas yang baik namun masih terdapat tumpang tindih kata-kata kritis yang ambigu.
        
        - **Kelas Netral (Aktual: 151 Komentar)**:
          - **134 komentar** dideteksi dengan benar sebagai **Netral**.
          - **5 komentar** salah diprediksi sebagai **Negatif**, dan **12 komentar** diprediksi sebagai **Positif**. Kelas ini memiliki performa deteksi terbaik karena volume data latih yang melimpah.
        
        - **Kelas Positif (Aktual: 61 Komentar)**:
          - **42 komentar** dideteksi dengan benar sebagai **Positif**.
          - **3 komentar** salah dideteksi sebagai **Negatif**, dan **16 komentar** diprediksi sebagai **Netral**. Hal ini lumrah terjadi karena ulasan positif seringkali menggunakan kata-kata santai yang mirip dengan ekspresi netral.
        """)
        
    st.markdown("---")
    st.markdown("#### **Interpretasi Ilmiah & Konfigurasi Model**")
    st.markdown("""
    Model ini dikonfigurasi menggunakan **Support Vector Machine (SVM)** dengan pembobotan fitur **TF-IDF (Term Frequency - Inverse Document Frequency)** dengan ukuran kosakata 5000 kata unik.
    
    1. **Kenapa Memilih Linear Kernel?**
       Klasifikasi teks berdimensi tinggi (5000 fitur TF-IDF) secara teoritis bersifat linear separable (dapat dipisahkan secara linear). Eksperimen membuktikan bahwa penggunaan linear kernel memberikan performa generalisasi data uji yang jauh lebih stabil dibandingkan RBF kernel yang rentan terhadap overfitting pada data latih yang dioversample.
       
    2. **Solusi Ketidakseimbangan Kelas (Class Imbalance):**
       Sebelum pelatihan, kelas minoritas disamakan proporsinya menggunakan **Random Oversampling**. Hal ini terbukti melipatgandakan sensitivitas (Recall) model terhadap komentar negatif pendek (seperti *'lagunya membosankan'*, *'tidak suka'*) dari yang sebelumnya gagal dideteksi menjadi terklasifikasi secara akurat.
    """)
