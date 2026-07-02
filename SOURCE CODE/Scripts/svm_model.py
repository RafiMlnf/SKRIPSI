# -*- coding: utf-8 -*-
"""
=============================================================================
MODEL SUPPORT VECTOR MACHINE (SVM) UNTUK ANALISIS SENTIMEN
Analisis Sentimen Lagu "Arteri" - .FEAST
=============================================================================
Author: Rafi Maulana Firdaus
NIM: 312210382
Tanggal: Februari 2026
Keperluan: Seminar Proposal (SEMPRO)

Metodologi:
1. TF-IDF Feature Extraction
2. Support Vector Machine (SVM) Classification
3. Hyperparameter Tuning dengan GridSearchCV

Output:
- Model terlatih (.pkl)
- Vectorizer (.pkl)
- Laporan evaluasi
- Confusion matrix
=============================================================================
"""

import pandas as pd
import numpy as np
import os
import pickle
import joblib
from datetime import datetime

# ============================================================================
# KONFIGURASI
# ============================================================================

class Config:
    """Konfigurasi model SVM"""
    
    # File paths
    INPUT_FILE = "Data/comments_labeled.csv"
    OUTPUT_DIR = "Output"
    MODEL_FILE = "Output/svm_sentiment_model.pkl"
    VECTORIZER_FILE = "Output/tfidf_vectorizer.pkl"
    REPORT_FILE = "Output/model_report.txt"
    
    # TF-IDF Configuration
    MAX_FEATURES = 5000           # Vocabulary size
    NGRAM_RANGE = (1, 2)          # Unigram + Bigram
    MIN_DF = 2                    # Minimum document frequency
    MAX_DF = 0.95                 # Maximum document frequency
    
    # SVM Configuration
    KERNEL = 'linear'             # Kernel type: linear, rbf, poly
    C_VALUES = [0.1, 1, 10]       # Regularization parameter untuk GridSearch
    GAMMA_VALUES = ['scale', 'auto']  # Gamma untuk RBF kernel
    
    # Training Configuration
    TEST_SIZE = 0.2               # 20% untuk test
    RANDOM_STATE = 42
    CV_FOLDS = 5                  # Cross-validation folds
    
    # Classes
    NUM_CLASSES = 3               # positif, netral, negatif
    LABEL_NAMES = ['negatif', 'netral', 'positif']


# ============================================================================
# DATA PREPARATION
# ============================================================================

def load_data(config):
    """
    Memuat data dari file CSV
    """
    print("\n" + "="*70)
    print("MEMUAT DATA")
    print("="*70)
    
    print(f"\n[INFO] Memuat data dari: {config.INPUT_FILE}")
    df = pd.read_csv(config.INPUT_FILE)
    print(f"[INFO] Total data: {len(df)}")
    
    # Get texts and labels
    texts = df['text_processed'].astype(str).values
    labels = df['sentiment_label'].values
    
    # Print distribution
    print("\n[INFO] Distribusi label:")
    unique, counts = np.unique(labels, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"       {config.LABEL_NAMES[u]}: {c} ({c/len(labels)*100:.1f}%)")
    
    return texts, labels, df


def split_data(X, y, config):
    """
    Split data menjadi train dan test
    """
    from sklearn.model_selection import train_test_split
    
    print("\n[INFO] Membagi data...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=config.TEST_SIZE, 
        random_state=config.RANDOM_STATE,
        stratify=y
    )
    
    print(f"       Train: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"       Test: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")
    
    return X_train, X_test, y_train, y_test


# ============================================================================
# FEATURE EXTRACTION - TF-IDF
# ============================================================================

def create_tfidf_vectorizer(config):
    """
    Membuat TF-IDF Vectorizer
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    vectorizer = TfidfVectorizer(
        max_features=config.MAX_FEATURES,
        ngram_range=config.NGRAM_RANGE,
        min_df=config.MIN_DF,
        max_df=config.MAX_DF,
        lowercase=True,
        strip_accents='unicode',
        sublinear_tf=True  # Apply sublinear tf scaling
    )
    
    return vectorizer


def extract_features(texts_train, texts_test, config, vectorizer=None):
    """
    Ekstraksi fitur TF-IDF
    """
    print("\n" + "="*70)
    print("EKSTRAKSI FITUR TF-IDF")
    print("="*70)
    
    if vectorizer is None:
        vectorizer = create_tfidf_vectorizer(config)
    
    print("\n[INFO] Fitting TF-IDF Vectorizer pada data training...")
    X_train_tfidf = vectorizer.fit_transform(texts_train)
    
    print(f"[INFO] Transformasi data test...")
    X_test_tfidf = vectorizer.transform(texts_test)
    
    print(f"\n[INFO] Hasil TF-IDF:")
    print(f"       Vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"       Train matrix shape: {X_train_tfidf.shape}")
    print(f"       Test matrix shape: {X_test_tfidf.shape}")
    
    # Show top features
    feature_names = vectorizer.get_feature_names_out()
    print(f"\n[INFO] Contoh fitur (10 pertama):")
    for feat in feature_names[:10]:
        print(f"       - {feat}")
    
    return X_train_tfidf, X_test_tfidf, vectorizer


# ============================================================================
# MODEL SVM
# ============================================================================

def build_svm_model(config, use_grid_search=True):
    """
    Membangun model SVM dengan atau tanpa GridSearchCV
    """
    from sklearn.svm import SVC
    from sklearn.model_selection import GridSearchCV
    
    print("\n" + "="*70)
    print("MEMBANGUN MODEL SVM")
    print("="*70)
    
    if use_grid_search:
        print("\n[INFO] Menggunakan GridSearchCV untuk hyperparameter tuning...")
        
        # Base SVM model
        svm_base = SVC(random_state=config.RANDOM_STATE, probability=True)
        
        # Parameter grid
        param_grid = {
            'kernel': ['linear', 'rbf'],
            'C': config.C_VALUES,
            'gamma': config.GAMMA_VALUES
        }
        
        # GridSearchCV
        model = GridSearchCV(
            svm_base,
            param_grid,
            cv=config.CV_FOLDS,
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=1
        )
        
        print(f"\n[INFO] Parameter grid:")
        for key, values in param_grid.items():
            print(f"       {key}: {values}")
        print(f"       Cross-validation folds: {config.CV_FOLDS}")
        
    else:
        print("\n[INFO] Menggunakan SVM dengan parameter default...")
        model = SVC(
            kernel=config.KERNEL,
            C=1.0,
            gamma='scale',
            random_state=config.RANDOM_STATE,
            probability=True
        )
    
    return model


def train_model(model, X_train, y_train, config):
    """
    Training model SVM
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    print("\n" + "="*70)
    print("TRAINING MODEL SVM")
    print("="*70)
    
    # Compute class weights for imbalanced data
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = dict(enumerate(class_weights))
    print(f"\n[INFO] Class weights: {class_weight_dict}")
    
    print(f"\n[INFO] Memulai training...")
    print(f"       Train samples: {X_train.shape[0]}")
    print(f"       Features: {X_train.shape[1]}")
    
    # Training
    import time
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    print(f"\n[INFO] Training selesai dalam {training_time:.2f} detik")
    
    # If GridSearchCV, show best parameters
    if hasattr(model, 'best_params_'):
        print(f"\n[INFO] Best parameters dari GridSearchCV:")
        for key, value in model.best_params_.items():
            print(f"       {key}: {value}")
        print(f"       Best CV score: {model.best_score_*100:.2f}%")
    
    return model, training_time


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_model(model, X_test, y_test, config):
    """
    Evaluasi performa model SVM
    """
    from sklearn.metrics import (
        classification_report, confusion_matrix, 
        accuracy_score, f1_score, precision_score, recall_score
    )
    
    print("\n" + "="*70)
    print("EVALUASI MODEL")
    print("="*70)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Get probabilities if available
    y_pred_proba = None
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test)
    elif hasattr(model, 'best_estimator_'):
        y_pred_proba = model.best_estimator_.predict_proba(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\n[HASIL EVALUASI]")
    print(f"Accuracy           : {accuracy*100:.2f}%")
    print(f"Precision (Weighted): {precision*100:.2f}%")
    print(f"Recall (Weighted)   : {recall*100:.2f}%")
    print(f"F1 Score (Macro)    : {f1_macro*100:.2f}%")
    print(f"F1 Score (Weighted) : {f1_weighted*100:.2f}%")
    
    # Classification report
    print("\n[CLASSIFICATION REPORT]")
    print(classification_report(y_test, y_pred, target_names=config.LABEL_NAMES))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\n[CONFUSION MATRIX]")
    print("                Predicted")
    print("              Neg  Net  Pos")
    print(f"Actual Neg    {cm[0][0]:3d}  {cm[0][1]:3d}  {cm[0][2]:3d}")
    print(f"       Net    {cm[1][0]:3d}  {cm[1][1]:3d}  {cm[1][2]:3d}")
    print(f"       Pos    {cm[2][0]:3d}  {cm[2][1]:3d}  {cm[2][2]:3d}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'confusion_matrix': cm,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }


def plot_confusion_matrix(cm, config, output_path='Output/confusion_matrix.png'):
    """
    Plot confusion matrix
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.figure(figsize=(10, 8))
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Create heatmap with both counts and percentages
        labels = np.asarray([
            [f'{cm[i,j]}\n({cm_normalized[i,j]*100:.1f}%)' 
             for j in range(cm.shape[1])] 
            for i in range(cm.shape[0])
        ])
        
        sns.heatmap(
            cm, 
            annot=labels, 
            fmt='', 
            cmap='Blues',
            xticklabels=config.LABEL_NAMES,
            yticklabels=config.LABEL_NAMES,
            annot_kws={'size': 12},
            square=True
        )
        
        plt.title('Confusion Matrix\nSVM Sentiment Analysis "Arteri" - .FEAST', 
                  fontsize=14, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\n[INFO] Confusion matrix tersimpan: {output_path}")
        
    except ImportError as e:
        print(f"[WARNING] Library tidak terinstall: {e}. Confusion matrix dilewati.")


def plot_metrics_comparison(eval_results, output_path='Output/metrics_comparison.png'):
    """
    Plot perbandingan metrik evaluasi
    """
    try:
        import matplotlib.pyplot as plt
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1 (Macro)', 'F1 (Weighted)']
        values = [
            eval_results['accuracy'] * 100,
            eval_results['precision'] * 100,
            eval_results['recall'] * 100,
            eval_results['f1_macro'] * 100,
            eval_results['f1_weighted'] * 100
        ]
        
        colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(metrics, values, color=colors, edgecolor='black', linewidth=1.2)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{val:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.set_ylim(0, 110)
        ax.set_ylabel('Score (%)', fontsize=12)
        ax.set_title('Metrik Evaluasi Model SVM\nAnalisis Sentimen "Arteri" - .FEAST', 
                     fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"[INFO] Grafik metrik tersimpan: {output_path}")
        
    except ImportError as e:
        print(f"[WARNING] Library tidak terinstall: {e}. Grafik dilewati.")


# ============================================================================
# GENERATE REPORT
# ============================================================================

def generate_report(config, eval_results, training_time, model, vectorizer, output_path):
    """
    Generate laporan lengkap model
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = []
    report.append("="*70)
    report.append("LAPORAN MODEL SVM ANALISIS SENTIMEN")
    report.append("Lagu 'Arteri' - .FEAST")
    report.append("="*70)
    report.append(f"\nTanggal/Waktu: {timestamp}")
    report.append(f"Author: Rafi Maulana Firdaus (312210382)")
    
    report.append("\n" + "-"*70)
    report.append("KONFIGURASI TF-IDF")
    report.append("-"*70)
    report.append(f"Max Features        : {config.MAX_FEATURES}")
    report.append(f"N-gram Range        : {config.NGRAM_RANGE}")
    report.append(f"Min DF              : {config.MIN_DF}")
    report.append(f"Max DF              : {config.MAX_DF}")
    report.append(f"Vocabulary Size     : {len(vectorizer.vocabulary_)}")
    
    report.append("\n" + "-"*70)
    report.append("KONFIGURASI SVM")
    report.append("-"*70)
    
    # Get best params if GridSearchCV
    if hasattr(model, 'best_params_'):
        report.append("(Best parameters dari GridSearchCV)")
        for key, value in model.best_params_.items():
            report.append(f"{key.capitalize():20s}: {value}")
        report.append(f"Best CV Score       : {model.best_score_*100:.2f}%")
    else:
        report.append(f"Kernel              : {config.KERNEL}")
        report.append(f"C                   : 1.0")
        report.append(f"Gamma               : scale")
    
    report.append(f"\nTraining Time       : {training_time:.2f} seconds")
    
    report.append("\n" + "-"*70)
    report.append("HASIL EVALUASI")
    report.append("-"*70)
    report.append(f"Accuracy            : {eval_results['accuracy']*100:.2f}%")
    report.append(f"Precision (Weighted): {eval_results['precision']*100:.2f}%")
    report.append(f"Recall (Weighted)   : {eval_results['recall']*100:.2f}%")
    report.append(f"F1 Score (Macro)    : {eval_results['f1_macro']*100:.2f}%")
    report.append(f"F1 Score (Weighted) : {eval_results['f1_weighted']*100:.2f}%")
    
    report.append("\n" + "-"*70)
    report.append("CONFUSION MATRIX")
    report.append("-"*70)
    cm = eval_results['confusion_matrix']
    report.append("                Predicted")
    report.append("              Neg  Net  Pos")
    report.append(f"Actual Neg    {cm[0][0]:3d}  {cm[0][1]:3d}  {cm[0][2]:3d}")
    report.append(f"       Net    {cm[1][0]:3d}  {cm[1][1]:3d}  {cm[1][2]:3d}")
    report.append(f"       Pos    {cm[2][0]:3d}  {cm[2][1]:3d}  {cm[2][2]:3d}")
    
    report.append("\n" + "-"*70)
    report.append("FILE OUTPUT")
    report.append("-"*70)
    report.append(f"Model               : {config.MODEL_FILE}")
    report.append(f"Vectorizer          : {config.VECTORIZER_FILE}")
    report.append(f"Report              : {config.REPORT_FILE}")
    
    report.append("\n" + "="*70)
    report.append("END OF REPORT")
    report.append("="*70)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print(f"\n[INFO] Laporan model tersimpan: {output_path}")


# ============================================================================
# PREDICTION FUNCTION
# ============================================================================

def predict_sentiment(text, model, vectorizer, config):
    """
    Prediksi sentimen untuk teks baru
    """
    # Preprocess
    text_clean = text.lower().strip()
    
    # Transform dengan TF-IDF
    text_tfidf = vectorizer.transform([text_clean])
    
    # Predict
    predicted_class = model.predict(text_tfidf)[0]
    
    # Get probabilities if available
    probabilities = {}
    confidence = 0.0
    
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(text_tfidf)[0]
        confidence = proba[predicted_class]
        for i, label in enumerate(config.LABEL_NAMES):
            probabilities[label] = float(proba[i])
    elif hasattr(model, 'best_estimator_'):
        proba = model.best_estimator_.predict_proba(text_tfidf)[0]
        confidence = proba[predicted_class]
        for i, label in enumerate(config.LABEL_NAMES):
            probabilities[label] = float(proba[i])
    
    return {
        'text': text,
        'sentiment': config.LABEL_NAMES[predicted_class],
        'confidence': float(confidence),
        'probabilities': probabilities
    }


def save_model(model, vectorizer, config):
    """
    Menyimpan model dan vectorizer
    """
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # Save model
    joblib.dump(model, config.MODEL_FILE)
    print(f"\n[INFO] Model tersimpan: {config.MODEL_FILE}")
    
    # Save vectorizer
    joblib.dump(vectorizer, config.VECTORIZER_FILE)
    print(f"[INFO] Vectorizer tersimpan: {config.VECTORIZER_FILE}")


def load_model(config):
    """
    Memuat model dan vectorizer yang sudah disimpan
    """
    model = joblib.load(config.MODEL_FILE)
    vectorizer = joblib.load(config.VECTORIZER_FILE)
    
    return model, vectorizer


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main function"""
    print("""
    ╔═══════════════════════════════════════════════════════════════════╗
    ║     MODEL SUPPORT VECTOR MACHINE (SVM)                            ║
    ║     ANALISIS SENTIMEN LAGU "ARTERI" - .FEAST                      ║
    ║     Machine Learning dengan Scikit-learn                          ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """)
    
    config = Config()
    
    # Check input file
    if not os.path.exists(config.INPUT_FILE):
        print(f"[ERROR] File tidak ditemukan: {config.INPUT_FILE}")
        print("        Jalankan labeling.py terlebih dahulu!")
        return
    
    # Load data
    texts, labels, df = load_data(config)
    
    # Split data
    X_train_text, X_test_text, y_train, y_test = split_data(texts, labels, config)
    
    # Extract TF-IDF features
    X_train, X_test, vectorizer = extract_features(X_train_text, X_test_text, config)
    
    # Build model with GridSearchCV
    model = build_svm_model(config, use_grid_search=True)
    
    # Train model
    model, training_time = train_model(model, X_train, y_train, config)
    
    # Evaluate
    eval_results = evaluate_model(model, X_test, y_test, config)
    
    # Save model and vectorizer
    save_model(model, vectorizer, config)
    
    # Plot confusion matrix
    plot_confusion_matrix(eval_results['confusion_matrix'], config)
    
    # Plot metrics comparison
    plot_metrics_comparison(eval_results)
    
    # Generate report
    generate_report(config, eval_results, training_time, model, vectorizer, config.REPORT_FILE)
    
    # Demo prediction
    print("\n" + "="*70)
    print("DEMO PREDIKSI")
    print("="*70)
    
    test_texts = [
        "lagu ini bagus banget keren",
        "jelek ga suka sama sekali",
        "baru pertama denger lagu ini",
        "relate banget sama kehidupan gue",
        "mirip MCR I'm not okay"
    ]
    
    for text in test_texts:
        result = predict_sentiment(text, model, vectorizer, config)
        print(f"\nTeks: '{text}'")
        print(f"Sentimen: {result['sentiment']} (confidence: {result['confidence']:.2%})")
    
    print("\n" + "="*70)
    print("[SELESAI] Training model SVM berhasil!")
    print(f"          - Model        : {config.MODEL_FILE}")
    print(f"          - Vectorizer   : {config.VECTORIZER_FILE}")
    print(f"          - Laporan      : {config.REPORT_FILE}")
    print("="*70)


if __name__ == "__main__":
    main()
