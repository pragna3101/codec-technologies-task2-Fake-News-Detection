"""
Fake News Detection Using NLP - Preprocessing & Model Training Pipeline
Author: Pragna | Professional NLP Engineer & Data Science Intern
Description: Cleans textual content, vectorizes via TF-IDF, trains and compares
             Logistic Regression, Naive Bayes, and SVM classifiers, saves
             performance visualizations, and dumps serialized model binaries.
Developed by Pragna.
"""

import os
import re
import string
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

# NLTK imports with robust fallback setup
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Attempt to download required NLTK resources dynamically
try:
    print("Downloading NLTK resources...")
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    NLTK_AVAILABLE = True
except Exception as e:
    print(f"Warning: NLTK download failed ({e}). Using robust offline fallbacks.")
    NLTK_AVAILABLE = False

# Hardcoded English Stopwords fallback list
FALLBACK_STOPWORDS = set([
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", 
    "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", 
    "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", 
    "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", 
    "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", 
    "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", 
    "with", "about", "against", "between", "into", "through", "during", "before", "after", 
    "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", 
    "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", 
    "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", 
    "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", 
    "should", "now"
])

# Initialize preprocessors
if NLTK_AVAILABLE:
    try:
        STOPWORDS_SET = set(stopwords.words('english'))
        LEMMATIZER = WordNetLemmatizer()
    except Exception:
        STOPWORDS_SET = FALLBACK_STOPWORDS
        LEMMATIZER = None
else:
    STOPWORDS_SET = FALLBACK_STOPWORDS
    LEMMATIZER = None

# Set plotting styling
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'figure.figsize': (10, 6),
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10
})

def clean_text(text):
    """
    Standard NLP preprocessing pipeline: lowercasing, punctuation removal,
    tokenization, stopword removal, and lemmatization.
    """
    if not isinstance(text, str):
        return ""
        
    # 1. Lowercasing
    text = text.lower()
    
    # 2. Punctuation and Special Character Removal
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', '', text)
    
    # 3. Tokenization & Stopwords removal
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS_SET]
    
    # 4. Lemmatization (with fallback)
    if LEMMATIZER:
        try:
            tokens = [LEMMATIZER.lemmatize(t) for t in tokens]
        except Exception:
            pass # Keep tokens as is
            
    # Re-join tokens into a clean string
    return " ".join(tokens)

def run_nlp_pipeline():
    print("====================================================================")
    print("          STARTING FAKE NEWS DETECTION ML PIPELINE                  ")
    print("====================================================================")
    
    data_path = "D:\\GITHUB\\Fake-News-Detection\\data\\fake_or_real_news.csv"
    images_dir = "D:\\GITHUB\\Fake-News-Detection\\images"
    models_dir = "D:\\GITHUB\\Fake-News-Detection\\models"
    
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    # ----------------------------------------------------------------------
    # 1. DATA LOADING & CLEANING
    # ----------------------------------------------------------------------
    print("\n--- Step 1: Loading News Dataset ---")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Run generate_news_data.py first.")
        
    df = pd.read_csv(data_path)
    print(f"Dataset loaded. Shape: {df.shape[0]} articles, {df.shape[1]} columns")
    
    # Remove null values
    df.dropna(subset=["title", "text"], inplace=True)
    
    # Duplicate removal
    num_dupes = df.duplicated().sum()
    if num_dupes > 0:
        df.drop_duplicates(inplace=True)
        print(f"Removed {num_dupes} duplicate news articles.")
        
    # Map Labels: FAKE -> 0, REAL -> 1
    df["target"] = df["label"].map({"FAKE": 0, "REAL": 1})
    
    # Concatenate title and text for complete content representation
    df["content"] = df["title"] + " " + df["text"]
    
    # ----------------------------------------------------------------------
    # 2. EXPLORATORY DATA ANALYSIS (EDA) & PLOTS
    # ----------------------------------------------------------------------
    print("\n--- Step 2: Running EDA & Visuals ---")
    
    # Plot 2.1: Class Distribution Chart
    plt.figure(figsize=(6, 5))
    colors = ["#FF6B6B", "#4D96FF"] # Soft Red for Fake, Blue for Real
    sns.countplot(data=df, x="label", palette=colors, hue="label", legend=False)
    plt.title("Linguistic Class Distribution (Fake vs Real)", weight="bold", pad=15)
    plt.xlabel("News Legitimacy Label")
    plt.ylabel("Article Count")
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, "class_distribution.png"), dpi=150)
    plt.close()
    print("Saved class_distribution.png")
    
    # ----------------------------------------------------------------------
    # 3. TEXT PREPROCESSING
    # ----------------------------------------------------------------------
    print("\n--- Step 3: Performing Text Preprocessing & Cleaning ---")
    df["clean_content"] = df["content"].apply(clean_text)
    print("Finished Tokenization, Stopwords removal, and Lemmatization.")
    
    # ----------------------------------------------------------------------
    # 4. WORD CLOUD GENERATION
    # ----------------------------------------------------------------------
    print("\n--- Step 4: Generating WordClouds ---")
    try:
        from wordcloud import WordCloud
        
        # Real News WordCloud
        real_text = " ".join(df[df["target"] == 1]["clean_content"])
        wc_real = WordCloud(width=800, height=400, background_color="white", colormap="viridis", max_words=100).generate(real_text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wc_real, interpolation="bilinear")
        plt.axis("off")
        plt.title("Real News Word Cloud", weight="bold", pad=15)
        plt.tight_layout(pad=0)
        plt.savefig(os.path.join(images_dir, "wordcloud_real.png"), dpi=150)
        plt.close()
        
        # Fake News WordCloud
        fake_text = " ".join(df[df["target"] == 0]["clean_content"])
        wc_fake = WordCloud(width=800, height=400, background_color="black", colormap="cool", max_words=100).generate(fake_text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wc_fake, interpolation="bilinear")
        plt.axis("off")
        plt.title("Fake News Word Cloud", weight="bold", pad=15, color="white")
        plt.tight_layout(pad=0)
        plt.savefig(os.path.join(images_dir, "wordcloud_fake.png"), dpi=150)
        plt.close()
        print("Successfully generated Word Cloud visuals.")
    except Exception as e:
        print(f"Warning: WordCloud generation skipped due to error ({e}).")
        
    # Plot 2.2: Top Frequent Words comparison
    print("Calculating top frequent words...")
    real_words = pd.Series(" ".join(df[df["target"] == 1]["clean_content"]).split()).value_counts()[:15]
    fake_words = pd.Series(" ".join(df[df["target"] == 0]["clean_content"]).split()).value_counts()[:15]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.barplot(x=real_words.values, y=real_words.index, palette="viridis", ax=axes[0], hue=real_words.index, legend=False)
    axes[0].set_title("Top 15 Real News Terms", weight="bold")
    axes[0].set_xlabel("Frequencies")
    
    sns.barplot(x=fake_words.values, y=fake_words.index, palette="flare", ax=axes[1], hue=fake_words.index, legend=False)
    axes[1].set_title("Top 15 Fake News Terms", weight="bold")
    axes[1].set_xlabel("Frequencies")
    
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, "top_words_comparison.png"), dpi=150)
    plt.close()
    print("Saved top_words_comparison.png")
    
    # ----------------------------------------------------------------------
    # 5. FEATURE INGESTION & TF-IDF VECTORIZATION
    # ----------------------------------------------------------------------
    print("\n--- Step 5: TF-IDF Vectorization ---")
    tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
    
    X = tfidf.fit_transform(df["clean_content"]).toarray()
    y = df["target"].values
    
    # Serialize TF-IDF Vectorizer
    with open(os.path.join(models_dir, "tfidf_vectorizer.pkl"), "wb") as f:
        pickle.dump(tfidf, f)
    print(f"TF-IDF Vectorizer fitted. Vocabulary shape: {X.shape[1]} dimensions. Saved to tfidf_vectorizer.pkl")
    
    # Train-test split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # ----------------------------------------------------------------------
    # 6. MODEL TRAINING & SCORING
    # ----------------------------------------------------------------------
    print("\n--- Step 6: Training Classifiers ---")
    
    # Using SVC with probability=True so we can query confidence scores in Streamlit
    models = {
        "Logistic Regression": LogisticRegression(random_state=42),
        "Naive Bayes": MultinomialNB(),
        "Support Vector Machine": SVC(kernel='linear', probability=True, random_state=42)
    }
    
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Scoring metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        results[name] = {
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "ConfMatrix": confusion_matrix(y_test, y_pred)
        }
        
        print(f"\nReport for {name}:")
        print(classification_report(y_test, y_pred))
        print("-" * 50)
        
    # ----------------------------------------------------------------------
    # 7. PERFORMANCE COMPARISON & VISUALIZATIONS
    # ----------------------------------------------------------------------
    print("\n--- Step 7: Benchmarking Visual Charts ---")
    
    comparison_df = pd.DataFrame(results).T.drop(columns=["ConfMatrix"])
    print(comparison_df.round(4))
    
    # Select Best Model based on F1-Score
    best_model_name = comparison_df["F1-Score"].idxmax()
    best_model = trained_models[best_model_name]
    print(f"\nBest Model Selected: {best_model_name} (F1: {results[best_model_name]['F1-Score']:.4f})")
    
    # Serialize the best model binary
    with open(os.path.join(models_dir, "best_model.pkl"), "wb") as f:
        pickle.dump(best_model, f)
    print("Best model successfully saved to best_model.pkl")
    
    # Plot 7.1: Accuracy Comparison Chart
    plt.figure(figsize=(7, 5))
    sns.barplot(x=comparison_df.index, y=comparison_df["Accuracy"], palette="coolwarm", hue=comparison_df.index, legend=False)
    plt.title("Model Classification Accuracy Comparison", weight="bold", pad=15)
    plt.ylim(0.8, 1.05)
    plt.ylabel("Accuracy Score")
    for i, v in enumerate(comparison_df["Accuracy"]):
        plt.text(i, v + 0.01, f"{v*100:.2f}%", ha="center", weight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, "model_accuracy_comparison.png"), dpi=150)
    plt.close()
    print("Saved model_accuracy_comparison.png")
    
    # Plot 7.2: Confusion Matrices
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, (name, res) in enumerate(results.items()):
        cm = res["ConfMatrix"]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[i], cbar=False,
                    xticklabels=["Fake", "Real"], yticklabels=["Fake", "Real"])
        axes[i].set_title(f"{name}\nConfusion Matrix", weight="bold")
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, "confusion_matrices.png"), dpi=150)
    plt.close()
    print("Saved confusion_matrices.png")
    
    print("\n====================================================================")
    print("          NLP PIPELINE SUCCESSFULLY RUN & COMPLETED!                 ")
    print("====================================================================")

if __name__ == "__main__":
    run_nlp_pipeline()
