"""
Fake News Detection Using NLP - Interactive Streamlit Dashboard
Author: Pragna | Professional NLP Engineer & Data Science Intern
Description: A premium web application allowing users to paste news articles and
             classify them as FAKE or REAL using our serialized NLP pipeline.
Developed by Pragna.
"""

import os
import re
import string
import pickle
import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image

# --------------------------------------------------------------------------
# PAGE INITIALIZATION
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Fake News Detection System",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Core paths
MODELS_DIR = "models"
IMAGES_DIR = "images"

# Load saved preprocessor and model artifacts
@st.cache_resource
def load_ml_assets():
    model_path = os.path.join(MODELS_DIR, "best_model.pkl")
    vectorizer_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
    
    if not (os.path.exists(model_path) and os.path.exists(vectorizer_path)):
        return None, None
        
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)
        
    return model, vectorizer

model, tfidf_vectorizer = load_ml_assets()

# --------------------------------------------------------------------------
# SUSPICIOUS TERMS ANALYZER DEFINITIONS
# --------------------------------------------------------------------------
SUSPICIOUS_BUZZWORDS = [
    "shocking", "conspiracy", "secret", "whistleblower", "unbelievable", "elite",
    "leaked", "insider", "clandestine", "cover-up", "undeniable", "wake up",
    "exposed", "shadow", "agenda", "chemicals", "microchip"
]

def analyze_linguistic_suspicion(text):
    """
    Highlights suspicious terms in the input news text.
    """
    found_buzzwords = []
    text_lower = text.lower()
    for word in SUSPICIOUS_BUZZWORDS:
        if word in text_lower:
            found_buzzwords.append(word)
    return found_buzzwords

# --------------------------------------------------------------------------
# APPLICATION HEADER
# --------------------------------------------------------------------------
st.title("🔮 Fake News Detection Using NLP")
st.markdown("""
### 👑 Developed by Pragna
***
This intelligent NLP system classifies news articles as **FAKE** or **REAL** by analyzing their vocabulary distributions.
It extracts mathematical word weights via a serialized **TF-IDF Vectorizer** and classifies them using our top-performing machine learning model.
""")

# If models haven't been trained yet
if model is None:
    st.error("⚠️ Trained model binaries not found in `models/` directory! Please run the training pipeline script first:")
    st.code("python models/train.py", language="bash")
    st.stop()

# Sidebar branding
st.sidebar.markdown("## 👑 Developed by Pragna")
st.sidebar.info("Fake News Detection System - Certified Machine Learning & NLP Internship Project.")

# Create tabs for multi-tab layout
tab_classifier, tab_analytics, tab_about = st.tabs([
    "🎯 News Article Classifier",
    "📊 Linguistic Exploratory Insights",
    "📝 Preprocessing & Methodology"
])

# --------------------------------------------------------------------------
# TAB 1: NEWS ARTICLE CLASSIFIER
# --------------------------------------------------------------------------
with tab_classifier:
    st.subheader("Analyze News Content")
    st.write("Paste the full text of a news article or headline below to evaluate its authenticity index.")
    
    # Text input area
    news_input = st.text_area(
        "Enter News Article Text (At least 15 words for optimal performance):", 
        height=220,
        placeholder="Paste article contents here..."
    )
    
    st.markdown("***")
    predict_btn = st.button("🔮 Analyze Legitimacy", use_container_width=True)
    
    if predict_btn:
        if len(news_input.strip().split()) < 5:
            st.warning("⚠️ Please input a longer text body to enable representative NLP word weights parsing.")
        else:
            # 1. Preprocess the text locally
            # Lowercase
            clean_text = news_input.lower()
            
            # Punctuation & Brackets removal
            clean_text = re.sub(r'\[.*?\]', '', clean_text)
            clean_text = re.sub(r'https?://\S+|www\.\S+', '', clean_text)
            clean_text = re.sub(r'<.*?>+', '', clean_text)
            clean_text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', clean_text)
            clean_text = re.sub(r'\n', ' ', clean_text)
            clean_text = re.sub(r'\w*\d\w*', '', clean_text)
            
            # 2. Vectorize via saved TF-IDF Vectorizer
            vectorized_input = tfidf_vectorizer.transform([clean_text]).toarray()
            
            # 3. Model Predict
            prediction = model.predict(vectorized_input)[0]
            probabilities = model.predict_proba(vectorized_input)[0]
            
            st.markdown("## 📊 NLP Classification Result")
            
            col_res, col_chart = st.columns([1, 2])
            
            with col_res:
                if prediction == 1:
                    # REAL news
                    conf_score = int(probabilities[1] * 100)
                    st.success("🟢 **REAL NEWS**")
                    st.metric(label="Legitimacy Index Score", value=f"{conf_score}%")
                    st.write("The linguistic structure of this article matches formal, objective journalism metrics.")
                else:
                    # FAKE news
                    conf_score = int(probabilities[0] * 100)
                    st.error("🔴 **FAKE NEWS**")
                    st.metric(label="Deception Likelihood", value=f"{conf_score}%")
                    st.write("Warning! The mathematical vocabulary signature of this text heavily aligns with sensationalized clicks or conspiracy propaganda.")
            
            with col_chart:
                st.markdown("### Credibility Spectrum")
                st.progress(probabilities[1]) # Progress towards 1.0 (REAL)
                
                # Dynamic Linguistic Analysis
                st.markdown("### 💡 Linguistic Suspicion Diagnostics")
                detected_buzzwords = analyze_linguistic_suspicion(news_input)
                
                if prediction == 0:
                    if detected_buzzwords:
                        st.markdown("⚠️ **Sensationalism Detected**: The NLP analyzer flagged highly emotional clickbait buzzwords in the text:")
                        buzzword_md = ", ".join([f"**'{word}'**" for word in detected_buzzwords])
                        st.write(f"- Suspicious markers found: {buzzword_md}")
                        st.write("- Recommendations: cross-reference statements with verified press releases and check other major publications.")
                    else:
                        st.write("- The model flagged this article based on sentence construction patterns and informal structure indices, even without direct clickbait triggers.")
                else:
                    if detected_buzzwords:
                        st.markdown("ℹ️ **Objective Structure with Clickbait Overlays**: Although the model classified this as REAL, it detected a few casual indicators:")
                        buzzword_md = ", ".join([f"**'{word}'**" for word in detected_buzzwords])
                        st.write(f"- Highlighted terms: {buzzword_md}")
                        st.write("- This is common when verified publications write about ongoing conspiracy topics objectively.")
                    else:
                        st.write("- ✅ Clean structural scoring. Formal sentences, neutral lexical weights, and zero clicked keywords found.")

# --------------------------------------------------------------------------
# TAB 2: EXPLORATORY INSIGHTS
# --------------------------------------------------------------------------
with tab_analytics:
    st.subheader("Global Linguistic Exploratory Insights")
    st.write("Analysis generated during our data science training cycle showing lexical differences between Fake and Real articles.")
    
    if not os.path.exists(IMAGES_DIR):
        st.warning("⚠️ Image assets folder not found! Run the training script to generate these charts.")
    else:
        col_img1, col_img2 = st.columns(2)
        
        with col_img1:
            st.markdown("### A. Vocabulary Word Cloud Comparison")
            img_cloud_real = os.path.join(IMAGES_DIR, "wordcloud_real.png")
            img_cloud_fake = os.path.join(IMAGES_DIR, "wordcloud_fake.png")
            
            if os.path.exists(img_cloud_real):
                st.image(Image.open(img_cloud_real), caption="Real News WordCloud: formal vocabulary focused on policies, reports, and economics.", use_container_width=True)
            if os.path.exists(img_cloud_fake):
                st.image(Image.open(img_cloud_fake), caption="Fake News WordCloud: loaded clickbait buzzwords focusing on conspiracies, whistleblowers, and leaks.", use_container_width=True)
                
            st.markdown("### B. Model Classification Accuracy Benchmark")
            img_accuracy = os.path.join(IMAGES_DIR, "model_accuracy_comparison.png")
            if os.path.exists(img_accuracy):
                st.image(Image.open(img_accuracy), caption="ベンチマーク - Benchmarking classifiers accuracy showing how well TF-IDF features separate legitimacy.", use_container_width=True)
                
        with col_img2:
            st.markdown("### C. Top 15 Frequent Word Distributions")
            img_top_words = os.path.join(IMAGES_DIR, "top_words_comparison.png")
            if os.path.exists(img_top_words):
                st.image(Image.open(img_top_words), caption="Histogram comparison displaying distinct vocabulary densities between classes.", use_container_width=True)
                
            st.markdown("### D. Model Confusion Matrix Grid")
            img_cm = os.path.join(IMAGES_DIR, "confusion_matrices.png")
            if os.path.exists(img_cm):
                st.image(Image.open(img_cm), caption="Confusion matrices representing True/False predictions for Logistic Regression, Naive Bayes, and SVMs.", use_container_width=True)

# --------------------------------------------------------------------------
# TAB 3: METHODOLOGY
# --------------------------------------------------------------------------
with tab_about:
    st.markdown("""
    ### 🔬 Preprocessing Pipeline & NLP Methodology
    
    #### 📋 Text Preprocessing Steps:
    1. **Data Cleaning & Deduplication**: Dropped null text rows and duplicate entries to retain training integrity.
    2. **Structural Cleaning**: Applied Regex filters to remove bracket notes, special characters, numerical values, and web URLs.
    3. **Tokenization**: Segmented complete text content into distinct token units.
    4. **Stopwords Elimination**: Removed high-frequency grammatical connector words using standard English stopwords templates.
    5. **Lemmatization**: Applied NLTK's Lemmatization routines to bring different conjugated formats of a word back to its original dictionary root form.
    
    #### 🧬 Feature Engineering:
    - **TF-IDF Vectorization**: Fitted a `TfidfVectorizer` mapping up to **3,000 top n-gram feature combinations (unigrams and bigrams)**, mathematically mapping the relative term significance score.
    
    #### 🛠️ Technology Stack:
    - **Language**: Python 3.10+
    - **Data Operations**: Pandas, NumPy
    - **NLP Package**: NLTK (Natural Language Toolkit)
    - **Machine Learning**: Scikit-Learn
    - **Visualizations**: Matplotlib, Seaborn, WordCloud
    - **Web Framework**: Streamlit
    
    #### 📧 Developer Contact:
    This project was proudly **Developed by Pragna**. For resume discussions or internship reviews, feel free to reach out to Pragna, your Data Science student candidate!
    """)
