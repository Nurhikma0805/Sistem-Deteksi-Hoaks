import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

# === Konfigurasi Halaman ===
st.set_page_config(
    page_title="Sistem Deteksi Hoaks",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# === Path files ===
DATASET_PATH = "dataset_gabungan_training.csv"
PREPROCESSED_PATH = "dataset_preprocessed.csv"
MODEL_PATH = "model_naivebayes.pkl"
VECTORIZER_PATH = "vectorizer_tfidf.pkl"

# === Preprocessing ===
STOPWORDS = set("yang dan di ke dari untuk pada adalah ini itu dengan juga tidak atau sebagai ada telah tapi kalau saya kita dia mereka kami oleh karena olehnya saja masih baru akan lebih sudah lagi semua setiap suatu tanpa seorang seorangnya sebuah".split())

def clean_text(text):
    """Preprocessing cepat tanpa Sastrawi"""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^0-9a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    
    tokens = []
    for word in text.split():
        if word not in STOPWORDS:
            word = re.sub(r'(kan|an|i|nya)$', '', word)
            if len(word) > 2:
                tokens.append(word)
    
    return " ".join(tokens)

# === Load Data dengan Cache ===
@st.cache_data
def load_data():
    if os.path.exists(PREPROCESSED_PATH):
        df = pd.read_csv(PREPROCESSED_PATH)
    else:
        df = pd.read_csv(DATASET_PATH)
        df["clean_text"] = df["content"].apply(clean_text)
        df.to_csv(PREPROCESSED_PATH, index=False)
    
    return df

@st.cache_resource
def load_model_and_vectorizer():
    df = load_data()
    le = LabelEncoder()
    df["label_enc"] = le.fit_transform(df["kategori"])
    
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        with open(VECTORIZER_PATH, 'rb') as f:
            vectorizer = pickle.load(f)
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            df["clean_text"], df["label_enc"], 
            test_size=0.2, random_state=42, stratify=df["label_enc"]
        )
        
        vectorizer = TfidfVectorizer(max_features=5000)
        X_train_tfidf = vectorizer.fit_transform(X_train)
        
        model = MultinomialNB()
        model.fit(X_train_tfidf, y_train)
        
        with open(VECTORIZER_PATH, 'wb') as f:
            pickle.dump(vectorizer, f)
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(model, f)
    
    tfidf_all = vectorizer.transform(df["clean_text"])
    
    return df, vectorizer, model, le, tfidf_all

# === Load semua data ===
df, vectorizer, model, le, tfidf_all = load_model_and_vectorizer()

# === Fungsi Pencarian ===
def retrieve_news(query, top_k=8):
    q_clean = clean_text(query)
    q_vec = vectorizer.transform([q_clean])
    sims = cosine_similarity(q_vec, tfidf_all).flatten()
    top_idx = sims.argsort()[::-1][:top_k]
    
    results = []
    for i in top_idx:
        if sims[i] < 0.1:
            continue
        
        text = df.iloc[i]["content"]
        pred = model.predict(vectorizer.transform([df.iloc[i]["clean_text"]]))[0]
        prob = model.predict_proba(vectorizer.transform([df.iloc[i]["clean_text"]]))[0][pred]
        label = le.inverse_transform([pred])[0]
        
        snippet = text[:400] + ("..." if len(text) > 400 else "")
        
        results.append({
            "title": df.iloc[i]["title"] if "title" in df.columns else "Berita Tanpa Judul",
            "snippet": snippet,
            "full_content": text,
            "label": label.lower(),
            "prob": round(prob * 100, 1),
            "similarity": round(sims[i] * 100, 1)
        })
    
    return results

# === Generate HTML Template ===
def generate_html(query="", results=None):
    results_html = ""
    
    if results:
        results_html = f"""
        <div class="results-header">
            <h2>🎯 Ditemukan {len(results)} hasil untuk: "{query}"</h2>
        </div>
        <div class="results-grid">
        """
        
        for idx, item in enumerate(results):
            badge_class = "badge-hoaks" if item['label'] == 'hoaks' else "badge-fakta"
            badge_text = "⚠️ HOAKS" if item['label'] == 'hoaks' else "✅ FAKTA"
            alert_class = "alert-hoaks" if item['label'] == 'hoaks' else "alert-fakta"
            alert_icon = "⚠️" if item['label'] == 'hoaks' else "✅"
            alert_title = "⚠️ TERDETEKSI HOAKS" if item['label'] == 'hoaks' else "✅ TERDETEKSI FAKTA"
            alert_desc = "Berita ini terindikasi sebagai informasi PALSU/HOAKS. Mohon untuk tidak menyebarkan dan verifikasi dari sumber terpercaya." if item['label'] == 'hoaks' else "Berita ini terindikasi sebagai informasi VALID/FAKTA berdasarkan analisis sistem."
            
            tips_html = ""
            if item['label'] == 'hoaks':
                tips_html = """
                <div class="tips-box">
                    <h4>💡 Tips Menghindari Hoaks:</h4>
                    <ul>
                        <li>Cek sumber berita dari media terpercaya</li>
                        <li>Jangan langsung percaya judul sensasional</li>
                        <li>Cari berita serupa dari sumber lain</li>
                        <li>Verifikasi melalui fact-checking websites</li>
                    </ul>
                </div>
                """
            
            results_html += f"""
            <div class="result-card" onclick="openModal({idx})">
                <div class="result-header">
                    <div class="result-title">{item['title']}</div>
                    <div class="result-badge {badge_class}">
                        {badge_text}
                    </div>
                </div>
                <div class="result-snippet">{item['snippet']}</div>
                <div class="result-footer">
                    <span class="confidence">Confidence: {item['prob']}%</span>
                    <span style="color: #456882;">Klik untuk detail →</span>
                </div>
            </div>

            <div id="modal{idx}" class="modal">
                <div class="modal-content">
                    <div class="modal-header">
                        <h2 class="modal-title">📰 Detail Informasi Berita</h2>
                        <span class="close" onclick="closeModal({idx})">&times;</span>
                    </div>
                    <div class="modal-body">
                        <h3 style="margin-bottom: 20px; color: #1B3C53;">{item['title']}</h3>
                        
                        <div class="alert-box {alert_class}">
                            <div class="alert-icon">{alert_icon}</div>
                            <div class="alert-content">
                                <h3>{alert_title}</h3>
                                <p>{alert_desc}</p>
                            </div>
                        </div>

                        <h4 style="margin-bottom: 15px; color: #1B3C53;">📋 Informasi Analisis</h4>
                        <div class="info-grid">
                            <div class="info-item">
                                <div class="info-label">🏷️ Klasifikasi</div>
                                <div class="info-value">{item['label'].upper()}</div>
                            </div>
                            <div class="info-item">
                                <div class="info-label">📈 Tingkat Kepercayaan</div>
                                <div class="info-value">{item['prob']}%</div>
                            </div>
                            <div class="info-item">
                                <div class="info-label">🔬 Metode Deteksi</div>
                                <div class="info-value" style="font-size: 0.9em;">Naive Bayes dengan TF-IDF</div>
                            </div>
                            <div class="info-item">
                                <div class="info-label">🎯 Kemiripan Query</div>
                                <div class="info-value" style="font-size: 0.9em;">Cosine Similarity</div>
                            </div>
                        </div>

                        <div class="content-box">
                            <h4>📄 Isi Berita</h4>
                            <div class="content-text">{item['full_content']}</div>
                        </div>

                        {tips_html}
                    </div>
                </div>
            </div>
            """
        
        results_html += "</div>"
    elif query:
        results_html = """
        <div class="no-results">
            <div class="no-results-icon">🔍</div>
            <h3>Tidak ada hasil ditemukan</h3>
            <p>Coba gunakan kata kunci yang berbeda</p>
        </div>
        """
    
    html = f"""
    <!DOCTYPE html>
    <html lang="id">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Sistem Deteksi Hoaks</title>
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: transparent;
                padding: 0;
                margin: 0;
            }}
            .results-header {{
                background: white;
                padding: 20px 30px;
                border-radius: 15px;
                margin-bottom: 20px;
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
                border-left: 5px solid #1B3C53;
            }}
            .results-header h2 {{
                color: #1B3C53;
                font-size: 1.5em;
            }}
            .results-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(500px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }}
            @media (max-width: 768px) {{
                .results-grid {{
                    grid-template-columns: 1fr;
                }}
            }}
            .result-card {{
                background: white;
                border-radius: 15px;
                padding: 25px;
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
                transition: transform 0.3s, box-shadow 0.3s;
                cursor: pointer;
                border-left: 5px solid #456882;
            }}
            .result-card:hover {{
                transform: translateY(-5px);
                box-shadow: 0 8px 25px rgba(27, 60, 83, 0.2);
            }}
            .result-header {{
                display: flex;
                justify-content: space-between;
                align-items: start;
                margin-bottom: 15px;
                flex-wrap: wrap;
                gap: 10px;
            }}
            .result-title {{
                font-size: 1.3em;
                font-weight: 600;
                color: #1B3C53;
                flex: 1;
                margin-right: 15px;
            }}
            .result-badge {{
                padding: 8px 16px;
                border-radius: 20px;
                font-size: 0.85em;
                font-weight: 600;
                white-space: nowrap;
            }}
            .badge-hoaks {{
                background: #fee;
                color: #c33;
            }}
            .badge-fakta {{
                background: #e8f5e9;
                color: #2e7d32;
            }}
            .result-snippet {{
                color: #456882;
                line-height: 1.6;
                margin-bottom: 15px;
            }}
            .result-footer {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding-top: 15px;
                border-top: 1px solid #D2C1B6;
            }}
            .confidence {{
                color: #1B3C53;
                font-weight: 600;
            }}
            .modal {{
                display: none;
                position: fixed;
                z-index: 1000;
                left: 0;
                top: 0;
                width: 100%;
                height: 100%;
                background: rgba(27, 60, 83, 0.8);
                overflow-y: auto;
            }}
            .modal-content {{
                background: white;
                margin: 50px auto;
                padding: 0;
                border-radius: 20px;
                max-width: 800px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                animation: slideDown 0.3s;
            }}
            @keyframes slideDown {{
                from {{ transform: translateY(-50px); opacity: 0; }}
                to {{ transform: translateY(0); opacity: 1; }}
            }}
            .modal-header {{
                padding: 30px;
                border-bottom: 2px solid #D2C1B6;
                position: relative;
                background: linear-gradient(135deg, #1B3C53 0%, #234C6A 100%);
                border-radius: 20px 20px 0 0;
            }}
            .modal-title {{
                font-size: 1.8em;
                color: white;
                padding-right: 40px;
            }}
            .close {{
                position: absolute;
                right: 25px;
                top: 25px;
                font-size: 2em;
                cursor: pointer;
                color: white;
                transition: color 0.3s;
            }}
            .close:hover {{
                color: #D2C1B6;
            }}
            .modal-body {{
                padding: 30px;
                max-height: 70vh;
                overflow-y: auto;
            }}
            .alert-box {{
                padding: 20px;
                border-radius: 10px;
                margin-bottom: 25px;
                display: flex;
                align-items: center;
                gap: 15px;
            }}
            .alert-hoaks {{
                background: #fee;
                border-left: 4px solid #c33;
            }}
            .alert-fakta {{
                background: #e8f5e9;
                border-left: 4px solid #2e7d32;
            }}
            .alert-icon {{
                font-size: 2em;
            }}
            .alert-content h3 {{
                margin-bottom: 5px;
                font-size: 1.3em;
                color: #1B3C53;
            }}
            .info-grid {{
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 20px;
                margin-bottom: 25px;
            }}
            @media (max-width: 768px) {{
                .info-grid {{
                    grid-template-columns: 1fr;
                }}
            }}
            .info-item {{
                padding: 15px;
                background: #f8f9fa;
                border-radius: 10px;
                border-left: 3px solid #1B3C53;
            }}
            .info-label {{
                font-size: 0.9em;
                color: #456882;
                margin-bottom: 5px;
            }}
            .info-value {{
                font-size: 1.2em;
                font-weight: 600;
                color: #1B3C53;
            }}
            .content-box {{
                background: #f8f9fa;
                padding: 20px;
                border-radius: 10px;
                margin-bottom: 25px;
                border: 1px solid #D2C1B6;
            }}
            .content-box h4 {{
                margin-bottom: 15px;
                color: #1B3C53;
            }}
            .content-text {{
                color: #456882;
                line-height: 1.8;
            }}
            .tips-box {{
                background: #fff8e1;
                border-left: 4px solid #f57c00;
                padding: 20px;
                border-radius: 10px;
            }}
            .tips-box h4 {{
                margin-bottom: 15px;
                color: #e65100;
            }}
            .tips-box ul {{
                margin-left: 20px;
                color: #f57c00;
                line-height: 1.8;
            }}
            .no-results {{
                background: white;
                border-radius: 15px;
                padding: 60px;
                text-align: center;
            }}
            .no-results-icon {{
                font-size: 5em;
                margin-bottom: 20px;
                opacity: 0.3;
                color: #1B3C53;
            }}
            .no-results h3 {{
                color: #1B3C53;
                margin-bottom: 10px;
            }}
            .no-results p {{
                color: #456882;
            }}
        </style>
    </head>
    <body>
        {results_html}

        <script>
            function openModal(index) {{
                document.getElementById('modal' + index).style.display = 'block';
                document.body.style.overflow = 'hidden';
            }}

            function closeModal(index) {{
                document.getElementById('modal' + index).style.display = 'none';
                document.body.style.overflow = 'auto';
            }}

            window.onclick = function(event) {{
                if (event.target.classList.contains('modal')) {{
                    event.target.style.display = 'none';
                    document.body.style.overflow = 'auto';
                }}
            }}
        </script>
    </body>
    </html>
    """
    
    return html

# === CSS untuk Streamlit ===
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #1B3C53 0%, #234C6A 50%, #456882 100%);
    }
    
    .main > div {
        padding-top: 2rem;
    }
    
    iframe {
        border: none;
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# === UI Streamlit ===

# Header
st.markdown("""
<div style="background: white; border-radius: 20px; padding: 40px; text-align: center; box-shadow: 0 10px 30px rgba(0,0,0,0.2); margin-bottom: 30px;">
    <h1 style="color: #1B3C53; font-size: 2.5em; margin-bottom: 10px; display: flex; align-items: center; justify-content: center; gap: 15px;">
        🛡️ Sistem Temu Kembali Deteksi Hoaks
    </h1>
    <p style="color: #456882; font-size: 1.1em;">
        Implementasi Temu Kembali Berita Indonesia dengan Integrasi Deteksi Hoaks Menggunakan TF-IDF dan Naive Bayes
    </p>
</div>
""", unsafe_allow_html=True)

# Stats
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div style="background: white; border-radius: 15px; padding: 30px; text-align: center; box-shadow: 0 5px 15px rgba(0,0,0,0.1); border-top: 4px solid #1B3C53;">
        <div style="font-size: 3em; margin-bottom: 15px;">💾</div>
        <div style="font-size: 2.5em; font-weight: bold; color: #1B3C53; margin-bottom: 5px;">{len(df)}</div>
        <div style="color: #456882; font-size: 1em;">Total Berita</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="background: white; border-radius: 15px; padding: 30px; text-align: center; box-shadow: 0 5px 15px rgba(0,0,0,0.1); border-top: 4px solid #1B3C53;">
        <div style="font-size: 3em; margin-bottom: 15px;">📊</div>
        <div style="font-size: 2.5em; font-weight: bold; color: #1B3C53; margin-bottom: 5px;">TF-IDF</div>
        <div style="color: #456882; font-size: 1em;">Metode Pencarian</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style="background: white; border-radius: 15px; padding: 30px; text-align: center; box-shadow: 0 5px 15px rgba(0,0,0,0.1); border-top: 4px solid #1B3C53;">
        <div style="font-size: 3em; margin-bottom: 15px;">📈</div>
        <div style="font-size: 2.5em; font-weight: bold; color: #1B3C53; margin-bottom: 5px;">Naive Bayes</div>
        <div style="color: #456882; font-size: 1em;">Algoritma ML</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# === FORM untuk menangkap Enter key ===
with st.form(key='search_form', clear_on_submit=False):
    query = st.text_input(
        "",
        placeholder="🔍 Masukkan kata kunci atau teks berita yang ingin diverifikasi...",
        key="search_query",
        label_visibility="collapsed"
    )
    
    # Button submit di dalam form
    search_button = st.form_submit_button("🔍 Analisis", use_container_width=True)

# Custom CSS untuk search box
st.markdown("""
<style>
    .stTextInput > div > div > input {
        border-radius: 50px;
        border: 2px solid #D2C1B6;
        padding: 18px 25px;
        font-size: 1.1em;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #1B3C53;
        box-shadow: 0 0 0 3px rgba(27, 60, 83, 0.1);
    }
    
    .stButton > button, .stFormSubmitButton > button {
        background: linear-gradient(135deg, #1B3C53 0%, #234C6A 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 50px !important;
        padding: 18px 45px !important;
        font-size: 1.1em !important;
        font-weight: 600 !important;
        height: 58px !important;
    }
    
    .stButton > button:hover, .stFormSubmitButton > button:hover {
        background: linear-gradient(135deg, #234C6A 0%, #456882 100%) !important;
        transform: scale(1.05);
    }
    
    /* Hide form border */
    .stForm {
        border: none !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Process search - TRIGGER BAIK DARI ENTER MAUPUN KLIK BUTTON
if search_button and query:
    with st.spinner("🔄 Menganalisis berita..."):
        results = retrieve_news(query, top_k=8)
    
    # Render HTML dengan modal
    html_content = generate_html(query, results)
    components.html(html_content, height=1200, scrolling=True)

else:
    st.markdown("""
    <div style="background: white; border-radius: 15px; padding: 60px; text-align: center; box-shadow: 0 5px 15px rgba(0,0,0,0.1);">
        <div style="font-size: 5em; margin-bottom: 20px; opacity: 0.3; color: #1B3C53;">🔍</div>
        <h3 style="color: #1B3C53; margin-bottom: 10px;">Mulai Pencarian</h3>
        <p style="color: #456882;">Masukkan kata kunci untuk mencari dan memverifikasi berita</p>
    </div>
    """, unsafe_allow_html=True)
