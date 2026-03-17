import streamlit as st
import pickle
import re
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="JobGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

.main { background: #0a0e1a; color: #e2e8f0; }
.stApp { background: linear-gradient(135deg, #0a0e1a 0%, #0d1526 50%, #0a1628 100%); }

.hero-banner {
    background: linear-gradient(135deg, #1a1f3a 0%, #0f2040 50%, #1a2a4a 100%);
    border: 1px solid #2a4080;
    border-radius: 16px;
    padding: 2.5rem;
    margin-bottom: 2rem;
    text-align: center;
    box-shadow: 0 8px 32px rgba(0,100,255,0.15);
}

.hero-title {
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(90deg, #60a5fa, #a78bfa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
}

.hero-sub {
    color: #94a3b8;
    font-size: 1.1rem;
    font-weight: 300;
}

.result-real {
    background: linear-gradient(135deg, #052e16, #064e3b);
    border: 2px solid #10b981;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    animation: pulse-green 2s infinite;
}

.result-fake {
    background: linear-gradient(135deg, #2d0a0a, #450a0a);
    border: 2px solid #ef4444;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    animation: pulse-red 2s infinite;
}

@keyframes pulse-green {
    0%, 100% { box-shadow: 0 0 0 0 rgba(16,185,129,0.4); }
    50%       { box-shadow: 0 0 20px 6px rgba(16,185,129,0.2); }
}
@keyframes pulse-red {
    0%, 100% { box-shadow: 0 0 0 0 rgba(239,68,68,0.4); }
    50%       { box-shadow: 0 0 20px 6px rgba(239,68,68,0.2); }
}

.result-label {
    font-size: 2rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
}

.metric-card {
    background: #111827;
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 1.2rem;
    text-align: center;
}

.metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.8rem;
    font-weight: 600;
    color: #60a5fa;
}

.metric-label {
    color: #64748b;
    font-size: 0.85rem;
    margin-top: 4px;
}

.section-card {
    background: #111827;
    border: 1px solid #1e2d4a;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}

.warning-box {
    background: #1c1400;
    border-left: 4px solid #f59e0b;
    border-radius: 0 8px 8px 0;
    padding: 1rem 1.2rem;
    margin-top: 1rem;
    color: #fcd34d;
    font-size: 0.9rem;
}

.tip-box {
    background: #0c1f3a;
    border-left: 4px solid #3b82f6;
    border-radius: 0 8px 8px 0;
    padding: 1rem 1.2rem;
    margin-top: 1rem;
    color: #93c5fd;
    font-size: 0.9rem;
}

.stTextInput input, .stTextArea textarea {
    background: #111827 !important;
    color: #e2e8f0 !important;
    border: 1px solid #1e3a5f !important;
    border-radius: 8px !important;
}

.stButton > button {
    background: linear-gradient(135deg, #2563eb, #7c3aed) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    padding: 0.7rem 2rem !important;
    width: 100% !important;
    transition: all 0.2s ease !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 20px rgba(99,102,241,0.4) !important;
}

.sidebar-info {
    background: #0f172a;
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 1rem;
    margin-bottom: 1rem;
    font-size: 0.9rem;
    color: #94a3b8;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Load Model & Vectorizer
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    except FileNotFoundError:
        return None, None

model, vectorizer = load_model()


# ─────────────────────────────────────────────
# Text Preprocessing  (must match training)
# ─────────────────────────────────────────────
def clean_text(text):
    if not text:
        return ""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def build_features(title, company, location, description,
                   requirements, benefits, employment_type,
                   required_experience, required_education,
                   industry, function):
    """Combine all fields — same order as training pipeline."""
    combined = " ".join([
        title, company, location, description, requirements,
        benefits, employment_type, required_experience,
        required_education, industry, function
    ])
    return clean_text(combined)


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ JobGuard AI")
    st.markdown("---")

    st.markdown("""
    <div class="sidebar-info">
    <b>🎓 Final Year Project</b><br>
    ML-powered system to detect fraudulent job postings
    using NLP and supervised learning.
    </div>
    """, unsafe_allow_html=True)

    if model is not None:
        st.success("✅ Model loaded successfully")
        st.markdown(f"""
        <div class="sidebar-info">
        <b>Model Info</b><br>
        Type: {type(model).__name__}<br>
        Features: {model.n_features_in_}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error("❌ Model not found! Run training notebook first.")

    st.markdown("---")
    st.markdown("""
    <div class="sidebar-info">
    <b>⚠️ Red Flags in Job Postings</b><br>
    • Asks for payment / fees<br>
    • Promises unrealistic salary<br>
    • No company information<br>
    • Requests bank details early<br>
    • Too urgent / vague description
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Hero Banner
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">🛡️ JobGuard AI</div>
    <div class="hero-sub">Intelligent Fake vs Real Job Posting Detection System</div>
    <div style="margin-top:1rem; color:#475569; font-size:0.85rem;">
        Powered by Machine Learning · TF-IDF · NLP Pipeline
    </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 Analyze Job Posting", "📊 Model Performance", "ℹ️ About"])


# ══════════════════════════════════════════════
# TAB 1 — Analyze
# ══════════════════════════════════════════════
with tab1:
    st.markdown("### Enter Job Posting Details")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("**📋 Basic Information**")
        title = st.text_input("Job Title *", placeholder="e.g., Software Engineer")
        company = st.text_input("Company Name", placeholder="e.g., Acme Corp")
        location = st.text_input("Location", placeholder="e.g., Hyderabad, India")
        employment_type = st.selectbox("Employment Type",
            ["", "Full-time", "Part-time", "Contract", "Temporary", "Internship", "Other"])
        required_experience = st.selectbox("Required Experience",
            ["", "Not Applicable", "Internship", "Entry level", "Associate",
             "Mid-Senior level", "Director", "Executive"])
        required_education = st.selectbox("Required Education",
            ["", "Unspecified", "High School or equivalent",
             "Some College Coursework Completed", "Associate Degree",
             "Bachelor's Degree", "Master's Degree", "Doctorate", "Professional"])
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("**📝 Detailed Information**")
        industry = st.text_input("Industry", placeholder="e.g., Information Technology")
        function = st.text_input("Job Function", placeholder="e.g., Engineering")
        description = st.text_area("Job Description *",
            placeholder="Paste the full job description here...", height=120)
        requirements = st.text_area("Requirements",
            placeholder="List skills, qualifications...", height=80)
        benefits = st.text_area("Benefits",
            placeholder="Health insurance, 401k...", height=60)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    analyze_btn = st.button("🔍 Analyze Job Posting", use_container_width=True)

    if analyze_btn:
        if not title and not description:
            st.warning("⚠️ Please enter at least a Job Title or Description.")
        elif model is None:
            st.error("❌ Model not loaded. Please run the training notebook first!")
        else:
            with st.spinner("🤖 Analyzing posting..."):

                # Build features — TF-IDF only (matches training)
                text = build_features(
                    title, company, location, description,
                    requirements, benefits, employment_type,
                    required_experience, required_education,
                    industry, function
                )
                X = vectorizer.transform([text])

                prediction = model.predict(X)[0]
                proba = model.predict_proba(X)[0] if hasattr(model, "predict_proba") else None

                st.markdown("---")
                st.markdown("## 🎯 Analysis Result")

                col_r1, col_r2, col_r3 = st.columns([1, 2, 1])
                with col_r2:
                    if prediction == 0:
                        # FAKE
                        confidence = proba[0] * 100 if proba is not None else None
                        conf_html = (f'<div style="color:#f87171;font-size:1.4rem;'
                                     f'margin-top:0.5rem;font-weight:700;">'
                                     f'Confidence: {confidence:.1f}%</div>'
                                     if confidence else '')
                        st.markdown(f"""
                        <div class="result-fake">
                            <div class="result-label">⚠️ FAKE JOB POSTING</div>
                            <div style="color:#fca5a5;font-size:1.1rem;">
                                This posting shows signs of fraud
                            </div>
                            {conf_html}
                        </div>
                        """, unsafe_allow_html=True)

                        st.markdown("""
                        <div class="warning-box">
                            ⚠️ <b>Warning:</b> Do NOT share personal documents,
                            pay any fees, or provide banking details.
                            Report this posting to the job platform immediately.
                        </div>
                        """, unsafe_allow_html=True)

                    else:
                        # REAL
                        confidence = proba[1] * 100 if proba is not None else None
                        conf_html = (f'<div style="color:#34d399;font-size:1.4rem;'
                                     f'margin-top:0.5rem;font-weight:700;">'
                                     f'Confidence: {confidence:.1f}%</div>'
                                     if confidence else '')
                        st.markdown(f"""
                        <div class="result-real">
                            <div class="result-label">✅ LEGITIMATE JOB POSTING</div>
                            <div style="color:#6ee7b7;font-size:1.1rem;">
                                This posting appears to be genuine
                            </div>
                            {conf_html}
                        </div>
                        """, unsafe_allow_html=True)

                        st.markdown("""
                        <div class="tip-box">
                            💡 <b>Tip:</b> Always research the company independently,
                            verify the recruiter's identity, and never share sensitive
                            information early in the hiring process.
                        </div>
                        """, unsafe_allow_html=True)

                # Probability Breakdown
                if proba is not None:
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("### 📊 Probability Breakdown")
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value" style="color:#ef4444">
                                {proba[0]*100:.1f}%
                            </div>
                            <div class="metric-label">Probability: FAKE</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with c2:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value" style="color:#10b981">
                                {proba[1]*100:.1f}%
                            </div>
                            <div class="metric-label">Probability: REAL</div>
                        </div>
                        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 2 — Model Performance
# ══════════════════════════════════════════════
with tab2:
    st.markdown("### 📊 Model Evaluation Metrics")

    try:
        with open("metrics.pkl", "rb") as f:
            metrics = pickle.load(f)

        col1, col2, col3, col4 = st.columns(4)
        for col, (label, key, color) in zip(
            [col1, col2, col3, col4],
            [("Accuracy",  "accuracy",  "#60a5fa"),
             ("Precision", "precision", "#a78bfa"),
             ("Recall",    "recall",    "#34d399"),
             ("F1 Score",  "f1",        "#f59e0b")]
        ):
            with col:
                val = metrics.get(key, 0) * 100
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="color:{color}">{val:.1f}%</div>
                    <div class="metric-label">{label}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        if "best_model_name" in metrics:
            st.info(f"🏆 Best Model: **{metrics['best_model_name']}**")

        if "report" in metrics:
            st.markdown("### 📋 Classification Report")
            st.code(metrics["report"], language="text")

        if "confusion_matrix" in metrics:
            st.markdown("### 🔢 Confusion Matrix")
            cm = metrics["confusion_matrix"]
            cm_df = pd.DataFrame(
                cm,
                index=["Actual Fake", "Actual Real"],
                columns=["Predicted Fake", "Predicted Real"]
            )
            st.dataframe(cm_df, use_container_width=True)

    except FileNotFoundError:
        st.warning("⚠️ Metrics file not found. Please run the training notebook first.")
        st.markdown("#### Expected Performance (After Training)")
        col1, col2, col3, col4 = st.columns(4)
        for col, (label, val, color) in zip(
            [col1, col2, col3, col4],
            [("Accuracy", "~97%", "#60a5fa"), ("Precision", "~96%", "#a78bfa"),
             ("Recall",   "~98%", "#34d399"), ("F1 Score",  "~97%", "#f59e0b")]
        ):
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="color:{color}">{val}</div>
                    <div class="metric-label">{label}</div>
                </div>
                """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 3 — About
# ══════════════════════════════════════════════
with tab3:
    st.markdown("### ℹ️ About This Project")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="section-card">
        <h4>🎯 Project Overview</h4>
        <p style="color:#94a3b8">
        JobGuard AI is a machine learning system that detects fraudulent job postings
        using NLP and classification algorithms. Built as a Final Year Project to help
        job seekers avoid scams.
        </p>
        <h4>📦 Dataset</h4>
        <p style="color:#94a3b8">
        Trained on the <b>EMSCAD (Employment Scam Aegean Corpus)</b> dataset containing
        ~17,880 real and fake job postings published between 2012–2014.
        </p>
        <h4>🔍 How It Works</h4>
        <p style="color:#94a3b8">
        1. Job posting text is cleaned and preprocessed<br>
        2. TF-IDF converts text to numerical features<br>
        3. ML model predicts: Fake (0) or Real (1)<br>
        4. Confidence score shown as probability %
        </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="section-card">
        <h4>🛠️ Tech Stack</h4>
        <ul style="color:#94a3b8">
            <li>Python 3.10+</li>
            <li>Scikit-learn (ML models)</li>
            <li>NLTK (text preprocessing)</li>
            <li>TF-IDF Vectorization</li>
            <li>Random Forest / Logistic Regression</li>
            <li>SMOTE (class balancing)</li>
            <li>Streamlit (Frontend UI)</li>
            <li>Google Colab (Training)</li>
            <li>Ngrok (Public deployment)</li>
        </ul>
        <h4>📊 ML Pipeline</h4>
        <ul style="color:#94a3b8">
            <li>Text cleaning & lemmatization</li>
            <li>Stop word removal</li>
            <li>TF-IDF feature extraction</li>
            <li>SMOTE oversampling</li>
            <li>Model training & evaluation</li>
            <li>Real-time prediction</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center; color:#475569; margin-top:2rem; font-size:0.85rem;">
        🎓 Final Year Project · Machine Learning · NLP · Streamlit · JobGuard AI
    </div>
    """, unsafe_allow_html=True)
