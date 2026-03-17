import streamlit as st
import joblib
import re
import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

# Load model and vectorizer
model = joblib.load('best_model.pkl')
tfidf = joblib.load('tfidf.pkl')

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r'<.*?>',   ' ', str(text))
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = text.lower().strip()
    tokens = [w for w in text.split() if w not in stop_words and len(w) > 2]
    return ' '.join(tokens)

st.set_page_config(page_title="Fake Job Detector", page_icon="🔍")
st.title("🔍 Fake Job Posting Detector")
st.write("Paste a job posting below and find out if it's real or fake.")

title       = st.text_input("Job Title")
description = st.text_area("Job Description", height=200)
requirements = st.text_area("Requirements", height=100)
company_profile = st.text_area("Company Profile", height=100)

if st.button("Detect"):
    combined = title + ' ' + description + ' ' + requirements + ' ' + company_profile
    cleaned  = clean_text(combined)
    vec      = tfidf.transform([cleaned])
    pred     = model.predict(vec)[0]
    proba    = model.predict_proba(vec)[0]

    if pred == 1:
        st.error(f"⚠️ This looks like a FAKE job posting! (Confidence: {proba[1]*100:.1f}%)")
    else:
        st.success(f"✅ This looks like a REAL job posting. (Confidence: {proba[0]*100:.1f}%)")
