# app_streamlit_logistic.py
import streamlit as st
import joblib
import re
import json
import requests
from streamlit_lottie import st_lottie

# Load model and vectorizer (change filenames if you saved differently)
model = joblib.load("logistic_regression_sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Stopwords (your list)
stop_words = set([
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "yourselves", "he", "him",
    "his", "himself", "she", "her", "hers", "herself", "it", "its",
    "itself", "they", "them", "their", "theirs", "themselves", "what",
    "which", "who", "whom", "this", "that", "these", "those", "am",
    "is", "are", "was", "were", "be", "been", "being", "have", "has",
    "had", "having", "do", "does", "did", "doing", "a", "an", "the",
    "and", "but", "if", "or", "because", "as", "until", "while",
    "of", "at", "by", "for", "with", "about", "against", "between",
    "into", "through", "during", "before", "after", "above", "below",
    "to", "from", "up", "down", "in", "out", "on", "off", "over",
    "under", "again", "further", "then", "once", "here", "there",
    "when", "where", "why", "how", "all", "any", "both", "each",
    "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "so", "than", "too", "very"
])

def clean_input_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

def load_lottie_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def load_lottie_file(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

# Load animation (local file)
try:
    lottie_sentiment = load_lottie_file("animation.json")
except Exception:
    lottie_sentiment = None

st.set_page_config(page_title="Medicine Review Sentiment", page_icon="💊", layout="centered")
st.title("💊 Medicine Review Sentiment Analysis")
st.subheader("Analyze the sentiment of any medicine review")

if lottie_sentiment:
    st_lottie(lottie_sentiment, height=200, key="sentiment")

user_input = st.text_area("📝 Enter your medicine review below:", height=150)

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter a review before analyzing.")
    else:
        cleaned = clean_input_text(user_input)
        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)[0]

        # Optionally show probability/confidence
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(vector)[0]
            # get probability for predicted class
            classes = model.classes_
            pred_idx = list(classes).index(prediction)
            confidence = proba[pred_idx]
        else:
            confidence = None

        if prediction == "positive":
            st.success(f"✅ Predicted Sentiment: Positive {f'({confidence:.2%})' if confidence is not None else ''}")
        elif prediction == "negative":
            st.error(f"❌ Predicted Sentiment: Negative {f'({confidence:.2%})' if confidence is not None else ''}")
        else:
            st.info(f"ℹ️ Predicted Sentiment: {prediction.capitalize()} {f'({confidence:.2%})' if confidence is not None else ''}")
