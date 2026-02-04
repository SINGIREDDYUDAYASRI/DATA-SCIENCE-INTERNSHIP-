import streamlit as st
import pickle
import re
import os
from nltk.corpus import stopwords

# -------------------------------
# Paths (CRITICAL FIX)
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "sentiment_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "tfidf_vectorizer.pkl")

# -------------------------------
# Load model & vectorizer (ONCE)
# -------------------------------
@st.cache_resource
def load_artifacts():
    model = pickle.load(open(MODEL_PATH, "rb"))
    vectorizer = pickle.load(open(VECTORIZER_PATH, "rb"))
    return model, vectorizer

model, vectorizer = load_artifacts()

# -------------------------------
# Stopwords (NO DOWNLOAD)
# -------------------------------
try:
    stop_words = set(stopwords.words("english"))
except LookupError:
    stop_words = set()  # fallback, app still works

# -------------------------------
# Text preprocessing
# -------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def remove_stopwords(text):
    return " ".join(word for word in text.split() if word not in stop_words)

def preprocess(text):
    return remove_stopwords(clean_text(text))

# -------------------------------
# UI
# -------------------------------
st.set_page_config(page_title="Flipkart Review Sentiment Analyzer")
st.title("🛒 Flipkart Review Sentiment Analyzer")
st.write("Enter a product review to predict its sentiment")

review = st.text_area("Enter your review")

if st.button("Predict Sentiment"):
    if not review.strip():
        st.warning("Please enter some text")
    else:
        processed = preprocess(review)
        vector = vectorizer.transform([processed])
        prediction = model.predict(vector)[0]

        if prediction == 1:
            st.success("✅ Positive Review 😊")
        else:
            st.error("❌ Negative Review 😞")


