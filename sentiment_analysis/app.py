import streamlit as st
import pickle
import re
from nltk.corpus import stopwords

# Safe stopwords loading (works on Streamlit Cloud)
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    import nltk
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

# Load model & vectorizer
model = pickle.load(open("sentiment_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

# Text preprocessing
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def remove_stopwords(text):
    return " ".join([word for word in text.split() if word not in stop_words])

def preprocess(text):
    return remove_stopwords(clean_text(text))

# UI
st.set_page_config(page_title="Flipkart Review Sentiment Analyzer")
st.title("🛒 Flipkart Review Sentiment Analyzer")
st.write("Enter a product review to predict its sentiment")

review = st.text_area("✍️ Enter your review")

if st.button("Predict Sentiment"):
    if review.strip() == "":
        st.warning("Please enter some text")
    else:
        processed = preprocess(review)
        vector = vectorizer.transform([processed])
        prediction = model.predict(vector)[0]

        if prediction == 1:
            st.success("✅ Positive Review 😊")
        else:
            st.error("❌ Negative Review 😞")


