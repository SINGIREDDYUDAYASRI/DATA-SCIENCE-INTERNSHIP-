import pandas as pd
import re
import pickle
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load data
file_path = "reviews_data_dump/reviews_badminton/data.csv"
df = pd.read_csv(file_path)

def label_sentiment(rating):
    if rating >= 4:
        return 1
    elif rating <= 2:
        return 0
    else:
        return None

df['sentiment'] = df['Ratings'].apply(label_sentiment)
df = df.dropna(subset=['sentiment'])

df['full_review'] = df['Review Title'].fillna('') + " " + df['Review text'].fillna('')

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['clean_review'] = df['full_review'].apply(clean_text)

stop_words = set(stopwords.words('english'))

def remove_stopwords(text):
    return " ".join([w for w in text.split() if w not in stop_words])

df['final_review'] = df['clean_review'].apply(remove_stopwords)

# Train model
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['final_review'])
y = df['sentiment']

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# Save model & vectorizer
pickle.dump(model, open("sentiment_model.pkl", "wb"))
pickle.dump(tfidf, open("tfidf_vectorizer.pkl", "wb"))

print("Model and vectorizer saved successfully")











