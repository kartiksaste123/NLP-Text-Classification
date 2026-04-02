# ==========================================
# NLP Preprocessing & Text Classification
# (Using Real SMS Spam Dataset)
# ==========================================

import pandas as pd
import numpy as np
import nltk
import re

# NLP tools
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

# ML tools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Download required data
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

# -------------------------------
# LOAD DATASET
# -------------------------------
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"

df = pd.read_csv(url, sep='\t', header=None, names=['label', 'text'])

# Convert labels to numeric
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# -------------------------------
# PREPROCESSING FUNCTION
# -------------------------------
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()
    
    # Remove special characters
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
    
    tokens = word_tokenize(text)
    
    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]
    
    # Stemming
    tokens = [stemmer.stem(word) for word in tokens]
    
    # Lemmatization
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return " ".join(tokens)

# Apply preprocessing
df['processed'] = df['text'].apply(preprocess)

# -------------------------------
# FEATURE EXTRACTION
# -------------------------------
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(df['processed'])

y = df['label']

# -------------------------------
# TRAIN TEST SPLIT
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# -------------------------------
# MODEL 1: NAIVE BAYES
# -------------------------------
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

nb_pred = nb_model.predict(X_test)

# -------------------------------
# MODEL 2: LOGISTIC REGRESSION
# -------------------------------
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)

# -------------------------------
# EVALUATION
# -------------------------------
print("\n--- NAIVE BAYES ---")
print("Accuracy:", accuracy_score(y_test, nb_pred))
print(classification_report(y_test, nb_pred))

print("\n--- LOGISTIC REGRESSION ---")
print("Accuracy:", accuracy_score(y_test, lr_pred))
print(classification_report(y_test, lr_pred))

# -------------------------------
# TEST CUSTOM INPUT
# -------------------------------
sample = ["Congratulations! You won a free lottery ticket"]

sample_processed = [preprocess(sample[0])]
sample_vector = tfidf.transform(sample_processed)

prediction = nb_model.predict(sample_vector)

print("\nSample Text:", sample[0])
print("Prediction:", "Spam" if prediction[0] == 1 else "Not Spam")