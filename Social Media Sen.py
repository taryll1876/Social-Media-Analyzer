import re
import nltk
import tkinter as tk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import pandas as pd
from alpha_vantage.timeseries import TimeSeries

#  Data Collection
# Collect your dataset of social media posts with sentiment labels (positive, negative, neutral)

# Data Preprocessing
def preprocess_text(text):
    # Remove special characters, URLs, hashtags, and mentions
    text = re.sub(r"http\S+|www\S+|https\S+|\#\w+|\@\w+", "", text)
    # Convert to lowercase
    text = text.lower()
    # Remove stopwords and tokenize
    stop_words = set(stopwords.words("english"))
    tokens = word_tokenize(text)
    filtered_tokens = [token for token in tokens if token not in stop_words]
    # Join tokens back into a string
    processed_text = " ".join(filtered_tokens)
    return processed_text

# Sentiment Analysis Model
def train_sentiment_model(X, y):
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Convert text data into numerical features using TF-IDF
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)
    # Train a support vector machine (SVM) classifier
    svm_classifier = SVC(kernel="linear")
    svm_classifier.fit(X_train, y_train)
    # Evaluate the model
    accuracy = svm_classifier.score(X_test, y_test)
    print("Model accuracy:", accuracy)
    return svm_classifier, vectorizer

#User Interface
def predict_sentiment(text, model, vectorizer):
    processed_text = preprocess_text(text)
    features = vectorizer.transform([processed_text])
    sentiment = model.predict(features)
    return sentiment[0]

def analyze_sentiment():
    text = entry_text.get("1.0", "end").strip()
    if text:
        sentiment = predict_sentiment(text, model, vectorizer)
        sentiment_label.config(text=f"Sentiment: {sentiment}")
    else:
        sentiment_label.config(text="Please enter some text.")

def visualize_sentiment_distribution():
    sentiment_counts = df["Sentiment"].value_counts()
    sentiment_counts.plot(kind="bar")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.title("Sentiment Distribution")
    plt.show()

def update_stock_price():
    ts = TimeSeries(key='YOUR_API_KEY')  # Replace with your Alpha Vantage API key
    data, _ = ts.get_intraday(symbol='MSFT', interval='1min', outputsize='compact')
    df = pd.DataFrame(data).transpose()
    df.columns = ["Open", "High", "Low", "Close", "Volume"]
    stock_price.config(text=f"Stock Price: {df['Close'].iloc[-1]}")

#Usage
# Load and preprocess your dataset
df = pd.read_csv("dataset.csv")  # Replace "dataset.csv" with the path to your dataset file

# X - input text data, y - corresponding sentiment labels (positive, negative, neutral)
Text,Sentiment
"I love this product!",positive
"This movie was terrible.",negative
"The weather is nice today.",neutral

# For demonstration purposes, let's assume we already have the preprocessed dataset
X = np.array(["I love this product!", "This movie was terrible.", "The weather is nice today."])
y = np.array(["positive", "negative", "neutral"])

# Train the sentiment analysis model
model, vectorizer = train_sent
