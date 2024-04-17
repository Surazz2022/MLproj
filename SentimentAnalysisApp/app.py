from flask import Flask, render_template, request
import joblib
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

app = Flask(__name__, template_folder='templates')

# Loading the trained sentiment analysis model
model = joblib.load("sentiment_analysis_model.pkl")

# Function to clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9 \']", "", text)
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    cleaned_text = " ".join(tokens)
    return cleaned_text

# Function to get sentiment label
def get_sentiment_label(text):
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text)
    if scores['compound'] >= 0.05:
        return 'positive'
    elif scores['compound'] <= -0.05:
        return 'negative'
    else:
        return 'neutral'

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    cleaned_text = clean_text(text)
    sentiment_label = get_sentiment_label(cleaned_text)
    return render_template('index.html', caption=text, sentiment_label=sentiment_label)

if __name__ == '__main__':
    app.run(debug=True)
