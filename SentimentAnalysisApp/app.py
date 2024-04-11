

from flask import Flask, render_template, request
import joblib
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

#app = Flask(__name__)
app = Flask(__name__, template_folder='templates')

# Load the trained sentiment analysis model
model = joblib.load("sentiment_analysis_model.pkl")

# Function to clean text
def clean_text(text):
    """
    This function cleans text data by removing irrelevant characters, converting to lowercase,
    tokenizing, and applying lemmatization.

    Args:
        text (str): The text to be cleaned.

    Returns:
        str: The cleaned text.
    """
    # Lowercase the text
    text = text.lower()

    # Remove special characters except for alphanumeric characters, spaces, and apostrophes
    text = re.sub(r"[^a-zA-Z0-9 \']", "", text)

    # Tokenize the text (split into words)
    tokens = word_tokenize(text)

    # Lemmatize each token (reduce words to their base form)
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Join the tokens back into a string
    cleaned_text = " ".join(tokens)

    return cleaned_text

# Function to get sentiment label
def get_sentiment_label(text):
    """
    This function uses Vader to analyze sentiment and assign a label.

    Args:
        text (str): The text to analyze.

    Returns:
        str: The sentiment label (positive, negative, neutral).
    """
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
    return render_template('C:\Users\hhhh\MLproj\MLproj\index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    cleaned_text = clean_text(text)
    sentiment_label = get_sentiment_label(cleaned_text)
    return render_template('C:\Users\hhhh\MLproj\MLproj\index.html', text=text, sentiment=sentiment_label)

def wsgi_app(self, environ, start_response):
    # Perform any additional setup or processing here
    return self(environ, start_response)

app.wsgi_app = wsgi_app


if __name__ == '__main__':
    app.run(debug=True)
    app.config['DEBUG'] = True


