from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
import joblib
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

app = FastAPI()

# Load the trained sentiment analysis model
model = joblib.load("sentiment_analysis_model.pkl")

# Initialize templates
templates = Jinja2Templates(directory="templates")

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

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(request: Request, text: str = Form(...)):
    cleaned_text = clean_text(text)
    sentiment_label = get_sentiment_label(cleaned_text)
    return templates.TemplateResponse("index.html", {"request": request, "caption": text, "sentiment_label": sentiment_label})
