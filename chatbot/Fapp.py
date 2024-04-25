import pathlib
import textwrap

import google.generativeai as genai
from flask import Flask, request, jsonify, render_template
from nltk.corpus import stopwords  # Import stopwords for preprocessing
from nltk.tokenize import word_tokenize  # Import word_tokenize for tokenization

from IPython.display import display
from IPython.display import Markdown


def to_markdown(text):
    text = text.replace('•', '  *')
    return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))


# Define userdata dictionary with Google API key
userdata = {
    'google_api_key': 'AIzaSyCVn0U15A6VkcJRP725hM342gLiB00861o'  # Replace with your actual key
}

# Configure GenerativeAI
GOOGLE_API_KEY = userdata.get('google_api_key')
genai.configure(api_key=GOOGLE_API_KEY)


model = genai.GenerativeModel('gemini-pro')

# Function to read a document from a file (replace with your document parsing logic if needed)
def read_document(file_path):
    with open(file_path, 'r') as f:
        document_content = f.read()
    return document_content


# Function to retrieve documents based on type and location (replace with your implementation)
def get_documents(document_type, location):
    # Handle different document types (e.g., database query, filesystem access)
    if document_type == 'file':
        if pathlib.Path(location).is_file():
            return [read_document(location)]  # Read a single document from the provided path
        else:
            raise ValueError(f"Invalid file path: {location}")
    elif document_type == 'directory':
        documents = []
        for file in pathlib.Path(location).iterdir():
            if file.is_file():
                documents.append(read_document(str(file)))  # Convert Path object to string
        return documents
    else:
        raise ValueError(f"Unsupported document type: {document_type}")


# Function to preprocess documents (replace with your actual preprocessing steps)
def preprocess_documents(documents):
    # Perform necessary cleaning and tokenization (using NLTK)
    stop_words = stopwords.words('english')  # Download stopwords corpus if needed
    preprocessed_documents = []
    for doc in documents:
        tokens = word_tokenize(doc.lower())  # Lowercase and tokenize
        filtered_tokens = [token for token in tokens if token not in stop_words]  # Remove stop words
        # Add further steps like stemming/lemmatization if needed
        preprocessed_documents.append(filtered_tokens)
    return preprocessed_documents


# --- Document Embedding Section (Modified using TF-IDF) ---

# Removed Gensim-based document embedding section (uncomment if you prefer TF-IDF)

# Function to generate TF-IDF vectors for documents (alternative to word embeddings)
def generate_tf_idf_vectors(documents):
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer()
    tf_idf_matrix = vectorizer.fit_transform(documents)
    return tf_idf_matrix

# Function to retrieve documents based on cosine similarity with TF-IDF
def retrieve_documents_by_similarity(query, tf_idf_matrix, k=3):
    from sklearn.metrics.pairwise import cosine_similarity

    query_vector = vectorizer.transform([query])
    similarity_scores = cosine_similarity(query_vector, tf_idf_matrix).flatten()
    top_documents = sorted(enumerate(similarity_scores), key=lambda item: item[1], reverse=True)[:k]
    return top_documents

# --- End of Document Embedding Section ---


# Function to preprocess user query (customizable)
def preprocess_query(query):
    text = query.lower()  # Lowercase
    text = text.strip()  # Remove leading/trailing whitespaces
    # Add further steps like stop word removal, stemming/lemmatization
    return text.split()


def simulate_gemini_call(documents):
    # Process documents using Gemini API (replace with your actual API call)
    answer = "Answer retrieved from documents using Gemini"  # Placeholder for Gemini's response
    return answer





# Flask application
app = Flask(__name__)

# Route to handle incoming messages from the user
@app.route('/message', methods=['POST'])
def handle_message():
    data = request.get_json()
    user_input = data['message']
    # Generate response using the GenerativeModel
    response = model.generate_content(user_input)
    return jsonify({'message': response.text})

# Route to render the chatbot interface
@app.route('/')
def chat_interface():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
    