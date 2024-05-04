from flask import Flask, request, render_template, jsonify, send_file
from langchain.agents.agent import AgentExecutor
from langchain.evaluation import load_dataset
from langchain.retrievers.document_compressors.chain_extract import LLMChainExtractor
import os
import uuid
from langchain.chains.qa_with_sources.base import BaseQAWithSourcesChain
from langchain.chains.qa_generation.base import QAGenerationChain
from langchain.chains.qa_with_sources.base import QAWithSourcesChain
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from google.cloud import language
from langchain_community.embeddings import HuggingFaceEmbeddings
from google.oauth2 import service_account
import csv
from transformers import pipeline
import google.generativeai as genai
from transformers import pipeline, T5ForConditionalGeneration, T5Tokenizer
import torch
import faiss
from rasa.nlu.extractors import EntityExtractor
from rasa.nlu.components import IntentClassifier
from rasa.nlu.training_data import TrainingData
from rasa.nlu.model import Metadata
from pymongo import MongoClient

app = Flask(__name__)

# Create a directory to store uploaded documents
UPLOAD_DIR = "uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# Load credentials from service account JSON key file
creds = service_account.Credentials.from_service_account_file(
    'C:/Users/hhhh/OneDrive/Desktop/Keys/chatbot-421205-83ea76959fa5.json',
    scopes=['https://www.googleapis.com/auth/cloud-language']
)

# Create a FAISS index with a flat index type and 128 dimensions
index = faiss.IndexFlatL2(128)  # Create a FAISS index with a flat index type and 128 dimensions
faiss_index = index
docstore = []  # Initialize an empty document store
index_to_docstore_id = {}  # Initialize an empty mapping from index IDs to document store IDs

faiss_index = FAISS(index, docstore, index_to_docstore_id)

# Define a function to encode documents using Hugging Face embeddings
def encode_document(document_text):
    embeddings = HuggingFaceEmbeddings("bert-base-uncased").encode(document_text)
    return embeddings

# Define a function to generate answers using a T5 model
def generate_answer(input_text):
    model = T5ForConditionalGeneration.from_pretrained("t5-base")
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    inputs = tokenizer.encode_plus(
        input_text,
        add_special_tokens=True,
        max_length=512,
        return_attention_mask=True,
        return_tensors="pt",
        truncation=True,
    )
    decoder_input_ids = torch.tensor([[tokenizer.pad_token_id] * 10])
    output = model(inputs["input_ids"], attention_mask=inputs["attention_mask"], decoder_input_ids=decoder_input_ids)
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    return answer

# Define a function to extract entities from user input
def extract_entities(text):
    extractor = EntityExtractor()
    entities = extractor.extract(text)
    return entities

# Define a function to classify user intent
def classify_intent(text):
    intent_classifier = IntentClassifier()
    intent = intent_classifier.classify(text)
    return intent

# Connect to MongoDB database
client = MongoClient('mongodb://localhost:27017/')
db = client['chatbot']
collection = db['conversations']

@app.route("/", methods=["GET"])
def index():
    return render_template("ui.html")

@app.route("/upload", methods=["POST"])
def upload_document():
    try:
        document = request.files["document"]
        if document:
            # Validate file type and size
            allowed_extensions = [".txt", ".pdf", ".docx"]
            if not any(document.filename.lower().endswith(ext) for ext in allowed_extensions):
                return jsonify({"error": "Invalid file type. Only .txt, .pdf, and .docx files are allowed."}), 400

            max_file_size = 10 * 1024 * 1024  # 10MB
            if document.content_length > max_file_size:
                return jsonify({"error": "File size exceeds the maximum limit of 10MB."}), 400

            # Save the uploaded document
            document_path = os.path.join(UPLOAD_DIR, document.filename)
            document.save(document_path)

            # Add the document to the FAISS index
            with open(document_path, "r", encoding="utf-8", errors="replace") as f:
                document_text = f.read()
            embeddings = encode_document(document_text)
            faiss_index.add(embeddings, document_path)

            return jsonify({"message": "Document uploaded successfully", "document_filename": document.filename})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.json
        user_query = data["query"]
        document_filename = data.get("document_filename", "")

        if not document_filename:  # No document uploaded, use Gemini Pro model for response generation
            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content(user_query)
            return jsonify({"answer": response.text})

        # Load the uploaded document
        document_path = os.path.join(UPLOAD_DIR, document_filename)
        with open(document_path, "r", encoding="utf-8", errors="replace") as f:
            document_text = f.read()

        # Expand the user query using keyword extraction
        query_expansion_model = pipeline("keyword-extraction")
        expanded_query = query_expansion_model(user_query)

        # Retrieve relevant documents from the FAISS index
        embeddings = encode_document(expanded_query)
        scores, indices = faiss_index.search(embeddings, k=5)

        # Get the top-scoring document
        top_document_path = faiss_index.get(indices[0])

        # Load the top-scoring document
        with open(top_document_path, "r", encoding="utf-8", errors="replace") as f:
            top_document_text = f.read()

        # Generate an answer using the top-scoring document
        input_text = f"Query: {user_query}\nDocument: {top_document_text}"
        answer = generate_answer(input_text)

        # Return the answer
        return jsonify({"answer": answer})

    except Exception as e:
        app.logger.error(f"Error in /ask endpoint: {str(e)}")
        return jsonify({"error": "An error occurred while processing the request."}), 500

@app.route("/callme", methods=["POST"])
def callme():
    try:
        data = request.json
        user_query = data["query"]
        name = data.get("name", "")
        phone_number = data.get("phone_number", "")
        email = data.get("email", "")

        if not (name and phone_number and email):
            return jsonify({"error": "Please provide all required information."}), 400

        # Extract entities from user input
        entities = extract_entities(user_query)
        intent = classify_intent(user_query)

        # Store user information in the database
        conversation = {
            "user_query": user_query,
            "name": name,
            "phone_number": phone_number,
            "email": email,
            "entities": entities,
            "intent": intent
        }
        collection.insert_one(conversation)

        return jsonify({"message": "Thank you for providing your information. We will call you soon."})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)