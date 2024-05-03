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

@app.route("/", methods=["GET"])
def index():
    return render_template("chatUI.html")

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

            return jsonify({"message": "Document uploaded successfully", "document_filename": document.filename})
        else:
            return jsonify({"error": "No document uploaded."}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.json
        user_query = data["query"]
        document_filename = data.get("document_filename", "")

        if not document_filename:
            return jsonify({"error": "No document uploaded."}), 400

        uuid_str = str(uuid.uuid4())

        # Load the uploaded document
        document_path = os.path.join(UPLOAD_DIR, document_filename)
        with open(document_path, "r", encoding="utf-8", errors="replace") as f:
            document_text = f.read()

        # Optimize chunking algorithm to produce chunks of 1000 characters
        chunk_size = 1000
        chunks = [document_text[i:i + chunk_size] for i in range(0, len(document_text), chunk_size)]

        # Create a vector store from the chunks
        vector_store = FAISS(
            embedding_function=HuggingFaceEmbeddings(model_name="bert-base-uncased"),  # or some other embedding function
            index="flat",  # or some other index type
            docstore={"docs": []},  # or some other docstore implementation
            index_to_docstore_id=lambda x: x  # or some other mapping function
        )

        # Create a language client with the credentials
        language_client = language.LanguageServiceClient(credentials=creds)

        for chunk in chunks:
            response = language_client.analyze_entities(
                document={"content": chunk, "type_": "PLAIN_TEXT"}
            )
            entities = [entity.mentions[0].text.content for entity in response.entities]
            vector_store.add_texts(entities)

        # Create an LLMChainExtractor to compress and retrieve documents
        extractor = LLMChainExtractor()

        # Create a BaseQAWithSourcesChain to handle user queries
        qa_chain = BaseQAWithSourcesChain(extractor)

        # Create a QAGenerationChain to generate answers
        generation_chain = QAGenerationChain()

        # Create an AgentExecutor to execute the QA chain
        executor = AgentExecutor(qa_chain, generation_chain)
        result = executor.run(user_query, vector_store, uuid_str)

        # Return the answer as JSON
        return jsonify({"answer": result})
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

        # Store user information in a database or a file (e.g., a CSV file)
        with open("user_info.csv", "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([name, phone_number, email])

        return jsonify({"message": "Thank you for providing your information. We will call you soon."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)