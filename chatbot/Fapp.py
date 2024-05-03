from flask import Flask, request, render_template, jsonify
from langchain import LangChain
from langchain.modules import TextDocumentLoader, QuestionAnswerer
import os
import uuid

app = Flask(__name__)

# Create a directory to store uploaded documents
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.route("/upload", methods=["POST"])
def upload_document():
    file = request.files["file"]
    filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
    filepath = os.path.join(UPLOAD_DIR, filename)
    file.save(filepath)
    return jsonify({"filename": filename})

@app.route("/ask", methods=["POST"])
def ask():
    filename = request.json["filename"]
    question = request.json["question"]
    filepath = os.path.join(UPLOAD_DIR, filename)
    document_loader = TextDocumentLoader(filepath)
    document = document_loader.load()
    lc = LangChain()
    qa_module = QuestionAnswerer(lc, document)
    answer = qa_module.answer(question)
    return render_template("chatUI.html", query=question, response=answer)

if __name__ == "__main__":
    app.run(debug=True)