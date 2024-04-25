import os
import PyPDF2
from langchain_core.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import torch
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

print("Loading PDF file...")
with open("C:/Users/hhhh/MLproj/MLproj/chatbot/updated CV.pdf", "rb") as f:
    pdf = PyPDF2.PdfReader(f)
    data = []
    for page in pdf.pages:
        data.append(page.extract_text())

print("Loaded data:", data)

if data:
    print("Data is not empty, proceeding to split text...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    content = "\n\n".join(data)
    texts = text_splitter.split_text(content)
    print("Split text into", len(texts), "chunks")
    print("Texts:", texts[:5])  # Print the first 5 text chunks

    if texts:
        print("Texts is not empty, proceeding to create vector store...")
        embeddings = HuggingFaceEmbeddings(model_name="distilbert-base-nli-mean-tokens")
        vector_store = Chroma.from_texts(texts, embeddings).as_retriever()
        print("Vector store created successfully")
        print("Embeddings:", vector_store.embeddings.shape)  # Print the shape of the embeddings

        prompt_template = """
          Please answer the question in as much detail as possible based on the provided context.
          Ensure to include all relevant details. If the answer is not available in the provided context,
          kindly respond with "The answer is not available in the context." Please avoid providing incorrect answers.
        \n\n
          Context:\n {context}?\n
          Question: \n{question}\n

          Answer:
        """

        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        model_name = "google/gemini-pro"
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        from langchain.chains.llm import LLMChain
        chain = LLMChain(llm=model, tokenizer=tokenizer, prompt=prompt)

        question = input("Enter your question: ")
        docs = vector_store.get_relevant_documents(question)

        response = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
        print("Response:", response)
    else:
        print("No text found in the PDF. Please try with a different document.")
else:
    print("No data loaded as the PDF. Please try with a different document.")