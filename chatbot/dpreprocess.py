import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import pdfplumber

# Load the documents from a CSV file (assuming one document per row)
#docs_df = pd.read_csv('documents.csv')

# Load the PDF file
with pdfplumber.open('chatbot/updated CV.pdf') as pdf:
    # Extract the text from the PDF
    text = ''
    for page in pdf.pages:
        text += page.extract_text()

# Preprocess the extracted text
docs = pd.DataFrame({'text': [text]})


# Load a pre-trained tokenizer and model for question answering
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModelForQuestionAnswering.from_pretrained('bert-base-uncased')

# Preprocess the documents
preprocessed_docs = []
for doc in docs['text']:
    # Tokenize the document
    inputs = tokenizer.encode_plus(
        doc,
        add_special_tokens=True,
        max_length=512,
        return_attention_mask=True,
        return_tensors='pt'
    )

    # Convert the tokenized input to a format suitable for the model
    input_ids = inputs['input_ids'].flatten()
    attention_mask = inputs['attention_mask'].flatten()

    # Create a dictionary to store the preprocessed document
    preprocessed_doc = {
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }

    preprocessed_docs.append(preprocessed_doc)

# Convert the preprocessed documents to a PyTorch dataset
class DocumentDataset(torch.utils.data.Dataset):
    def __init__(self, preprocessed_docs):
        self.preprocessed_docs = preprocessed_docs

    def __len__(self):
        return len(self.preprocessed_docs)

    def __getitem__(self, idx):
        doc = self.preprocessed_docs[idx]
        return {
            'input_ids': doc['input_ids'],
            'attention_mask': doc['attention_mask']
        }

dataset = DocumentDataset(preprocessed_docs)

# Now you can use the dataset to train a question answering model