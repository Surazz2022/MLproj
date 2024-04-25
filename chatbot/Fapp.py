import torch
from transformers import GeminiForQuestionAnswering, GeminiTokenizer
import docx
import pdfplumber

def read_document(file_path):
    """
    Read a document from a file path and return the text content.
    Supported file formats: .txt, .docx, .pdf
    """
    if file_path.endswith('.txt'):
        with open(file_path, 'r') as f:
            return f.read()
    elif file_path.endswith('.docx'):
        doc = docx.Document(file_path)
        return '\n'.join([p.text for p in doc.paragraphs])
    elif file_path.endswith('.pdf'):
        with pdfplumber.open(file_path) as pdf:
            text = ''
            for page in pdf.pages:
                text += page.extract_text()
            return text
    else:
        raise ValueError("Unsupported file format")

def answer_user_query(document, user_query):
    # Load the pre-trained Gemini model and tokenizer
    model = GeminiForQuestionAnswering.from_pretrained('gemini-base')
    tokenizer = GeminiTokenizer.from_pretrained('gemini-base')

    # Preprocess the document
    input_ids = tokenizer.encode(document, return_tensors='pt')
    attention_mask = tokenizer.encode(document, return_tensors='pt', max_length=512, truncation=True)

    # Prepare the user query
    query_input_ids = tokenizer.encode(user_query, return_tensors='pt')
    query_attention_mask = tokenizer.encode(user_query, return_tensors='pt', max_length=512, truncation=True)

    # Use the Gemini model to answer the user query
    outputs = model(input_ids, attention_mask=attention_mask, question_input_ids=query_input_ids, question_attention_mask=query_attention_mask)
    answer_start_scores = outputs.start_scores
    answer_end_scores = outputs.end_scores

    # Get the predicted answer
    answer_start = torch.argmax(answer_start_scores)
    answer_end = torch.argmax(answer_end_scores) + answer_start

    # Extract the answer from the document
    answer = document[answer_start:answer_end]
    return answer

# Get the document file path
document_file_path = input("Enter the document file path:C:/Users/hhhh/MLproj/MLproj/chatbot/updated CV.pdf ")

# Read the document
document = read_document(document_file_path)

# Get the user query
user_query = input("Enter your question: ")

# Answer the user query
answer = answer_user_query(document, user_query)
print("Answer:", answer)