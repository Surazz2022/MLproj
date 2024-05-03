import os

def load_documents():
    # Load document data from files
    documents = []
    for file in os.listdir('data/documents'):
        with open(f'data/documents/{file}', 'r') as f:
            documents.append(f.read())
    return documents