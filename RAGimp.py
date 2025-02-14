import fitz  # PyMuPDF

# Function to extract text from multiple PDFs
def extract_text_from_multiple_pdfs(pdf_paths):
    extracted_texts = {}  # Dictionary to store extracted texts with file names
    for pdf_path in pdf_paths:
        # Open the PDF file
        doc = fitz.open(pdf_path)
        text = ""
        
        # Loop through all pages in the PDF and extract text
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)  # Load each page
            text += page.get_text("text")  # Extract text from the page
            
        extracted_texts[pdf_path] = text  # Store the extracted text with the file path as the key
    
    return extracted_texts

# Example usage
pdf_paths = [
    r"C:\Users\etc",
]  # Add paths to your PDF files here

extracted_texts = extract_text_from_multiple_pdfs(pdf_paths)  # Call the function

from transformers import AutoTokenizer, AutoModel
import torch

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-large")
model = AutoModel.from_pretrained("intfloat/multilingual-e5-large")

# Function to generate embedding from text
def generate_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)  # Mean pooling of token embeddings
    return embeddings.squeeze().numpy()  # Convert to numpy

# Generate embeddings for each extracted resume text
resume_embeddings = {}
for path, text in extracted_texts.items():
    embedding = generate_embedding(text)  # Generate embedding for the extracted text
    resume_embeddings[path] = embedding

import os
from pinecone import Pinecone, ServerlessSpec

# Initialize Pinecone using the Pinecone class
pc = Pinecone(api_key="b575aa98-8790-4450-8dfc-b77dd6fbb121")

# Create or connect to an index
if 'linkedindb' not in pc.list_indexes().names():
    pc.create_index(
        name='linkedindb',  # Index name
        dimension=1024,  # Model output dimension (e.g., 1024 for multilingual-e5-large)
        metric='cosine',  # Metric for similarity
        spec=ServerlessSpec(
            cloud='aws',  # Cloud provider
            region='us-east-1'  # Your region
        )
    )

# Connect to the created index
index = pc.Index("linkedindb")

# Upsert each resume embedding into Pinecone
for path, embedding in resume_embeddings.items():
    index.upsert([(path, embedding)])  # Upsert (ID, embedding)
def generate_embedding(text):
    # Generate embedding using the API (adjust based on the response structure)
    embedding_response = pc.inference.embed(
        model="multilingual-e5-large",  # Using the same model as before
        inputs=[text],  # Text to generate embedding for
        parameters={"input_type": "passage", "truncate": "END"}
    )
    
    # Extract the embedding values (assuming it is in 'values' field)
    embedding = embedding_response.data[0]['values']
    
    print(f"Generated embedding for text '{text}': {embedding}, shape: {len(embedding)}")
    
    return embedding  # Return the list of floats (the actual embedding vector)

def query_resume(query_text):
    # Generate embedding for the query text
    query_embedding = generate_embedding(query_text)

    # Perform similarity search in the Pinecone index
    query_results = index.query(vector=query_embedding, top_k=5, include_metadata=True)

    # Print the most similar resumes
    for match in query_results['matches']:
        print(f"Resume: {match['id']}, Score: {match['score']}")

# Example usage
query_text = "Software engineer with experience in Python and AI development"
query_resume(query_text)
