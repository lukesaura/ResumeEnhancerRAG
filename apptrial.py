import streamlit as st
from transformers import AutoTokenizer, AutoModel
import torch
import pinecone
from PyPDF2 import PdfReader
from io import BytesIO
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from docx import Document
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os
import fitz  # PyMuPDF

# Load environment variables (Hugging Face API key)
load_dotenv()

# Hugging Face API settings
repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
hf_api_token = os.getenv("use your huggingface api token")

# Initialize Pinecone using the Pinecone class
pc = Pinecone(api_key="pinecone api token")

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

# Initialize tokenizer and model for embeddings
tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-large")
model = AutoModel.from_pretrained("intfloat/multilingual-e5-large")


# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


# Function to generate prompt
def generate_prompt(profile_text, section):
    template = f""" 
    You are an AI designed to read, analyze, and provide optimization suggestions for LinkedIn profiles.
    Given a LinkedIn profile, generate a detailed and professional report tailored to the profile owner.
    The report should assess the current optimization level of each section in percentage and provide actionable suggestions for improvement 
    The report should be structured clearly, with specific, actionable suggestions for improvement in the following section:
    
    {section}
    Please provide how much it is optimized already.
 
    Extracted LinkedIn Profile:
    {profile_text}
    """
    return template


# Function to get LLM response from Hugging Face
def get_llm_response(prompt, profile_text):
    llm = HuggingFaceEndpoint(
        repo_id=repo_id, max_new_tokens=2048, temperature=0.5, huggingfacehub_api_token=hf_api_token
    )
    prompt_template = PromptTemplate.from_template(prompt)
    llm_chain = prompt_template | llm
    response = llm_chain.invoke({"profile_text": profile_text})
    return response


# Function to generate embedding for text
def generate_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings.squeeze().numpy()


# Function to retrieve relevant resumes from Pinecone using query embedding
def retrieve_relevant_resumes(profile_embedding):
    # Assuming profile_embedding is a NumPy array, convert it to a list
    query_results = index.query(vector=profile_embedding.tolist(), top_k=5, include_metadata=True)
    return [match['id'] for match in query_results['matches']]


# Function to generate suggestions based on profile and relevant resumes
def generate_suggestions(profile_text, section, resumes):
    st.subheader(f"{section} Suggestions")
    with st.spinner(f"Generating {section} Suggestions..."):
        # Generate prompt using profile text and resumes
        prompt = generate_prompt(profile_text, section)
        suggestions = get_llm_response(prompt, profile_text)
    st.text_area(f"{section} Suggestions", suggestions, height=300)

    st.download_button(
        label="Download as Text",
        data=save_as_text(suggestions),
        file_name=f"LinkedIn_Profile_{section}_Suggestions.txt",
        mime="text/plain"
    )
    st.download_button(
        label="Download as Word",
        data=save_as_word(suggestions),
        file_name=f"LinkedIn_Profile_{section}_Suggestions.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )


# Function to save suggestions as text
def save_as_text(content):
    return BytesIO(content.encode('utf-8'))


# Function to save suggestions as Word document
def save_as_word(content):
    doc = Document()
    doc.add_paragraph(content)
    buffer = BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer


# Streamlit App
def main():
    st.set_page_config(page_title="LinkedIn Profile Optimizer", page_icon="📄", layout="centered")
    st.title("📄 AI LinkedIn Profile Optimizer")
    
    st.markdown("""
    ### Welcome to the LinkedIn Profile Optimizer!
    
    Upload a PDF of your LinkedIn profile, and we'll analyze it to provide you with a comprehensive summary and suggestions for improvement.
    """)

    pdf_file = st.file_uploader("Upload your LinkedIn profile in PDF format", type=["pdf"])
    
    if pdf_file is not None:
        st.info("File uploaded successfully. Extracting text...")
        pdf_bytes = BytesIO(pdf_file.read())
        profile_text = extract_text_from_pdf(pdf_bytes)
        
        st.subheader("Extracted Text")
        st.text_area("Extracted Text", profile_text, height=300)

        # Generate embeddings for the uploaded profile text
        profile_embedding = generate_embedding(profile_text)

        # Retrieve similar resumes from Pinecone
        relevant_resumes = retrieve_relevant_resumes(profile_embedding)

        if st.button("Get Headline Enhancement Suggestions"):
            generate_suggestions(profile_text, "Headline Enhancement", relevant_resumes)

        if st.button("Get Summary Optimization Suggestions"):
            generate_suggestions(profile_text, "Summary Optimization", relevant_resumes)

        if st.button("Get Detailed Experience Descriptions Suggestions"):
            generate_suggestions(profile_text, "Detailed Experience Descriptions", relevant_resumes)

        if st.button("Get Skills Section Enhancement Suggestions"):
            generate_suggestions(profile_text, "Skills Section Enhancement", relevant_resumes)

        if st.button("Get Keywords Suggestions"):
            generate_suggestions(profile_text, "Keywords", relevant_resumes)

        if st.button("Get Profile Summary Suggestions"):
            generate_suggestions(profile_text, "Profile Summary", relevant_resumes)


if __name__ == "__main__":
    main()
