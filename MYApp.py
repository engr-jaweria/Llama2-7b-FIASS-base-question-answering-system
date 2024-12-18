import os
import tempfile
import numpy as np
import faiss
import requests
import pandas as pd
from sentence_transformers import SentenceTransformer
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import streamlit as st
import fitz  # PyMuPDF for PDF handling
from docx import Document as DocxDocument
from pptx import Presentation
from dotenv import load_dotenv
from langchain.schema import Document

# Load environment variables
load_dotenv()

# Set environment variables
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
if not REPLICATE_API_TOKEN:
    raise ValueError("API token for Replicate is not set in environment variables.")

# Initialize embedding model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
embeddings_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# Utility functions
def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file."""
    text = ""
    with fitz.open(pdf_file) as doc:
        for page in doc:
            text += page.get_text()
    return text

def extract_text_from_docx(docx_file):
    """Extract text from a DOCX file."""
    doc = DocxDocument(docx_file)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text_from_pptx(pptx_file):
    """Extract text from a PPTX file."""
    presentation = Presentation(pptx_file)
    text = ""
    for slide in presentation.slides:
        for shape in slide.shapes:
            if shape.has_text_frame:
                text += shape.text + "\n"
    return text

def extract_text_from_xlsx(xlsx_file):
    """Extract text from an XLSX file."""
    df = pd.read_excel(xlsx_file)
    return df.to_string()

def extract_text_from_url(url):
    """Fetch text content from a URL."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching URL content: {e}")
        return ""

def validate_total_file_size(files, max_size_mb=20):
    """Validate that the total size of uploaded files does not exceed a limit."""
    total_size = sum(file.size for file in files) / (1024 * 1024)  # Convert bytes to MB
    if total_size > max_size_mb:
        st.error(f"Total file size exceeds {max_size_mb} MB. Please reduce the total file size.")
        return False
    return True

def llama2_generate_replicate(prompt, temperature=0.75, max_new_tokens=800):
    """Generate text using the Llama2 model on Replicate."""
    import replicate
    client = replicate.Client(api_token=REPLICATE_API_TOKEN)
    try:
        output = client.run(
            "meta/llama-2-7b-chat",
            input={
                "prompt": prompt,
                "temperature": temperature,
                "max_tokens": max_new_tokens,
                "top_p": 1,
            },
        )
        return output
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return None

# Main Streamlit application
def main():
    st.title("Llama2-7B Q&A System")

    st.sidebar.header("Upload or Provide Input")
    uploaded_files = st.sidebar.file_uploader("Upload files", type=["txt", "pdf", "docx", "pptx", "xlsx"], accept_multiple_files=True)
    web_links = st.sidebar.text_area("Or provide web links (comma-separated):", "").split(',')

    chunk_size = st.sidebar.slider("Set Chunk Size", 500, 2000, 1000)
    documents = []

    # Process uploaded files
    if uploaded_files:
        if not validate_total_file_size(uploaded_files):
            return
        with tempfile.TemporaryDirectory() as temp_dir:
            for uploaded_file in uploaded_files:
                file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.read())
                if uploaded_file.name.endswith(".txt"):
                    with open(file_path, "r", encoding="utf-8") as f:
                        documents.append(f.read())
                elif uploaded_file.name.endswith(".pdf"):
                    documents.append(extract_text_from_pdf(file_path))
                elif uploaded_file.name.endswith(".docx"):
                    documents.append(extract_text_from_docx(file_path))
                elif uploaded_file.name.endswith(".pptx"):
                    documents.append(extract_text_from_pptx(file_path))
                elif uploaded_file.name.endswith(".xlsx"):
                    documents.append(extract_text_from_xlsx(file_path))

    # Process web links
    for web_link in web_links:
        if web_link.strip():
            documents.append(extract_text_from_url(web_link.strip()))

    # Process documents and generate embeddings
    if documents:
        chunks = [doc[i:i + chunk_size] for doc in documents for i in range(0, len(doc), chunk_size)]
        docs = [Document(page_content=chunk, metadata={"source": "uploaded file"}) for chunk in chunks]
        embeddings = embeddings_model.embed_documents(chunks)
        embeddings_array = np.array(embeddings)
        dimension = embeddings_array.shape[1]
        faiss_index = faiss.IndexFlatL2(dimension)
        faiss_index.add(embeddings_array.astype(np.float32))

        # Ask a question
        st.header("Ask a Question")
        question = st.text_input("Your Question:")
        format_choice = st.selectbox("Answer Format:", ["Default", "Bullet Points", "Summary", "Specific Length"])
        word_limit = st.number_input("Word/Character Limit:", min_value=10, max_value=500, value=100) if format_choice == "Specific Length" else None

        if st.button("Get Answer") and question:
            context = " ".join(chunks)
            prompt = f"Answer the question based on the context: {context}\n\nQuestion: {question}"
            if format_choice == "Bullet Points":
                prompt += "\nProvide the answer as bullet points."
            elif format_choice == "Summary":
                prompt += "\nProvide a concise summary."
            elif format_choice == "Specific Length":
                prompt += f"\nLimit the answer to {word_limit} words."

            answer = llama2_generate_replicate(prompt)
            if answer:
                st.markdown(f"### Generated Response:\n{answer}")

    else:
        st.write("Upload documents or provide a web link to begin.")

if __name__ == "__main__":
    main()
