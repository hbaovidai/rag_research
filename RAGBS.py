import streamlit as st
import os
import shutil
import faiss
import numpy as np
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS  # Use FAISS instead of Chroma
from langchain.embeddings import HuggingFaceEmbeddings
import re

def get_embedding_function():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Define paths
FAISS_PATH = "faiss_index"  # Changed from CHROMA_PATH
DATA_PATH = "data"

# Function to save uploaded file
def save_uploaded_file(uploaded_file):
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
    file_path = os.path.join(DATA_PATH, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

from langchain.docstore.document import Document
def load_documents():
    docs = []
    # Read TXT files
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        
    # Read all files in DATA_PATH
    for file in os.listdir(DATA_PATH):
        file_path = os.path.join(DATA_PATH, file)
        if file.endswith(".txt"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    docs.append(Document(page_content=content))
            except Exception as e:
                st.error(f"Error loading {file}: {str(e)}")
        elif file.endswith(".pdf"):
            try:
                loader = PyPDFLoader(file_path)
                docs.extend(loader.load())  # Add PDF content to docs list
            except Exception as e:
                st.error(f"Error loading PDF {file}: {str(e)}")
    return docs

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=256,  # Size to keep context
        chunk_overlap=50  # Slight overlap to ensure no information is lost
    )
    return text_splitter.split_documents(documents)

def add_to_faiss(chunks):
    if not chunks:
        return 0
        
    embedding_function = get_embedding_function()
    
    try:
        # Check if index already exists
        if os.path.exists(FAISS_PATH):
            # Load existing index with allow_dangerous_deserialization=True
            db = FAISS.load_local(FAISS_PATH, embedding_function, allow_dangerous_deserialization=True)
            # Add new documents
            db.add_documents(chunks)
        else:
            # Create new index
            db = FAISS.from_documents(chunks, embedding_function)
        
        # Save the index
        if not os.path.exists(FAISS_PATH):
            os.makedirs(FAISS_PATH)
        db.save_local(FAISS_PATH)
        return len(chunks)
    except Exception as e:
        st.error(f"Error creating FAISS index: {str(e)}")
        return 0

def search_with_faiss(query):
    try:
        embedding_function = get_embedding_function()
        
        # Load the saved index with allow_dangerous_deserialization=True
        if not os.path.exists(FAISS_PATH):
            return "Chưa có dữ liệu được tải lên. Vui lòng tải dữ liệu trước."
            
        db = FAISS.load_local(FAISS_PATH, embedding_function, allow_dangerous_deserialization=True)
        
        # Search
        results = db.similarity_search(query, k=2)
        best_match = results[0].page_content if results else "Không tìm thấy câu trả lời."
        return best_match
    except Exception as e:
        return f"Lỗi khi tìm kiếm: {str(e)}"

def extract_year(text):
    match = re.search(r'\b(19\d{2}|20\d{2})\b', text)
    return match.group(0) if match else "Không tìm thấy năm"

def clean_response(text, query):
    sentences = text.split(". ")  # Split into sentences
    relevant_sentences = [s for s in sentences if query.lower() in s.lower()]
    return ". ".join(relevant_sentences) if relevant_sentences else text  # Return most relevant content

# ============= Streamlit Interface =============
st.set_page_config(page_title="RAG System", layout="wide")
st.title("CHATBOT RAG")

# File upload area
uploaded_file = st.file_uploader("📂 Upload your file:", type=["pdf", "txt"])
if uploaded_file:
    file_path = save_uploaded_file(uploaded_file)
    st.success(f"✅ File '{uploaded_file.name}' đã được tải lên!")

# Load Data button
if st.button("🔄 Load Data"):
    with st.spinner("Đang xử lý dữ liệu..."):
        documents = load_documents()
        if documents:
            chunks = split_documents(documents)
            num_added = add_to_faiss(chunks)
            st.success(f"✅ Đã thêm {num_added} đoạn văn bản vào FAISS index!")
        else:
            st.warning("Không tìm thấy tài liệu nào để xử lý. Vui lòng tải file lên trước.")

# Question input area
st.subheader("💬 Nhập câu hỏi:")
user_input = st.text_area("Enter text:", "Bạn muốn hỏi gì?")
if st.button("Submit"):
    if user_input and user_input != "Bạn muốn hỏi gì?":
        with st.spinner("Đang tìm câu trả lời..."):
            raw_result = search_with_faiss(user_input)
            final_answer = clean_response(raw_result, user_input)
            st.write("💡 **Câu trả lời:**", final_answer)
    else:
        st.warning("Vui lòng nhập câu hỏi của bạn.")
