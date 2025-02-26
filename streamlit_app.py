import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.prompts import ChatPromptTemplate

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def generate_response(input_text):
    # Select embedding
    embeddings = GoogleGenerativeAIEmbeddings(google_api_key=google_api_key)
    # Vector store
    db = Chroma(embedding_function=embeddings)  
    # Search the DB
    results = db.similarity_search_with_relevance_scores(input_text, k=3)
    if len(results) == 0 or results[0][1] < 0.7:
        st.info("Không tìm thấy kết quả nào ")      
    else:
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=input_text)
        #st.info(prompt)
        model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7, google_api_key=google_api_key)
        st.info(model.invoke(prompt).content)

# Page title
st.title = "FINAL PROJECT"
google_api_key = st.sidebar.text_input("Google Gemini API Key", type="password")

# File upload
uploaded_file = st.file_uploader('Upload your file:', type='txt')
if st.button("Load Data"):
    # Load document if file is uploaded
    if uploaded_file is not None:
        documents = [uploaded_file.read().decode()]
        # Split documents into chunks
        text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=100)
        chunks = text_splitter.create_documents(documents)
        
        # Select embedding
        embeddings = GoogleGenerativeAIEmbeddings(google_api_key=google_api_key)
        # Create vector store from document
        db = Chroma.from_documents(chunks, embeddings)    
        'Data Load OK'

with st.form("my_form"):
    text = st.text_area(
        "Enter text:",
        "Bạn muốn hỏi gì?",
    )
    submitted = st.form_submit_button("Submit", disabled=not uploaded_file)
    if not google_api_key:
        st.warning("Please enter your Google Gemini API Key", icon=None)
    if submitted and google_api_key:
        generate_response(text)


