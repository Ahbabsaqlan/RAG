import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA


# Load API key
load_dotenv()

st.set_page_config(page_title="PDF Chatbot", page_icon="🤖")
st.title("📄 Chat with your PDF")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    # Save file temporarily
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    # Load PDF
    loader = PyPDFLoader("temp.pdf")
    documents = loader.load()

    # Split text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    docs = text_splitter.split_documents(documents)

    # Embeddings
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)

    # LLM
    llm = ChatOpenAI(temperature=0)

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever()
    )

    # Chat UI
    query = st.text_input("Ask a question about the PDF:")

    if query:
        answer = qa.run(query)
        st.success(answer)
