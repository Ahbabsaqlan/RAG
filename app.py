import streamlit as st
import os
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain

# ---------------- UI CONFIG ----------------
st.set_page_config(
    page_title="Ahbabs RAG Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("📄 RAG Assistant")
st.caption("Ask questions about your uploaded documents")

# ---------------- API KEY ----------------
if "OPENAI_API_KEY" not in st.secrets:
    st.error("Please add OPENAI_API_KEY in Streamlit secrets.")
    st.stop()

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# ---------------- CHAT HISTORY ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------- SIDEBAR ----------------
st.sidebar.header("Upload Documents feeling")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF files",
    type="pdf",
    accept_multiple_files=True
)

process_button = st.sidebar.button("Process Documents")

# ---------------- PROCESS DOCUMENTS ----------------
if process_button and uploaded_files:
    with st.spinner("Processing documents..."):
        docs = []

        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name

            loader = PyPDFLoader(tmp_path)
            docs.extend(loader.load())

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100
        )
        splits = splitter.split_documents(docs)

        embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
        vectorstore = FAISS.from_documents(splits, embeddings)

        st.session_state.retriever = vectorstore.as_retriever()

        st.success("Documents processed successfully!")

# ---------------- DISPLAY CHAT HISTORY ----------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------- CHAT INPUT ----------------
if prompt := st.chat_input("Ask a question about your documents..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    if "retriever" not in st.session_state:
        response = "Please upload and process documents first."
    else:
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=OPENAI_API_KEY,
            temperature=0
        )

        prompt_template = ChatPromptTemplate.from_template(
            """
            Answer the question based only on the context below.

            Context:
            {context}

            Question:
            {input}
            """
        )

        document_chain = create_stuff_documents_chain(llm, prompt_template)
        retrieval_chain = create_retrieval_chain(
            st.session_state.retriever,
            document_chain
        )

        result = retrieval_chain.invoke({"input": prompt})
        answer = result["answer"]

        # Source citations
        sources = result.get("context", [])
        if sources:
            source_text = "\n\n**Sources:**\n"
            for i, doc in enumerate(sources[:3]):
                source_text += f"- Page {doc.metadata.get('page', 'N/A')}\n"
            answer += source_text

        response = answer

    st.chat_message("assistant").markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
