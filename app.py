import streamlit as st
import os
import tempfile
import fitz  # PyMuPDF

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="📄",
    layout="wide"
)

# ---------------- SESSION STATE ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "pdf_files" not in st.session_state:
    st.session_state.pdf_files = {}

if "selected_pdf" not in st.session_state:
    st.session_state.selected_pdf = None

if "selected_page" not in st.session_state:
    st.session_state.selected_page = 0

# ---------------- API KEY ----------------
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# ---------------- SIDEBAR ----------------
st.sidebar.title("📂 Documents")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF files",
    type="pdf",
    accept_multiple_files=True
)

if st.sidebar.button("Process Documents") and uploaded_files:
    with st.spinner("Processing documents..."):
        docs = []
        pdf_storage = {}

        for uploaded_file in uploaded_files:
            file_name = uploaded_file.name

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name

            pdf_storage[file_name] = tmp_path

            loader = PyPDFLoader(tmp_path)
            file_docs = loader.load()

            for d in file_docs:
                d.metadata["source"] = file_name

            docs.extend(file_docs)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100
        )
        splits = splitter.split_documents(docs)

        embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
        vectorstore = FAISS.from_documents(splits, embeddings)

        st.session_state.retriever = vectorstore.as_retriever()
        st.session_state.pdf_files = pdf_storage

        st.sidebar.success("Documents ready!")

# ---------------- MAIN LAYOUT ----------------
chat_col, viewer_col = st.columns([2, 1])

# ================= CHAT COLUMN =================
with chat_col:
    st.title("🤖 AI Document Assistant")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask something about your documents..."):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        if not st.session_state.retriever:
            response = "Please upload and process documents first."
        else:
            llm = ChatOpenAI(
                model="gpt-4o-mini",
                api_key=OPENAI_API_KEY,
                temperature=0
            )

            prompt_template = ChatPromptTemplate.from_template(
                """
                Answer based only on the context.

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

            # Sources
            sources = result.get("context", [])
            source_blocks = []

            seen = set()
            for doc in sources[:3]:
                filename = doc.metadata.get("source", "Document")
                page = doc.metadata.get("page", 0) + 1
                key = f"{filename}-{page}"

                if key not in seen:
                    source_blocks.append((filename, page))
                    seen.add(key)

            if source_blocks:
                answer += "\n\n**Sources:**\n"
                for i, (fname, page) in enumerate(source_blocks):
                    if st.button(f"📄 {fname} — Page {page}", key=f"src_{i}"):
                        st.session_state.selected_pdf = fname
                        st.session_state.selected_page = page - 1

            response = answer

        st.chat_message("assistant").markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

# ================= PDF VIEWER =================
with viewer_col:
    st.subheader("📄 Document Preview")

    if st.session_state.selected_pdf:
        pdf_name = st.session_state.selected_pdf
        page_num = st.session_state.selected_page

        pdf_path = st.session_state.pdf_files.get(pdf_name)

        if pdf_path:
            doc = fitz.open(pdf_path)
            page = doc.load_page(page_num)
            pix = page.get_pixmap()
            st.image(pix.tobytes(), caption=f"{pdf_name} — Page {page_num+1}")
    else:
        st.info("Click a source to preview the document.")
