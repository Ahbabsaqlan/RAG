import streamlit as st
import os
import tempfile
import fitz  # PyMuPDF

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA


# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="📄",
    layout="wide"
)

# ---------------- NOTION STYLE CSS ----------------
st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}
.chat-message {
    padding: 12px 16px;
    border-radius: 10px;
    margin-bottom: 10px;
}
.user-msg {
    background-color: #f2f2f2;
}
.assistant-msg {
    background-color: #ffffff;
    border: 1px solid #e6e6e6;
}
.source-button button {
    width: 100%;
    text-align: left;
}
</style>
""", unsafe_allow_html=True)


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

    # Display chat history
    for msg in st.session_state.messages:
        role_class = "user-msg" if msg["role"] == "user" else "assistant-msg"
        st.markdown(
            f"<div class='chat-message {role_class}'>{msg['content']}</div>",
            unsafe_allow_html=True
        )

    # Chat input
    if prompt := st.chat_input("Ask something about your documents..."):
        st.session_state.messages.append(
            {"role": "user", "content": prompt}
        )

        if not st.session_state.retriever:
            response = "Please upload and process documents first."
            sources = []
        else:
            llm = ChatOpenAI(
                model="gpt-4o-mini",
                api_key=OPENAI_API_KEY,
                temperature=0
            )

            prompt_template = ChatPromptTemplate.from_template(
                """
                Answer only using the provided context.

                Context:
                {context}

                Question:
                {question}
                """
            )

            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=st.session_state.retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": prompt_template}
            )

            result = qa_chain({"query": prompt})
            response = result["result"]
            sources = result["source_documents"]

        # Show assistant message
        st.markdown(
            f"<div class='chat-message assistant-msg'>{response}</div>",
            unsafe_allow_html=True
        )

        st.session_state.messages.append(
            {"role": "assistant", "content": response}
        )

        # Show sources
        if sources:
            st.markdown("**Sources**")

            seen = set()
            for i, doc in enumerate(sources[:3]):
                filename = doc.metadata.get("source", "Document")
                page = doc.metadata.get("page", 0) + 1
                key = f"{filename}-{page}"

                if key not in seen:
                    if st.button(
                        f"📄 {filename} — Page {page}",
                        key=f"src_{i}"
                    ):
                        st.session_state.selected_pdf = filename
                        st.session_state.selected_page = page - 1
                    seen.add(key)


# ================= PDF VIEWER =================
with viewer_col:
    st.subheader("📄 Source Preview")

    if (
        st.session_state.selected_pdf
        and st.session_state.selected_pdf in st.session_state.pdf_files
    ):
        pdf_path = st.session_state.pdf_files[
            st.session_state.selected_pdf
        ]
        page_num = st.session_state.selected_page

        try:
            doc = fitz.open(pdf_path)
            page = doc.load_page(page_num)
            pix = page.get_pixmap(dpi=150)

            image_bytes = pix.tobytes("png")

            st.image(
                image_bytes,
                caption=f"{st.session_state.selected_pdf} — Page {page_num+1}",
                use_container_width=True
            )

            with open(pdf_path, "rb") as f:
                st.download_button(
                    "⬇ Download PDF",
                    data=f,
                    file_name=st.session_state.selected_pdf
                )

        except:
            st.info("Click a source to preview the page.")
    else:
        st.info("No source selected yet.")
