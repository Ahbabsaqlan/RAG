import streamlit as st
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.schema import Document
from PIL import Image
import io

st.set_page_config(
    page_title="AI Document Assistant",
    layout="wide",
    page_icon="📄"
)

# ---------------- UI STYLING ----------------
st.markdown("""
<style>
body {
    background-color: #0e1117;
}

.user-bubble {
    background: #1f6feb;
    color: white;
    padding: 12px 16px;
    border-radius: 12px;
    margin: 6px 0;
    text-align: right;
    max-width: 75%;
    margin-left: auto;
}

.assistant-bubble {
    background: #30363d;
    color: white;
    padding: 12px 16px;
    border-radius: 12px;
    margin: 6px 0;
    text-align: left;
    max-width: 75%;
}
</style>
""", unsafe_allow_html=True)

st.title("📄 AI Document Assistant")

# ---------------- SESSION STATE ----------------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "pdf_bytes" not in st.session_state:
    st.session_state.pdf_bytes = {}

if "selected_source" not in st.session_state:
    st.session_state.selected_source = None

if "selected_page" not in st.session_state:
    st.session_state.selected_page = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("📂 Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:
        for file in uploaded_files:
            st.markdown(f"**{file.name}**")
            if st.button("Preview", key=f"preview_{file.name}"):
                st.session_state.selected_source = file.name
                st.session_state.selected_page = 1
                st.session_state.pdf_bytes[file.name] = file.read()

    if st.button("Process Documents"):
        if uploaded_files:
            all_docs = []

            for file in uploaded_files:
                pdf_bytes = file.read()
                st.session_state.pdf_bytes[file.name] = pdf_bytes

                pdf = fitz.open(stream=pdf_bytes, filetype="pdf")

                for i, page in enumerate(pdf):
                    text = page.get_text()
                    if text.strip():
                        all_docs.append(
                            Document(
                                page_content=text,
                                metadata={
                                    "source": file.name,
                                    "page": i + 1
                                }
                            )
                        )

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=150
            )

            chunks = splitter.split_documents(all_docs)

            embeddings = OpenAIEmbeddings()
            vectorstore = FAISS.from_documents(chunks, embeddings)

            st.session_state.vectorstore = vectorstore
            st.success("Documents processed!")

# ---------------- MAIN LAYOUT ----------------
col1, col2 = st.columns([2, 1])

# ---------------- CHAT AREA ----------------
with col1:

    # Show chat history
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(
                f"<div class='user-bubble'>{msg['content']}</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div class='assistant-bubble'>{msg['content']}</div>",
                unsafe_allow_html=True
            )

    # Chat input
    question = st.chat_input("Ask about your documents...")

    if question:
        st.session_state.messages.append(
            {"role": "user", "content": question}
        )

        if st.session_state.vectorstore:
            retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 4})
            llm = ChatOpenAI(temperature=0)

            qa = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=retriever,
                return_source_documents=True
            )

            result = qa(question)
            answer = result["result"]

            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )

            # Save sources
            st.session_state.last_sources = result["source_documents"]

        else:
            answer = "Please process documents first."
            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )

        st.rerun()

    # Show sources under last answer
    if "last_sources" in st.session_state:
        st.subheader("Sources")
        for i, doc in enumerate(st.session_state.last_sources):
            source = doc.metadata["source"]
            page = doc.metadata["page"]
            text = doc.page_content

            label = f"{source} — page {page}"

            if st.button(label, key=f"src_{i}"):
                st.session_state.selected_source = source
                st.session_state.selected_page = page
                st.session_state.selected_text = text
                st.rerun()

# ---------------- SOURCE PREVIEW ----------------
with col2:
    st.subheader("Source Preview")

    if st.session_state.selected_source:
        if st.button("❌ Close Preview"):
            st.session_state.selected_source = None
            st.session_state.selected_page = None
            st.rerun()

        source = st.session_state.selected_source
        page_num = st.session_state.selected_page
        source_text = st.session_state.get("selected_text", "")

        if source in st.session_state.pdf_bytes:
            pdf_bytes = st.session_state.pdf_bytes[source]
            pdf = fitz.open(stream=pdf_bytes, filetype="pdf")
            page = pdf[page_num - 1]

            # Highlight
            if source_text:
                areas = page.search_for(source_text[:200])
                for area in areas:
                    highlight = page.add_highlight_annot(area)
                    highlight.update()

            pix = page.get_pixmap(dpi=150)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            st.image(img, use_column_width=True)
            st.caption(f"{source} — page {page_num}")

        else:
            st.warning("Source file not found.")
    else:
        st.info("Click a source to preview.")
