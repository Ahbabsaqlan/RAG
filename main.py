import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Load API key
load_dotenv()

# 1. Load PDF
loader = PyPDFLoader("DOC.pdf")
documents = loader.load()

# 2. Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
docs = text_splitter.split_documents(documents)

# 3. Create embeddings
embeddings = OpenAIEmbeddings()

# 4. Store in vector database
vectorstore = FAISS.from_documents(docs, embeddings)

# 5. Create LLM
llm = ChatOpenAI(temperature=0)

# 6. Create QA chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever()
)

# 7. Chat loop
while True:
    query = input("\nAsk a question about the PDF (or type 'exit'): ")
    if query.lower() == "exit":
        break

    answer = qa.run(query)
    print("Answer:", answer)
