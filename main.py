import os
import faiss
from dotenv import load_dotenv
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

DB_NAME = "my_story"
DATA_PATH = "rag-dataset"
OLLAMA_URL = "http://localhost:11434"


# Store RAG Data
def store_rag_data():
    # Load PDFs
    docs = []
    for root, _, files in os.walk(DATA_PATH):
        for file in files:
            if file.endswith(".pdf"):
                loader = PyMuPDFLoader(os.path.join(root, file))
                docs.extend(loader.load())

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100
    )
    chunks = splitter.split_documents(docs)

    # Embeddings
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url=OLLAMA_URL
    )

    # FAISS index
    dim = len(embeddings.embed_query("test"))
    index = faiss.IndexFlatL2(dim)

    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={}
    )

    # Store
    vector_store.add_documents(chunks)
    vector_store.save_local(DB_NAME)

    print("✅ RAG data stored successfully")


# QUERY RAG
def query_rag(question: str):
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url=OLLAMA_URL
    )

    # Load vector DB
    vector_store = FAISS.load_local(
        DB_NAME,
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3}
    )

    docs = retriever.invoke(question)
    print(docs)
    context = "\n\n".join([doc.page_content for doc in docs])

    # LLM
    model = ChatOllama(
        model="qwen3.5:0.8b",
        base_url=OLLAMA_URL
    )

    prompt = ChatPromptTemplate.from_template("""
        You are an assistant for question-answering tasks.

        Use ONLY the provided context.
        If answer is not in context, say "I don't know".

        Answer in bullet points.

        Question: {question}
        Context: {context}
        Answer:
    """)

    chain = prompt | model
    print("before calling ollama")
    response = chain.invoke({
        "question": question,
        "context": context
    })
    print(response)
    return response.content

# Main Function
if __name__ == "__main__":
    # store_rag_data()  # Run once

    answer = query_rag("What is Harshil's school name?")
    print(answer)