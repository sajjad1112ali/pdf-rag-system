import fitz
import ollama

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


def extract_text_from_pdf(pdf_path):

    doc = fitz.open(pdf_path)
    text = ""

    for page in doc:
        text += page.get_text()

    return text


def chunk_text(text):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = text_splitter.split_text(text)

    return chunks


def create_vector_store(chunks):

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_texts(chunks, embedding_model)

    return vectorstore


def retrieve_chunks(vectorstore, question):

    docs = vectorstore.similarity_search(question, k=3)

    context = "\n\n".join([doc.page_content for doc in docs])

    return context


def ask_llm(question, context):

    prompt = f"""
You are a helpful assistant.

Answer the question using ONLY the context below.

Context:
{context}

Question:
{question}
"""

    response = ollama.chat(
        model="mistral",
        messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]


if __name__ == "__main__":

    pdf_path = "data/me.pdf"

    print("\nReading PDF...")

    text = extract_text_from_pdf(pdf_path)

    print("Chunking text...")

    chunks = chunk_text(text)

    print("Creating vector database...")

    vectorstore = create_vector_store(chunks)

    print("\nSystem ready! Ask questions about the PDF.\n")

    while True:

        question = input("Ask a question (or type 'exit'): ")

        if question.lower() == "exit":
            break

        context = retrieve_chunks(vectorstore, question)

        answer = ask_llm(question, context)

        print("\nAnswer:\n", answer)
