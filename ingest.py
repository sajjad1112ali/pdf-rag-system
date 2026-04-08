import fitz

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


PDF_PATH = "data/sample.pdf"
VECTOR_DB_PATH = "vectorstore"


def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)

    text = ""
    for page in doc:
        text += page.get_text()

    return text


def chunk_text(text):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    return splitter.split_text(text)


def build_vector_db(chunks):

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_texts(chunks, embeddings)

    vectorstore.save_local(VECTOR_DB_PATH)

    print("Vector database saved successfully!")


if __name__ == "__main__":

    print("Reading PDF...")
    text = extract_text_from_pdf(PDF_PATH)

    print("Chunking...")
    chunks = chunk_text(text)

    print("Creating vector database...")
    build_vector_db(chunks)
