import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


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


def create_embeddings(chunks):

    model = SentenceTransformer("all-MiniLM-L6-v2")

    embeddings = model.encode(chunks)

    return embeddings


if __name__ == "__main__":

    pdf_path = "data/sample.pdf"

    extracted_text = extract_text_from_pdf(pdf_path)

    chunks = chunk_text(extracted_text)

    embeddings = create_embeddings(chunks)

    print("\nTotal Chunks:", len(chunks))

    print("\nEmbedding Vector Length:", len(embeddings[0]))

    print("\nFirst Embedding Vector (first 10 values):")
    print(embeddings[0][:10])
