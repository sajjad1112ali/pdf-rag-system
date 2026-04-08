import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter


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


if __name__ == "__main__":

    pdf_path = "data/sample.pdf"

    extracted_text = extract_text_from_pdf(pdf_path)

    chunks = chunk_text(extracted_text)

    print("\nTotal Chunks:", len(chunks))

    print("\n--- FIRST CHUNK ---\n")
    print(chunks[0])
