# Local PDF Question Answering System (RAG)

A fully local AI system that answers questions from PDF documents using Retrieval-Augmented Generation (RAG).

The system extracts text from PDFs, converts them into embeddings, stores them in a FAISS vector database, and retrieves relevant document sections to generate answers using a local LLM.

## Features

• Ask questions about any PDF
• Local LLM (Mistral via Ollama)
• Semantic search using embeddings
• Persistent FAISS vector database
• Streaming responses like ChatGPT
• Simple Streamlit web interface
• 100% open-source (no paid APIs)

## Architecture

PDF
↓
Text Extraction (PyMuPDF)
↓
Text Chunking (LangChain)
↓
Embeddings (SentenceTransformers)
↓
Vector Database (FAISS)
↓
Similarity Retrieval
↓
Local LLM (Mistral via Ollama)
↓
Answer Generation

## Tech Stack

Python
LangChain
SentenceTransformers
FAISS
Ollama
Mistral LLM
Streamlit
PyMuPDF

## Installation

```bash
git clone https://github.com/yourusername/pdf-rag-system
cd pdf-rag-system
pip install -r requirements.txt
```

Install Ollama and download the model:

```bash
ollama pull mistral
```

## Build Vector Database

```bash
python ingest.py
```

## Run the App

```bash
streamlit run app.py
```

Open browser:

http://localhost:8501

## Example Use Cases

• AI knowledge base for documents
• Customer support documentation search
• Legal document analysis
• Research paper Q&A
• Internal company knowledge assistants

## Future Improvements

• Multi-PDF knowledge base
• Chat memory
• Drag-and-drop PDF upload
• Hybrid search (BM25 + vector search)
• Docker deployment
