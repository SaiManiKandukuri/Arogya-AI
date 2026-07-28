"""
Pinecone Vector Database Indexing Script.

Executes document ingestion, chunking, embedding generation using HuggingFace
models, creates a serverless Pinecone vector index (dimension=384), and upserts
all chunks into the index for similarity search.
"""

import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from src.helper import (
    load_pdf_file,
    filter_to_minimal_docs,
    text_split,
    download_hugging_face_embeddings
)

# Load environment variables from .env
load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

if not PINECONE_API_KEY:
    raise ValueError(
        "PINECONE_API_KEY is not set in environment or .env file! "
        "Please paste your Pinecone API key in the .env file."
    )

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


def main():
    print("[INDEXING] Starting PDF Document Ingestion from data/ ...")
    extracted_data = load_pdf_file(data_dir='data/')
    print(f"[INDEXING] Loaded {len(extracted_data)} total document pages.")

    print("[INDEXING] Cleaning and filtering metadata...")
    filtered_data = filter_to_minimal_docs(extracted_data)

    print("[INDEXING] Splitting document into text chunks...")
    text_chunks = text_split(filtered_data)
    print(f"[INDEXING] Created {len(text_chunks)} text chunks.")

    print("[INDEXING] Downloading HuggingFace embeddings model (sentence-transformers/all-MiniLM-L6-v2)...")
    embeddings = download_hugging_face_embeddings()

    print("[INDEXING] Initializing Pinecone Client...")
    pc = Pinecone(api_key=PINECONE_API_KEY)

    index_name = "medical-chatbot"

    # Check if vector index exists; create serverless index if it doesn't
    if not pc.has_index(index_name):
        print(f"[INDEXING] Index '{index_name}' not found. Creating serverless index...")
        pc.create_index(
            name=index_name,
            dimension=384,  # Matching sentence-transformers output dimension
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        print(f"[INDEXING] Created Pinecone index '{index_name}'.")
    else:
        print(f"[INDEXING] Pinecone index '{index_name}' already exists.")

    print("[INDEXING] Upserting vector embeddings to Pinecone...")
    docsearch = PineconeVectorStore.from_documents(
        documents=text_chunks,
        index_name=index_name,
        embedding=embeddings
    )
    print("[INDEXING] Vector indexing complete! All document vectors are ready for retrieval.")


if __name__ == '__main__':
    main()
