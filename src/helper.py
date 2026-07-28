"""
Helper Utilities for Ingestion, Chunking, and Embedding Generation.

This module provides reusable helper functions to parse PDF documents,
split text into manageable chunks, clean document metadata, and load
HuggingFace embeddings for vector store operations.
"""

from typing import List
try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document



def load_pdf_file(data_dir: str) -> List[Document]:
    """
    Extracts data from all PDF files located inside the given directory.

    Args:
        data_dir (str): Relative or absolute path to directory containing PDF files.

    Returns:
        List[Document]: List of LangChain Document objects parsed from PDFs.
    """
    from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
    loader = DirectoryLoader(
        data_dir,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    return documents


def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """
    Cleans metadata to retain only necessary fields ('source') and content.
    Reduces vector store payload size.

    Args:
        docs (List[Document]): Raw document list with full metadata.

    Returns:
        List[Document]: Cleaned list of Document objects.
    """
    minimal_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source", "unknown")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": src}
            )
        )
    return minimal_docs


def text_split(extracted_data: List[Document]) -> List[Document]:
    """
    Splits extracted document texts into uniform chunks for embedding generation.

    Args:
        extracted_data (List[Document]): Processed document objects.

    Returns:
        List[Document]: List of chunked Document objects.
    """
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20
    )
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks


def download_hugging_face_embeddings():
    """
    Initializes HuggingFace sentence transformer embeddings model.
    Default model: 'sentence-transformers/all-MiniLM-L6-v2' (Output Dimension: 384).

    Returns:
        HuggingFaceEmbeddings: Pre-trained embedding instance.
    """
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return embeddings

