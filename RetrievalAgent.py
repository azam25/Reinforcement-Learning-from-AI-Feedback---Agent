import logging
import os
from pathlib import Path
from typing import Optional

import openai
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .settings import settings

logger = logging.getLogger(__name__)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=30),
    retry=retry_if_exception_type((openai.APIError, openai.APIConnectionError, openai.RateLimitError)),
    reraise=True,
)
def _build_faiss_index(texts, embeddings):
    """Build FAISS index with retry on transient embedding API errors."""
    return FAISS.from_documents(texts, embeddings)


def load_and_process_document(
    file_path: str,
    index_dir: Optional[str] = None,
    force_rebuild: bool = False,
):
    """
    Load a PDF, embed it, and return a retriever.

    The FAISS index is persisted to disk (index_dir) so that subsequent
    calls with the same file can skip re-embedding (cheaper + faster).

    Args:
        file_path:     Path to the PDF file.
        index_dir:     Directory for the persisted FAISS index.
                       Defaults to settings.faiss_index_dir.
        force_rebuild: Force re-embedding even if an index already exists.

    Returns:
        A LangChain retriever backed by the FAISS index.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Document not found: {file_path}")

    # Use a per-document sub-directory so different docs don't collide.
    base_dir = index_dir or settings.faiss_index_dir
    safe_name = Path(file_path).stem.replace(" ", "_")
    persist_path = os.path.join(base_dir, safe_name)

    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
        base_url=settings.openai_api_base,
    )

    if not force_rebuild and os.path.isdir(persist_path):
        logger.info("Loading existing FAISS index from %s", persist_path)
        db = FAISS.load_local(
            persist_path,
            embeddings,
            allow_dangerous_deserialization=True,
        )
    else:
        logger.info("Building FAISS index for %s", file_path)
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        texts = text_splitter.split_documents(documents)

        db = _build_faiss_index(texts, embeddings)
        os.makedirs(persist_path, exist_ok=True)
        db.save_local(persist_path)
        logger.info("FAISS index saved to %s", persist_path)

    retriever = db.as_retriever(search_kwargs={"k": settings.retrieval_k})
    logger.info("Retriever ready (k=%d) for document: %s", settings.retrieval_k, file_path)
    return retriever