import shutil
from pathlib import Path
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

VECTOR_STORE_PATH = Path("vector_store")

# Free, runs locally — no API key needed
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def load_vector_store() -> FAISS | None:
    if not (VECTOR_STORE_PATH / "index.faiss").exists():
        return None
    return FAISS.load_local(
        str(VECTOR_STORE_PATH),
        get_embeddings(),
        allow_dangerous_deserialization=True,
    )


def ingest_documents(texts: List[str], metadatas: List[dict]) -> int:
    docs = [Document(page_content=t, metadata=m) for t, m in zip(texts, metadatas)]
    chunks = _splitter.split_documents(docs)

    embeddings = get_embeddings()
    existing = load_vector_store()

    if existing:
        existing.add_documents(chunks)
        existing.save_local(str(VECTOR_STORE_PATH))
    else:
        VECTOR_STORE_PATH.mkdir(exist_ok=True)
        store = FAISS.from_documents(chunks, embeddings)
        store.save_local(str(VECTOR_STORE_PATH))

    return len(chunks)


def similarity_search(query: str, k: int = 4) -> List[Document]:
    store = load_vector_store()
    if not store:
        return []
    return store.similarity_search(query, k=k)


def clear_vector_store():
    if VECTOR_STORE_PATH.exists():
        shutil.rmtree(VECTOR_STORE_PATH)
