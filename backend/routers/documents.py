import io
from typing import List

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from core.vector_store import clear_vector_store, ingest_documents

router = APIRouter()


def extract_text(filename: str, content: bytes) -> str:
    ext = filename.lower().split(".")[-1]

    if ext == "txt":
        return content.decode("utf-8", errors="ignore")

    if ext == "pdf":
        try:
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(content))
            return "\n\n".join(p.extract_text() or "" for p in reader.pages)
        except ImportError:
            raise HTTPException(400, "pypdf not installed. Run: pip install pypdf")

    if ext in ("md", "markdown"):
        return content.decode("utf-8", errors="ignore")

    raise HTTPException(400, f"Unsupported file type: .{ext}")


class IngestResponse(BaseModel):
    filename: str
    chunks_created: int


@router.post("/upload", response_model=List[IngestResponse])
async def upload_documents(files: List[UploadFile] = File(...)):
    results = []
    for file in files:
        raw = await file.read()
        text = extract_text(file.filename, raw)
        if not text.strip():
            raise HTTPException(400, f"{file.filename} appears empty or unreadable.")
        n = ingest_documents([text], [{"filename": file.filename}])
        results.append(IngestResponse(filename=file.filename, chunks_created=n))
    return results


@router.delete("/clear")
def clear_documents():
    clear_vector_store()
    return {"status": "vector store cleared"}
