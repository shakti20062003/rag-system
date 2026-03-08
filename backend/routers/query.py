from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from core.llm import LLMProvider, build_prompt, run_llm
from core.vector_store import similarity_search

router = APIRouter()


class QueryRequest(BaseModel):
    question: str
    provider: LLMProvider = LLMProvider.MISTRAL
    model: str | None = None
    top_k: int = 4


class SourceChunk(BaseModel):
    content: str
    filename: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    provider: str


@router.post("/", response_model=QueryResponse)
def query(req: QueryRequest):
    docs = similarity_search(req.question, k=req.top_k)
    if not docs:
        raise HTTPException(404, "No documents in the vector store. Upload documents first.")

    prompt = build_prompt(req.question, docs)
    answer = run_llm(prompt, req.provider, req.model)

    sources = [
        SourceChunk(
            content=d.page_content,
            filename=d.metadata.get("filename", "unknown"),
        )
        for d in docs
    ]
    return QueryResponse(answer=answer, sources=sources, provider=req.provider.value)
