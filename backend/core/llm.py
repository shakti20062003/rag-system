import os
from enum import Enum
from typing import List

from langchain_core.documents import Document

# Get your free token at https://huggingface.co/settings/tokens
HF_TOKEN = os.getenv("HF_TOKEN", "")

HF_CHAT_URL = "https://router.huggingface.co/v1/chat/completions"


class LLMProvider(str, Enum):
    MISTRAL = "mistral"
    LLAMA   = "llama"

PROVIDER_MODELS = {
    LLMProvider.MISTRAL: "Qwen/Qwen2.5-72B-Instruct",
    LLMProvider.LLAMA:   "meta-llama/Llama-3.3-70B-Instruct",
}

SYSTEM_PROMPT = (
    "You are a precise assistant that answers questions strictly based on the "
    "provided context. If the context doesn't contain enough information, say "
    "so clearly — do not fabricate."
)


def build_prompt(question: str, docs: List[Document]) -> str:
    context = "\n\n---\n\n".join(
        f"[Source: {d.metadata.get('filename', 'unknown')}]\n{d.page_content}"
        for d in docs
    )
    return f"Context:\n{context}\n\nQuestion: {question}"


def query_hf(prompt: str, model_id: str) -> str:
    import httpx

    if not HF_TOKEN:
        raise ValueError("HF_TOKEN not set. Add it to your .env file.")

    response = httpx.post(
        HF_CHAT_URL,
        headers={"Authorization": f"Bearer {HF_TOKEN}"},
        json={
            "model": model_id,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 512,
            "temperature": 0.2,
        },
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()

    return data["choices"][0]["message"]["content"].strip()


def run_llm(prompt: str, provider: LLMProvider, model: str | None = None) -> str:
    model_id = model or PROVIDER_MODELS.get(provider)
    if not model_id:
        raise ValueError(f"Unknown provider: {provider}")
    return query_hf(prompt, model_id)

