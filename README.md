# RAG Q&A System

Retrieval-Augmented Generation pipeline with FastAPI backend and chat UI.  
100% free — uses **HuggingFace free tier** for LLMs and **local sentence-transformers** for embeddings.

## Stack

| Layer | Tech | Cost |
|---|---|---|
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local) | Free |
| Vector Store | FAISS (local, persisted) | Free |
| Chunking | LangChain `RecursiveCharacterTextSplitter` | Free |
| LLM option A | Mistral-7B-Instruct via HF Inference API | Free |
| LLM option B | Llama-3.2-3B-Instruct via HF Inference API | Free |
| API | FastAPI | Free |
| Frontend | Vanilla HTML/CSS/JS | Free |

---

## Setup

### 1. Get a free HuggingFace token

1. Sign up at [huggingface.co](https://huggingface.co)
2. Go to **Settings → Access Tokens → New token** (select `read`)
3. Copy the token starting with `hf_...`

### 2. Install dependencies

```bash
cd backend
pip install -r requirements.txt
```

> First run will download the embedding model (~90MB) automatically.

### 3. Configure environment

```bash
cp .env.example .env
# Paste your HF_TOKEN into .env
```

---

## Run

```bash
cd backend
uvicorn main:app --reload --port 8000
```

Open `frontend/index.html` in your browser (or serve it):

```bash
python -m http.server 3000 --directory frontend
```

---

## API

### `POST /documents/upload`

```bash
curl -X POST http://localhost:8000/documents/upload \
  -F "files=@my_doc.pdf"
```

### `POST /query/`

```json
{
  "question": "What is the refund policy?",
  "provider": "mistral",   // or "llama"
  "top_k": 4
}
```

### `DELETE /documents/clear`
Wipe the FAISS vector store.

---

## Project Structure

```
rag-system/
├── backend/
│   ├── main.py                  # FastAPI app
│   ├── requirements.txt
│   ├── .env.example
│   ├── core/
│   │   ├── vector_store.py      # FAISS + HF sentence-transformers
│   │   └── llm.py               # HF Inference API (Mistral + LLaMA)
│   └── routers/
│       ├── documents.py         # Upload endpoints
│       └── query.py             # Query endpoint
└── frontend/
    └── index.html               # Chat UI
```

---

## How it works

1. **Ingest** — Uploaded files are split into ~500-token chunks with 50-token overlap.
2. **Embed** — Each chunk is embedded locally via `all-MiniLM-L6-v2` and stored in FAISS.
3. **Retrieve** — On query, top-K semantically similar chunks are fetched.
4. **Generate** — Retrieved chunks are injected into a prompt sent to Mistral or LLaMA via HuggingFace free API.
5. **Respond** — The API returns the answer + source chunks for transparency.

## Notes on HF Free Tier

- Rate limits apply (~1000 req/day for Inference API). For heavier use, run models locally via [Ollama](https://ollama.com).
- First request to a model may be slow (cold start ~10–20s) — HF loads the model on demand.
- LLaMA 3.2 requires accepting the model license on HuggingFace before use.

