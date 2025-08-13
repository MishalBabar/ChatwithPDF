## Chat With PDF Backend System

This repository implements a backend service that allows users to ask questions over a corpus of PDF documents, with built‑in support for:

1. **Retrieval-Augmented Generation (RAG)** : answers grounded in the provided PDFs

2. **Clarification Agent**: detects and resolves ambiguous queries

3. **Web Search Agent**: fetches online results when queries fall outside the PDF corpus

4. **Session Memory**: maintains context over a single user session

5. **RESTful API**: endpoints to ask questions and clear memory

## Architecture Overview

### Agent Roles

**Clarifier Agent** – Detects ambiguous/underspecified queries. Returns either a single clarifying question or OK to proceed.

**RAG Agent** – Embeds the question, retrieves top‑K chunks from the FAISS index, and answers grounded in the retrieved context. Always returns a Sources list.

**Web Search Agent** – Used only when explicitly requested, or when there are zero PDF hits and web fallback is enabled. Uses Tavily if TAVILY_API_KEY is set, otherwise DuckDuckGo with backoff.

### Orchestration (LangGraph)

**PDF‑first policy**: If retrieval returns any hits, the system answers from PDFs and does not query the web.

**Fallback control**: ALLOW_WEB_FALLBACK=false by default; set to true to enable web when PDFs have zero hits.

## Getting Started

### Prerequisites

1. Docker & Docker Compose

2. An OpenAI API key.
3. A Tavily API key for web search.

### Environment Variables

Create a .env file in the repo root:

OPENAI_API_KEY=***
TAVILY_API_KEY=***


1) ***Build containers***

docker-compose up --build -d

2) Ingest PDFs into the vector index

Place .pdf files in ./data/, then run ingestion inside the container:

docker compose exec app python ingestion/ingest_pdfs.py --pdf_dir ./data --index_path ./vector_index


3) ***Health check***

curl http://localhost:8000/healthz

4) ***Ask a question***

curl -X POST http://localhost:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"Which prompt template gave the highest zero-shot accuracy on Spider in Zhang et al. (2024)?"}'

## Future Improvements

1. ***Persistence Layer***: Swap FAISS + local pickle for a managed vector DB (Weaviate, Pinecone, pgvector) with auth, backups, and hybrid search.

2. ***Evaluation & Confidence***: Golden Q&A sets, retrieval hit‑rate metrics, LLM self‑grading, and thresholding to refuse low‑confidence answers.

3. ***UI Frontend***: Minimal React client with chat history, source highlighting, and file upload progress.

4. ***Agent Extensibility***: Specialized research tools (ArXiv, Semantic Scholar), citation formatting, and query rewriting.
