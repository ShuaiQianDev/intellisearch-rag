# IntelliSearch — RAG-Powered Document QA System

A Retrieval-Augmented Generation (RAG) system for natural language question answering over technical documents. Combines hybrid retrieval (dense vector search + BM25 sparse retrieval) with Cross-Encoder reranking to deliver high-accuracy answers, packaged as a low-latency FastAPI microservice.

## Results

| Metric | Baseline (dense retrieval only) | Optimized (hybrid retrieval + reranking) |
|---|---|---|
| Top-5 context retrieval accuracy | 64% | 92% |
| P95 query latency (100 concurrent users) | — | < 200ms |

> Numbers above are produced by the evaluation scripts in `eval/`, measured against a self-built set of query–document relevance annotations.

## Architecture

```
User query
   │
   ▼
┌──────────────┐     ┌──────────────┐
│ Dense Search  │     │  BM25 Search  │   ← run in parallel
│  (FAISS)      │     │  (sparse)     │
└──────┬───────┘     └──────┬───────┘
       │                    │
       └────────┬───────────┘
                 ▼
        RRF Fusion Ranking
                 │
                 ▼
        Cross-Encoder Reranking
                 │
                 ▼
        LLM Answer Generation (with context)
```

Index building (offline): document loading → chunking → embedding → FAISS index.

## Tech Stack

- **Language**: Python 3.12
- **RAG Orchestration**: LangChain
- **Vector Search**: FAISS
- **Sparse Retrieval**: BM25 (rank_bm25)
- **Embedding / Reranking Models**: HuggingFace `sentence-transformers`
- **API Service**: FastAPI + Uvicorn (async)
- **Deployment**: Docker

## Project Structure

```
intellisearch/
├── data/               # Raw documents
├── src/
│   ├── ingest.py       # Document loading + chunking
│   ├── embed.py        # Embedding + FAISS index building
│   ├── retrieve.py      # Hybrid retrieval + reranking
│   ├── generate.py     # Prompt construction + LLM call
│   └── main.py          # FastAPI service entrypoint
├── eval/                # Retrieval evaluation scripts and labeled data
├── tests/
├── Dockerfile
├── requirements.txt
└── README.md
```

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/intellisearch-rag.git
cd intellisearch-rag

# 2. Create a virtual environment and install dependencies
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Configure environment variables (LLM API key, etc.)
cp .env.example .env
# edit .env and fill in your API key

# 4. Build the index (run once, and again whenever documents change)
python src/ingest.py
python src/embed.py

# 5. Start the service
uvicorn src.main:app --reload
```

Once running, visit `http://localhost:8000/docs` for the interactive API documentation.

### Run with Docker

```bash
docker build -t intellisearch .
docker run -p 8000:8000 --env-file .env intellisearch
```

## API Example

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What index types does FAISS support?"}'
```

## Evaluation Methodology

The scripts in `eval/`:
1. Load a hand-labeled mapping of queries to relevant document IDs
2. Run all queries through both "dense retrieval only" and "hybrid retrieval + reranking" pipelines
3. Compute Top-5 hit rate (Recall@5) and output a comparison report

```bash
python eval/run_eval.py
```

## Roadmap

- [x] Basic dense vector retrieval
- [x] BM25 hybrid retrieval
- [x] Cross-encoder reranking
- [x] FastAPI service layer
- [ ] Incremental index updates (without full rebuild)
- [ ] Support additional document formats (Confluence / Notion exports)
- [ ] Retrieval result explainability (highlight matched spans)

## License

MIT
