# Production RAG Pipeline & Local SLM Gateway

## Description
This project implements a production-ready Retrieval-Augmented Generation (RAG) system entirely orchestrated within a Dockerized environment. It solves the challenge of securely ingesting complex offline documentation (PDF/DOCX) and surfacing accurate, context-aware answers in real-time. By leveraging a local vector database and self-hosted Large Language Models (LLMs), the system prevents external data leakage while maintaining robust observability across retrieval and generation execution paths.

## Architecture
The system follows a microservice architecture built on Docker Compose.
1. **Ingestion Layer:** Raw documents (`data/raw`) are parsed via PyMuPDF4LLM, chunked via semantic boundaries, encoded using the Jina API, and stored alongside rich metadata into the Qdrant vector database.
2. **API & Routing (FastAPI):** A high-performance ASGI server handles incoming HTTP requests for both data ingestion (`/api/v1/ingest`) and user queries (`/api/v1/query`).
3. **Retrieval & Generation:** Incoming queries are contextually optimized before executing a semantic similarity search in Qdrant. The retrieved context is formatted into a strict prompt and passed to a local LLM via Ollama endpoints.
4. **Observability Layer:** The Langfuse Python SDK instruments the execution flow, transmitting traces asynchronously to a self-hosted Langfuse stack (backed by PostgreSQL and ClickHouse) for latency, token-usage, and quality auditing.

## Tech Stack
| Component | Technology |
| :--- | :--- |
| **API Framework** | Python 3.11, FastAPI, Uvicorn |
| **Vector Database** | Qdrant |
| **LLM Provider** | Ollama (Local), Groq (Evaluation) |
| **Embeddings** | Jina Embeddings API |
| **Data Orchestration** | PyMuPDF4LLM, Langchain Text Splitters |
| **Observability** | Langfuse |
| **Infrastructure** | Docker, Docker Compose |
| **Evaluation** | RAGAS (Faithfulness, Relevancy, Precision) |

## Features
- **Offline Data Privacy:** Local embeddings and LLM generation guarantee that sensitive corporate documents never leave the internal network.
- **Asynchronous Processing:** Built on `httpx` and `asyncio` to ensure non-blocking concurrent request handling across the API layer.
- **Multi-Stage Dockerization:** Secured, lightweight container builds executing under a non-root user (`aiuser`) to minimize the attack surface.
- **Full-Stack Observability:** Automatic Langfuse tracing of the complete RAG waterfall (Query -> Retrieval -> LLM Generation) for deep debugging.
- **Automated Quality Gates:** Integrated RAGAS evaluation scripts assessing Faithfulness, Answer Relevancy, and Context Precision recursively.

## Project Structure
```text
.
├── config
│   ├── prompts.yaml
│   └── settings.yaml
├── data
│   └── raw
├── docker
│   ├── docker-compose.yml
│   └── Dockerfile
├── docs
│   └── screenshots
├── eval
│   ├── completed_dataset.json
│   ├── dataset.json
│   └── results
│       └── ragas_scores.json
├── scripts
│   ├── evaluate.py
│   └── ingest.py
├── src
│   ├── api
│   │   ├── routes
│   │   │   ├── health.py
│   │   │   ├── ingest.py
│   │   │   └── query.py
│   │   └── schemas
│   │       └── models.py
│   ├── core
│   │   ├── config.py
│   │   └── logger.py
│   ├── evaluation
│   │   └── ragas_evaluator.py
│   ├── generation
│   │   ├── llm_client.py
│   │   └── response_builder.py
│   ├── ingestion
│   │   ├── document_loader.py
│   │   ├── embedder.py
│   │   ├── indexer.py
│   │   └── splitter.py
│   ├── main.py
│   └── retrieval
│       ├── query_processor.py
│       └── retriever.py
├── .env.example
└── requirements.txt
```

## Quickstart

**1. Clone the repository and configure the environment**
```bash
git clone <repository_url>
cd <repository_directory>
cp .env.example .env
```
*Note: Update the `.env` file with your specific API keys (Jina, Groq) and desired credentials.*

**2. Boot the infrastructure**
Start all 8 required containers (API, Qdrant, Langfuse, Postgres, Minio, Redis, Clickhouse).
```bash
docker compose --env-file .env -f docker/docker-compose.yml up -d --build
```

**3. Configure Observability**
Navigate to the local Langfuse UI (`http://localhost:3000`). Create a new project and generate your API keys. Copy the Public Key, Secret Key, and Host into your `.env` file.

**4. Ingest documents**
Place target PDFs in `data/raw`, then execute the ingestion script from inside the API container.
```bash
docker compose -f docker/docker-compose.yml exec api python -m scripts.ingest
```

**5. Execute a query**
Interact via the Swagger UI (`http://localhost:8000/docs`) or via cURL:
```bash
curl -X POST "http://localhost:8000/api/v1/query" \
     -H "Content-Type: application/json" \
     -d '{"question": "What is the primary policy?"}'
```
