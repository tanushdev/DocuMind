# DocuMind 🧠📄

> **Production-Grade AI Document Intelligence System**

A sophisticated Retrieval-Augmented Generation (RAG) system demonstrating elite-tier systems and ML engineering. Built with **Python + Go**, featuring custom vector search, distributed architecture, and performance-first design.

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![Go](https://img.shields.io/badge/Go-1.21+-00ADD8.svg)](https://golang.org)
[![Redis](https://img.shields.io/badge/Redis-7+-red.svg)](https://redis.io)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED.svg)](https://docker.com)

---

## 🎯 Project Overview

DocuMind allows users to:
- **Upload documents** (PDF / TXT)
- **Ask natural-language questions**
- **Receive AI-powered answers with cited sources**

### Key Differentiators

| Aspect | What We Built | Why It Matters |
|--------|---------------|----------------|
| **Vector Search** | Custom HNSW in Go | Demonstrates algorithmic depth, not just "using FAISS" |
| **RAG Pipeline** | Manual implementation | No LangChain = full understanding and control |
| **Architecture** | True microservices | Go + Python with async boundaries |
| **Performance** | Caching + metrics | Production-ready observability |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                   CLIENT                                     │
│                           (React / HTML UI)                                  │
│                              Port 3000                                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          PYTHON ORCHESTRATOR                                 │
│                           (FastAPI - Port 8000)                              │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐│
│  │  Document   │ │  Embedding  │ │ Re-ranking  │ │    Context Assembly     ││
│  │  Chunking   │ │  Generation │ │  (Cross-    │ │    + LLM Integration    ││
│  │  (Manual)   │ │  (SBERT)    │ │  Encoder)   │ │    (Ollama/HF)          ││
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────────────────┘│
│                                      │                                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │              Background Task Processing (asyncio)                       ││
│  └─────────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────┘
          │                                │                        │
          ▼                                ▼                        ▼
┌──────────────────┐          ┌──────────────────┐       ┌──────────────────┐
│  GO VECTOR SVC   │          │     REDIS        │       │   LLM SERVICE    │
│  (Port 8001)     │          │  (Port 6379)     │       │   (Ollama)       │
│  ┌────────────┐  │          │  ┌────────────┐  │       │   (Port 11434)   │
│  │  HNSW      │  │          │  │  Embedding │  │       └──────────────────┘
│  │  Index     │  │          │  │  Cache     │  │
│  ├────────────┤  │          │  ├────────────┤  │
│  │  Brute     │  │          │  │  Query     │  │
│  │  Force     │  │          │  │  Cache     │  │
│  ├────────────┤  │          │  ├────────────┤  │
│  │  Cosine    │  │          │  │  Task      │  │
│  │  Similarity│  │          │  │  Status    │  │
│  └────────────┘  │          │  └────────────┘  │
└──────────────────┘          └──────────────────┘
```

---

## 🔄 Request Flow

### Query Processing Pipeline

```
User Query: "What are the key findings?"
         │
         ▼
┌─────────────────────────────────────┐
│ 1. CACHE CHECK                      │ ◄── Cache hit? Return immediately
│    Key: hash(query)                 │     ~5ms
└─────────────────────────────────────┘
         │ Cache Miss
         ▼
┌─────────────────────────────────────┐
│ 2. EMBEDDING GENERATION             │ ◄── sentence-transformers
│    Model: all-MiniLM-L6-v2          │     ~20-50ms
│    Dimensions: 384                  │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│ 3. VECTOR SEARCH (Go Service)       │ ◄── HTTP POST /search
│    Algorithm: HNSW                  │     O(log n) complexity
│    Returns: top-20 candidates       │     ~10-50ms
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│ 4. CROSS-ENCODER RE-RANKING         │ ◄── ms-marco-MiniLM-L-6-v2
│    Input: (query, chunk) pairs      │     Scores each pair
│    Output: top-5 reordered          │     ~50-200ms
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│ 5. CONTEXT ASSEMBLY                 │ ◄── Token budget: 2048
│    - Fit chunks to limit            │     Add source citations
│    - Build prompt template          │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│ 6. LLM GENERATION                   │ ◄── Ollama (llama2/mistral)
│    Streaming response               │     ~500ms-2s
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│ 7. CACHE & RETURN                   │
│    - Cache result (1h TTL)          │
│    - Return with sources            │
│    - Log metrics                    │
└─────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **API Server** | Python + FastAPI | Orchestration, ML inference |
| **Vector Search** | Go + custom HNSW | High-performance similarity search |
| **Cache** | Redis | Embeddings, queries, task status |
| **Embeddings** | sentence-transformers | Text → vectors |
| **Re-ranking** | Cross-encoder | Improve retrieval quality |
| **LLM** | Ollama / HuggingFace | Answer generation |
| **Frontend** | HTML/CSS/JS | User interface |
| **Container** | Docker Compose | Deployment |

---

## 📁 Project Structure

```
DocuMind/
├── README.md                    # This file
├── docker-compose.yml           # Container orchestration
├── .env.example                 # Environment template
│
├── go-vector-service/           # Go Vector Search Service
│   ├── cmd/server/main.go       # Entry point
│   ├── internal/
│   │   ├── api/                 # HTTP handlers & router
│   │   │   ├── handlers.go
│   │   │   └── router.go
│   │   └── index/               # Search algorithms
│   │       ├── similarity.go    # Cosine/L2 distance
│   │       ├── bruteforce.go    # O(n) baseline
│   │       └── hnsw.go          # O(log n) ANN
│   ├── pkg/types/vector.go      # Data types
│   ├── Dockerfile
│   └── go.mod
│
├── python-api/                  # Python FastAPI Backend
│   ├── app/
│   │   ├── main.py              # FastAPI app
│   │   ├── config.py            # Settings
│   │   ├── api/routes/          # Endpoints
│   │   │   ├── documents.py     # Upload/status
│   │   │   ├── query.py         # Q&A
│   │   │   └── health.py        # Health/metrics
│   │   ├── core/                # ML components
│   │   │   ├── chunking.py      # Text splitting
│   │   │   ├── embeddings.py    # Vector generation
│   │   │   ├── reranking.py     # Cross-encoder
│   │   │   ├── context.py       # Prompt assembly
│   │   │   └── llm.py           # LLM clients
│   │   ├── services/            # External services
│   │   │   ├── vector_client.py # Go service client
│   │   │   ├── redis_client.py  # Cache operations
│   │   │   └── document_processor.py
│   │   ├── models/              # Pydantic schemas
│   │   └── utils/metrics.py     # Performance logging
│   ├── requirements.txt
│   └── Dockerfile
│
└── frontend/                    # Web UI
    ├── index.html
    ├── styles.css
    ├── app.js
    ├── nginx.conf
    └── Dockerfile
```

---

## 🚀 Quick Start

### Prerequisites

- **Docker** & **Docker Compose**
- **Ollama** (optional, for LLM) - [Install Ollama](https://ollama.ai)
- **8GB+ RAM** recommended

### 1. Clone & Configure

```bash
git clone https://github.com/yourusername/documind.git
cd documind

# Copy environment template
cp .env.example .env
```

### 2. Start Ollama (Optional)

```bash
# Install Ollama, then:
ollama run llama2
# Or use mistral for better quality:
ollama run mistral
```

### 3. Launch with Docker Compose

```bash
# Build and start all services
docker-compose up --build

# Or run in background
docker-compose up -d --build
```

### 4. Access the Application

| Service | URL |
|---------|-----|
| **Frontend** | http://localhost:3000 |
| **API Docs** | http://localhost:8000/docs |
| **API Health** | http://localhost:8000/health |
| **Vector Service** | http://localhost:8001/health |

---

## 📊 API Reference

### Document Upload

```bash
# Upload a PDF
curl -X POST http://localhost:8000/api/documents/upload \
  -F "file=@document.pdf"

# Response
{
  "task_id": "abc123",
  "status": "processing"
}
```

### Check Processing Status

```bash
curl http://localhost:8000/api/documents/status/abc123

# Response
{
  "task_id": "abc123",
  "status": "completed",
  "progress": 1.0,
  "document_id": "xyz789",
  "num_chunks": 42
}
```

### Query Documents

```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the main conclusions?",
    "top_k": 5
  }'

# Response
{
  "answer": "Based on the documents, the main conclusions are...",
  "sources": [
    {
      "document_id": "xyz789",
      "chunk_text": "...",
      "relevance_score": 0.89
    }
  ],
  "latency": {
    "embedding_ms": 25.4,
    "search_ms": 8.2,
    "rerank_ms": 142.1,
    "llm_ms": 892.3,
    "total_ms": 1068.0
  }
}
```

---

## ⚡ Performance Targets

| Stage | p50 Target | p95 Target |
|-------|------------|------------|
| Query Embedding | < 20ms | < 50ms |
| Vector Search (10K vectors) | < 20ms | < 50ms |
| Cross-encoder Rerank | < 100ms | < 250ms |
| LLM Generation | < 1s | < 3s |
| **End-to-end (cached)** | < 50ms | < 100ms |
| **End-to-end (uncached)** | < 2s | < 5s |

---

## 🧪 Running Tests

### Go Vector Service

```bash
cd go-vector-service

# Run tests
go test ./... -v

# Run benchmarks
go test -bench=. ./internal/index/ -benchmem
```

### Python API

```bash
cd python-api

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v
```

---

## 🏛️ Design Decisions & Tradeoffs

### Why Custom HNSW Instead of FAISS?

| Custom HNSW | FAISS |
|-------------|-------|
| ✅ Deep algorithmic understanding | ❌ Black box |
| ✅ "I built it" on resume | ❌ "I configured it" |
| ✅ Full control & customization | ❌ Limited flexibility |
| ⚠️ Less optimized | ✅ Highly optimized |

**Decision**: Learning value and interview differentiation outweigh raw performance for a demo system.

### Why Go for Vector Search?

- **Goroutines**: Natural fit for parallel search
- **Memory efficiency**: Better control vs Python
- **Learning opportunity**: You're learning Go
- **Service boundary**: Clean separation enforces good architecture

### Why No LangChain?

| Without LangChain | With LangChain |
|-------------------|----------------|
| 50+ lines of code | 5 lines of code |
| Full visibility | Magic abstractions |
| "I understand RAG" | "I used a framework" |
| Easy to customize | Framework lock-in |

---

## 🚫 What We Didn't Build (Intentionally)

| Feature | Reason to Skip |
|---------|----------------|
| Multi-node sharding | Overkill for demo, adds complexity |
| GPU support | CPU-only keeps it accessible |
| Authentication | Not the focus of the demo |
| Kubernetes | Docker Compose is sufficient |
| Fine-tuned models | Pre-trained works well enough |
| Real-time sync | Batch processing is fine |

---

## 📈 Resume Bullet Points

After completing this project, add these to your resume:

> **DocuMind | Production-Grade Document Intelligence System**
> - Designed and implemented a **RAG system** achieving <50ms p50 latency for cached queries using custom vector search, Redis caching, and async processing
> - Built a **custom HNSW graph-based ANN search** in Go from scratch, demonstrating O(log n) vs O(n) brute-force complexity
> - Implemented **microservices architecture** with Python (FastAPI) and Go, featuring goroutine-based concurrency, cross-encoder re-ranking, and token-aware context assembly
> - Developed **complete RAG pipeline without high-level frameworks** (no LangChain), including manual chunking, embedding generation, and prompt engineering

### Skills Demonstrated

✅ Systems Design & Distributed Architecture  
✅ Algorithm Implementation (HNSW, Similarity Search)  
✅ Performance Engineering (Caching, Concurrency)  
✅ ML Engineering (Embeddings, Cross-encoders, RAG)  
✅ Multi-language Development (Python + Go)  
✅ Observability (Latency Tracking, p50/p95 Metrics)

---

## 🔧 Development

### Local Development (without Docker)

```bash
# Terminal 1: Redis
docker run -p 6379:6379 redis:7-alpine

# Terminal 2: Go Vector Service
cd go-vector-service
go run ./cmd/server

# Terminal 3: Python API
cd python-api
pip install -r requirements.txt
uvicorn app.main:app --reload

# Terminal 4: Frontend (optional)
cd frontend
python -m http.server 3000
```

### Environment Variables

See `.env.example` for all configuration options.

---

## 📄 License

MIT License - feel free to use this for learning, portfolios, and interviews.

---

## 🙏 Acknowledgments

- [Sentence-Transformers](https://www.sbert.net/) for embedding models
- [HNSW Paper](https://arxiv.org/abs/1603.09320) by Malkov & Yashunin
- [Ollama](https://ollama.ai) for local LLM inference

---

**Built with ❤️ for learning and demonstrating systems engineering skills.**
