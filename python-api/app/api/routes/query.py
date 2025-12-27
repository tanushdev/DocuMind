"""Query endpoint for document Q&A."""
import hashlib
from fastapi import APIRouter, HTTPException

from app.config import get_settings
from app.models import (
    QueryRequest,
    QueryResponse,
    SourceDocument,
    LatencyBreakdown
)
from app.core import (
    EmbeddingServiceFactory,
    RerankerFactory,
    ContextAssemblerFactory,
    LLMServiceFactory
)
from app.services import VectorServiceFactory, RedisClientFactory
from app.utils import RequestTimer, get_metrics_collector

router = APIRouter(prefix="/query", tags=["query"])


@router.post("", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Query documents and get an AI-generated answer.
    
    This endpoint orchestrates the full RAG pipeline:
    1. Check cache for previous identical query
    2. Generate query embedding
    3. Vector search for relevant chunks
    4. Re-rank results with cross-encoder
    5. Assemble context within token limits
    6. Generate answer with LLM
    7. Cache and return results
    
    Args:
        request: Query request with question and options
        
    Returns:
        Answer with source citations and latency breakdown
    """
    settings = get_settings()
    timer = RequestTimer()
    metrics = get_metrics_collector()
    cache = RedisClientFactory.get_cache_service()
    
    # Generate cache key
    query_hash = hashlib.sha256(
        f"{request.query}:{request.top_k}:{request.document_ids}".encode()
    ).hexdigest()[:16]
    
    # Check cache
    cached_result = await cache.get_query_result(query_hash)
    if cached_result:
        await metrics.record_cache_hit()
        return QueryResponse(
            answer=cached_result["answer"],
            sources=[SourceDocument(**s) for s in cached_result["sources"]],
            latency=LatencyBreakdown(**cached_result["latency"]),
            cached=True
        )
    
    await metrics.record_cache_miss()
    
    try:
        # 1. Generate query embedding
        with timer.stage("embedding"):
            embedding_service = EmbeddingServiceFactory.get_instance()
            query_embedding = await embedding_service.embed_single(request.query)
        
        # 2. Vector search
        with timer.stage("search"):
            vector_client = VectorServiceFactory.get_instance()
            search_results = await vector_client.search(
                embedding=query_embedding,
                top_k=settings.top_k_retrieval,  # Get more for re-ranking
                algorithm="hnsw"
            )
        
        if not search_results:
            raise HTTPException(
                status_code=404,
                detail="No relevant documents found. Please upload documents first."
            )
        
        # Filter by document IDs if specified
        if request.document_ids:
            search_results = [
                r for r in search_results
                if r.document_id in request.document_ids
            ]
        
        # 3. Re-rank with cross-encoder
        with timer.stage("rerank"):
            reranker = RerankerFactory.get_instance()
            documents = [
                {
                    "id": r.id,
                    "text": r.text,
                    "metadata": {
                        "document_id": r.document_id,
                        "chunk_index": r.chunk_index,
                        "page_number": r.page_number
                    }
                }
                for r in search_results
            ]
            ranked_docs = await reranker.rerank(
                query=request.query,
                documents=documents,
                top_k=request.top_k
            )
        
        # 4. Assemble context
        with timer.stage("context"):
            context_assembler = ContextAssemblerFactory.get_instance()
            assembled = context_assembler.assemble(
                query=request.query,
                ranked_docs=ranked_docs
            )
        
        # 5. Generate LLM response
        with timer.stage("llm"):
            llm = await LLMServiceFactory.get_available()
            if llm is None:
                # Fallback: return context without LLM
                answer = _generate_fallback_answer(request.query, ranked_docs)
            else:
                response = await llm.generate(
                    prompt=assembled.prompt,
                    max_tokens=500,
                    temperature=0.7
                )
                answer = response.text
        
        # Build sources
        sources = [
            SourceDocument(
                document_id=chunk.document_id,
                chunk_text=chunk.text[:500],  # Truncate for response
                chunk_index=chunk.chunk_index,
                page_number=chunk.page_number,
                relevance_score=chunk.relevance_score
            )
            for chunk in assembled.chunks
        ]
        
        # Build latency breakdown
        latency = LatencyBreakdown(
            embedding_ms=timer.stages.get("embedding", 0),
            search_ms=timer.stages.get("search", 0),
            rerank_ms=timer.stages.get("rerank", 0),
            context_ms=timer.stages.get("context", 0),
            llm_ms=timer.stages.get("llm", 0),
            total_ms=timer.total_ms
        )
        
        # Record metrics
        await timer.record_all(metrics)
        
        # Cache result
        result_dict = {
            "answer": answer,
            "sources": [s.model_dump() for s in sources],
            "latency": latency.model_dump()
        }
        await cache.set_query_result(query_hash, result_dict)
        
        return QueryResponse(
            answer=answer,
            sources=sources,
            latency=latency,
            cached=False
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Query processing failed: {str(e)}"
        )


def _generate_fallback_answer(query: str, ranked_docs) -> str:
    """Generate a simple answer when LLM is unavailable."""
    if not ranked_docs:
        return "No relevant information found in the documents."
    
    # Combine top chunks as the answer
    answer_parts = ["Based on the documents, here's what I found:\n"]
    for i, doc in enumerate(ranked_docs[:3], 1):
        answer_parts.append(f"\n[Source {i}]: {doc.text[:300]}...")
    
    answer_parts.append("\n\n(Note: LLM is not available. This is a fallback response showing relevant excerpts.)")
    
    return "".join(answer_parts)
