import time
import logging
from fastapi import FastAPI, Request, HTTPException

from app.dependencies import load_system
from app.schemas import QueryRequest, QueryResponse, AnswerBullet
from app.utils.citations import extract_citations, build_sources_from_citations


app = FastAPI(
    title="Environmental Research Synthesizer",
    description="Semantic retrieval + evidence-based synthesis from academic literature",
    version="0.1.0"
)


@app.on_event("startup")
def startup_event():
    load_system(app)


@app.get("/health")
def health_check(req: Request):
    retriever = getattr(req.app.state, "retriever", None)

    return {
        "status": "ok",
        "retriever_loaded": retriever is not None,
        "index_loaded": (
            retriever is not None
            and hasattr(retriever, "index")
            and retriever.index is not None
        ),
        "index_size": retriever.index.ntotal if retriever else None,
        "synthesizer_loaded": hasattr(req.app.state, "synthesizer"),
    }


@app.post("/query", response_model=QueryResponse)
def query_endpoint(request: QueryRequest, req: Request):
    
    start_time = time.perf_counter()
    retriever = req.app.state.retriever
    synthesizer = req.app.state.synthesizer

    #
    # --- Retrieval ---
    #
    t0 = time.perf_counter()
    
    retrieved_chunks = retriever.search(
        request.question,
        top_k=request.top_k
    )
    
    retrieval_time = time.perf_counter() - t0
    
    if not retrieved_chunks:
        return QueryResponse(
            question=request.question,
            in_scope=False,
            answer=[],
            limitations=["No relevant sources were retrieved for this question."],
            sources=[],
            meta={"chunks_retrieved": 0},
        )

    #
    # --- Synthesis ---
    #
    t1 = time.perf_counter()
    
    try:
        synthesis_output = synthesizer.synthesize(
            request.question,
            retrieved_chunks
        )
    except ValueError as e:
        logger = logging.getLogger(__name__)
        logger.error(
            "Synthesis failed due to invalid LLM output",
            exc_info=e
        )
        raise HTTPException(
            status_code=502,
            detail="LLM returned invalid structured output"
        )
    
    synthesis_time = time.perf_counter() - t1
    total_time = time.perf_counter() - start_time

    #
    # --- Output preparation ---
    #

    if not synthesis_output["in_scope"]:
        sources = []
    else:
        cited_refs = extract_citations(synthesis_output["answer"])
        sources = build_sources_from_citations(retrieved_chunks, cited_refs)
        # sources = [Source(**s) for s in build_sources_from_citations(...)]


    return QueryResponse(
        question=request.question,
        in_scope=synthesis_output["in_scope"],
        answer=[AnswerBullet(**b) for b in synthesis_output["answer"]],
        limitations=synthesis_output["limitations"],
        sources=sources,
        meta={
            "chunks_retrieved": len(retrieved_chunks),
            "retrieval_time_sec": round(retrieval_time, 3),
            "synthesis_time_sec": round(synthesis_time, 3),
            "total_time_sec": round(total_time, 3),
        }
    )
