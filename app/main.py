import time
import logging
from fastapi import FastAPI, Request, HTTPException

from app.dependencies import load_system
from app.schemas import QueryRequest, QueryResponse, Sentence
from app.utils.citations import remove_citations_inside_text, resolve_answer_citations, build_source_entry
from app.utils.heuristics import determine_reason
from app.utils.debug_info import get_debug_info

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
            meta={
                "top_k": request.top_k,
                "chunks_retrieved": 0
                },
        )

    source_lookup = {
        c["chunk_id"]: c
        for c in retrieved_chunks
    }

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
    
    ## Backend reason enforcement
    synthesis_output["reason"] = determine_reason(synthesis_output, source_lookup)

    ## Get answer with citations in the [Authors, Year] format 
    resolved_answer, sentence_citations = resolve_answer_citations(
        synthesis_output["answer"],
        source_lookup
    )

    ## Remove any references included in the synthesis text
    resolved_answer = remove_citations_inside_text(resolved_answer)

    ## Build list of sources
    if synthesis_output["reason"] == "out_of_scope":
        sources = []
    else:
        cited_paper_ids = set().union(*sentence_citations)
        sources = [build_source_entry(pid, source_lookup) for pid in cited_paper_ids]

    ## Add debug info for UI
    used_chunks_ids = {
        cid
        for sentence in synthesis_output["answer"]
        for cid in sentence.get("citations", [])
    }
    debug = get_debug_info(retrieved_chunks, used_chunks_ids, sentence_citations)


    return QueryResponse(
        question=request.question,
        reason=synthesis_output["reason"],
        answer=[Sentence(**s) for s in resolved_answer],
        limitations=synthesis_output["limitations"],
        sources=sources,
        meta={
            "top_k": request.top_k,
            "chunks_retrieved": len(retrieved_chunks),
            "retrieval_time_sec": round(retrieval_time, 3),
            "synthesis_time_sec": round(synthesis_time, 3),
            "total_time_sec": round(total_time, 3),
        },
        debug=debug
    )

