import time
import logging
from fastapi import FastAPI, Request

from app.dependencies import load_system
from app.schemas import QueryRequest, QueryResponse, Sentence, Confidence
from app.utils.synthesis_prompt import BASIC_SYNTHESIS_PROMPT, RETRY_SYNTHESIS_PROMPT
from app.utils.citations import remove_citations_inside_text, resolve_answer_citations, build_source_entry
from app.utils.heuristics import determine_reason, should_retry
from app.utils.evidence_analysis import aggregate_evidence, extract_sentence_paper_ids, compute_evidence_metrics, get_debug_info
from app.utils.confidence import compute_confidence

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
    
    logger = logging.getLogger(__name__)
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
    
    max_attempts = 2
    attempt = 0
    best_output, best_score = None, -1
    
    sources = []
    best_metrics = None
    debug = {}
    confidence = None
    
    prompt = BASIC_SYNTHESIS_PROMPT
    last_error = None
    
    t1 = time.perf_counter()

    while attempt < max_attempts:
        attempt += 1
        
        try:
            synthesis_output = synthesizer.synthesize(
                request.question,
                retrieved_chunks,
                prompt
            )
        except ValueError as e:
            last_error = e
            logger.error("Synthesis failed after retries", exc_info=e)
            break  # hard failure → exit loop
        

        ## Backend reason enforcement
        synthesis_output["reason"] = determine_reason(synthesis_output, source_lookup)

        if synthesis_output["reason"] == "out_of_scope":
            best_output = synthesis_output
            break


        ## Evidence metrics
        sentence_papers = extract_sentence_paper_ids(synthesis_output["answer"], source_lookup)

        used_chunks_ids = {
            cid
            for sentence in synthesis_output["answer"]
            for cid in sentence.get("citations", [])
        }

        aggregation = aggregate_evidence(retrieved_chunks, used_chunks_ids)
        metrics = compute_evidence_metrics(aggregation, sentence_papers)

        score, label, explanation = compute_confidence(metrics, synthesis_output["reason"])
        
        if score > best_score:
            best_score = score
            best_label = label
            best_explanation = explanation
            best_output = synthesis_output
            best_sentence_papers = sentence_papers
            best_aggregation = aggregation
            best_metrics = metrics

        # --- Retry decision ---
        if should_retry(metrics) and attempt < max_attempts:
            logger.info(
                "Retrying synthesis due to weak evidence metrics",
                extra={"metrics": metrics}
            )
            prompt = RETRY_SYNTHESIS_PROMPT
            continue
        else:
            break  # synthesis accepted


    # --- Failure fallback ---
    if last_error and not synthesis_output:
        return QueryResponse(
            question=request.question,
            reason="generation_failed",
            answer=[],
            limitations=[
                "The system was unable to generate a stable, well-supported synthesis."
            ],
            sources=[],
        )

    synthesis_time = time.perf_counter() - t1
    total_time = time.perf_counter() - start_time

    #
    # --- Output preparation ---
    #

    ## Use citations in the [Authors, Year] format 
    resolved_answer = resolve_answer_citations(best_output["answer"], source_lookup)
    
    ## Remove references from synthesis text
    resolved_answer = remove_citations_inside_text(resolved_answer)

    ## Build list of sources
    if best_output["reason"] != "out_of_scope":
        cited_paper_ids = set().union(*best_sentence_papers)
        sources = [build_source_entry(pid, source_lookup) for pid in cited_paper_ids]
        debug = get_debug_info(best_aggregation)
        confidence = Confidence(score=best_score, label=best_label, explanation=best_explanation)


    return QueryResponse(
        question=request.question,
        reason=best_output["reason"],
        answer=[Sentence(**s) for s in resolved_answer],
        limitations=best_output["limitations"],
        sources=sources,
        meta={
            "top_k": request.top_k,
            "chunks_retrieved": len(retrieved_chunks),
            "retrieval_time_sec": round(retrieval_time, 3),
            "synthesis_time_sec": round(synthesis_time, 3),
            "total_time_sec": round(total_time, 3),
            "retry": {
                "attempted": attempt>1,
                "num_attempts": attempt,
                "trigger": "low_source_diversity"
            }
        },
        evidence_metrics=best_metrics,
        debug=debug, 
        confidence = confidence
    )