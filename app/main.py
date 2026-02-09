import pandas as pd
import time
import logging
from fastapi import FastAPI, Request

from app.dependencies import load_system
from app.schemas import QueryRequest, QueryResponse, Sentence, Confidence
from app.utils.synthesis_prompt import SCOPE_CLASSIFIER_PROMPT, TASK_HEADER, CORE_SYNTHESIS_INSTRUCTIONS, RETRY_PROMPTS
from app.utils.citations import build_citation_index, build_sources, CitationStyle, FORMATTERS, resolve_answer_citations, remove_citations_inside_text 
from app.utils.heuristics import determine_reason, determine_retry_reason
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
        "scope_classifier_loaded": hasattr(req.app.state, "scope_classifier"),
        "index_loaded": (
            retriever is not None
            and hasattr(retriever, "index")
            and retriever.index is not None
        ),
        "index_size": retriever.index.ntotal if retriever else None,
        "retriever_loaded": retriever is not None,
        "relevance_gate_loaded": hasattr(req.app.state, "relevance_gate"),
        "synthesizer_loaded": hasattr(req.app.state, "synthesizer"),
    }


@app.post("/query", response_model=QueryResponse)
def query_endpoint(request: QueryRequest, req: Request):
    
    logger = logging.getLogger(__name__)
    start_time = time.perf_counter()
    metadata = req.app.state.metadata
    scope_classifier = req.app.state.scope_classifier
    retriever = req.app.state.retriever
    relevance_gate = req.app.state.relevance_gate
    synthesizer = req.app.state.synthesizer
    
    #
    # --- Zero-shot classification of scope---
    #
    if not scope_classifier.is_in_scope(request.question, SCOPE_CLASSIFIER_PROMPT):
        return QueryResponse(
            question=request.question,
            reason="out_of_scope",
            answer=[],
            limitations=["The available literature does not address the question."],
            sources=[],
        )

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
            reason="retrieval_failed",
            answer=[],
            limitations=[
                "The system was unable to retrieve information to be used for synthesis."
            ],
            sources=[],
            meta={
                "top_k": request.top_k,
                "chunks_retrieved": 0
                },
        )

    relevant, num_relevant_chunks = relevance_gate.is_relevant(request.question, retrieved_chunks)

    if not relevant:

        if num_relevant_chunks == 0:
            return QueryResponse(
                question=request.question,
                reason="absent_evidence",
                answer=[],
                limitations=[],
                sources=[],
                meta={
                    "relevance_gate": {
                        "method": "cross_encoder",
                        "passed": False
                    }
                }
            )
        
        else:
            relevant_chunks = relevance_gate.debug(request.question, retrieved_chunks, show_only_relevant=True)
            for c in relevant_chunks:
                c_metadata = metadata.loc[c["paper_id"]]
                c['title'] = str(c_metadata['title'])
                c['authors'] = str(c_metadata['authors'])
                c['year'] = int(c_metadata['year'])
                c['journal'] = str(c_metadata['journal'])
            
            debug = {"chunks": relevant_chunks}

            return QueryResponse(
                question=request.question,
                reason="isolated_evidence",
                answer=[],
                limitations=[
                    "The retrieved literature did not contain sufficiently relevant evidence to support a reliable answer to the question.",
                    "This topic may be discussed in the literature at the level of specific technologies, projects, or local contexts rather than in general terms."
                    ],
                sources=[],
                meta={
                    "relevance_gate": {
                        "method": "cross_encoder",
                        "passed": False
                    }
                }, 
                debug = debug
            )
            


    #
    # --- Synthesis ---
    #    

    for c in retrieved_chunks:
        c_metadata = metadata.loc[c["paper_id"]]
        c['title'] = str(c_metadata['title'])
        c['authors'] = str(c_metadata['authors'])
        c['year'] = int(c_metadata['year'])
        c['journal'] = str(c_metadata['journal'])
        c['first_tag'] = (None if pd.isna(c_metadata['first_tag']) else str(c_metadata['first_tag']))
        c['second_tag'] = (None if pd.isna(c_metadata['second_tag']) else str(c_metadata['second_tag']))

    source_lookup = {
        c["chunk_id"]: c
        for c in retrieved_chunks
    }

    max_attempts = 2
    attempt = 0
    synthesis_output = None
    best_output, best_score = None, -1
    last_error = None
    retry_triggers = set()

    prompt = TASK_HEADER + CORE_SYNTHESIS_INSTRUCTIONS

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


        answer = synthesis_output["answer"]
        
        if not answer:
            limitations = synthesis_output.get("limitations") or [
                "No meaningful answer could be produced from the available literature."
                ]
            
            return QueryResponse(
                question=request.question,
                reason="insufficient_evidence",
                answer=[],
                limitations=limitations,
                sources=[],
                evidence_metrics=None,
                confidence={
                    "score": 0.0,
                    "label": "Low",
                    "explanation": []
                }
            )

        ## Evidence metrics
        sentence_papers = extract_sentence_paper_ids(synthesis_output["answer"], source_lookup)

        used_chunks_ids = {
            cid
            for sentence in synthesis_output["answer"]
            for cid in sentence.get("citations", [])
        }

        aggregation = aggregate_evidence(retrieved_chunks, used_chunks_ids)
        metrics = compute_evidence_metrics(aggregation, sentence_papers)

        failure_reason = determine_reason(synthesis_output, source_lookup)
        score, label, explanation = compute_confidence(metrics, failure_reason)
        
        if score > best_score:
            best_failure_reason = failure_reason
            best_output = synthesis_output
            best_sentence_papers = sentence_papers
            best_aggregation = aggregation
            best_metrics = metrics
            best_score = score
            best_label = label
            best_explanation = explanation

        # --- Retry decision ---
        retry_reason = determine_retry_reason(metrics)
        
        if retry_reason:
            retry_triggers.add(retry_reason)  

        if metrics and retry_reason and attempt < max_attempts:
            logger.info(
                "Retrying synthesis due to weak evidence metrics",
                extra={"metrics": metrics}
            )
            prompt = TASK_HEADER + RETRY_PROMPTS[retry_reason] + CORE_SYNTHESIS_INSTRUCTIONS
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

    ## Build list of sources
    citation_index = build_citation_index(best_sentence_papers)
    sources = build_sources(citation_index, source_lookup)

    debug = get_debug_info(best_aggregation)
    confidence = Confidence(score=best_score, label=best_label, explanation=best_explanation)
    
    #
    # --- Output preparation ---
    #
    
    style = CitationStyle.NUMERIC
    resolved_answer = resolve_answer_citations(best_output["answer"], source_lookup, citation_index, FORMATTERS[style])
    
    resolved_answer = remove_citations_inside_text(resolved_answer) ## remove references from synthesis text

    return QueryResponse(
        question=request.question,
        reason=best_failure_reason,
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
                "trigger": list(retry_triggers)
            }
        },
        evidence_metrics=best_metrics,
        confidence = confidence,
        debug=debug
    )