import pandas as pd
import time
import logging
from fastapi import FastAPI, Request

from app.dependencies import load_system
from app.schemas import QueryRequest, QueryResponse, Sentence
from app.response import build_response
from app.helpers import deduplicate
from app.utils.prompt import SCOPE_CLASSIFIER_PROMPT, QUERY_EXPANDER_PROMPT, TASK_HEADER, CORE_SYNTHESIS_INSTRUCTIONS, RETRY_PROMPTS
from app.utils.citations import build_citation_index, build_sources, CitationStyle, FORMATTERS, resolve_answer_citations, remove_citations_inside_text 
from app.utils.heuristics import determine_retry_reason
from app.utils.evidence_analysis import aggregate_evidence, extract_sentence_paper_ids, compute_evidence_metrics, get_debug_info
from app.utils.confidence import evaluate_evidence_structure, evaluate_grounding_quality, evaluate_confidence_profile


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
        "metadata_loaded": hasattr(req.app.state, "metadata"),
        "scope_classifier_loaded": hasattr(req.app.state, "scope_classifier"),
        "retriever_loaded": hasattr(req.app.state, "retriever"),
        "index_loaded": (
            retriever is not None
            and hasattr(retriever, "index")
            and retriever.index is not None
        ),
        "index_size": retriever.index.ntotal if retriever else None,
        "relevance_profiler_loaded": hasattr(req.app.state, "relevance_profiler"),
        "query_expander_loaded": hasattr(req.app.state, "query_expander"),
        "synthesizer_loaded": hasattr(req.app.state, "synthesizer"),
    }


@app.post("/query", response_model=QueryResponse)
def query_endpoint(request: QueryRequest, req: Request):
    
    logger = logging.getLogger(__name__)
    start_time = time.perf_counter()
    metadata = req.app.state.metadata
    scope_classifier = req.app.state.scope_classifier
    retriever = req.app.state.retriever
    relevance_profiler = req.app.state.relevance_profiler
    query_expander = req.app.state.query_expander
    synthesizer = req.app.state.synthesizer
    
    pipeline_status = "success"
    retrieval_retry = False

    #
    # --- Zero-shot classification of scope---
    #
    query = request.question

    if not scope_classifier.is_in_scope(query, SCOPE_CLASSIFIER_PROMPT):
        pipeline_status = "out_of_scope"
        limitations = ["The question cannot be answered from the available sources"]
        meta = {"scope_decision": "out_of_scope"}
        confidence_profile = evaluate_confidence_profile(pipeline_status, reason="Early termination because the question is out of scope")
        return build_response(query, pipeline_status, limitations, meta=meta, confidence=confidence_profile)

    #
    # --- Retrieval ---
    #
    topk_faiss, topk_bm25 = request.topk_faiss, request.topk_bm25
    expanded_query = None

    t0 = time.perf_counter()
    retrieved_chunks = retriever.search(query, topk_faiss=topk_faiss, topk_bm25=topk_bm25)
    retrieval_time = time.perf_counter() - t0
    
    if not retrieved_chunks:
        pipeline_status = "retrieval_failed"
        limitations = ["No documents could be retrieved for this question"]
        meta={"topk": topk_faiss + topk_bm25, "chunks_retrieved": 0}
        return build_response(query, pipeline_status, limitations, meta=meta)

    #
    # --- Reranking & Profiling ---
    #
    t1 = time.perf_counter()
    reranked_chunks = relevance_profiler.rerank(query, retrieved_chunks)
    profiling_time = time.perf_counter() - t1

    evidence_score, evidence_flags, evidence_meta, strong_chunks = evaluate_evidence_structure(reranked_chunks)

    #
    # --- Retrieval retry ---
    #
    if evidence_flags['absent'] or evidence_flags['isolated'] or evidence_flags['low_density']:
        
        retrieval_retry = True
        expanded_query = query_expander.produce_expansion(query, QUERY_EXPANDER_PROMPT)
        queries = [query, expanded_query]
        # topk_faiss, topk_bm25 = int(1.5 * topk_faiss), int(1.5 * topk_bm25)  #no increase in topk for now
        retrieved_chunks = []

        t0 = time.perf_counter()
        for q in queries:
            retrieved_chunks.extend(retriever.search(q, topk_faiss=topk_faiss, topk_bm25=topk_bm25))
        retrieval_time += time.perf_counter() - t0
        
        if not retrieved_chunks:
            pipeline_status = "retrieval_failed"
            limitations = ["No documents could be retrieved for this question"]
            meta={"topk": topk_faiss + topk_bm25, "chunks_retrieved": 0, "retrieval_retry": retrieval_retry}
            debug={"expanded_query": expanded_query}
            return build_response(query, pipeline_status, limitations, meta=meta, debug=debug)
        
        retrieved_chunks = deduplicate(retrieved_chunks)

        t1 = time.perf_counter()
        reranked_chunks = relevance_profiler.rerank(query, retrieved_chunks)
        profiling_time += time.perf_counter() - t1

        evidence_score, evidence_flags, evidence_meta, strong_chunks = evaluate_evidence_structure(reranked_chunks)

    #
    # --- Early returns ---
    #

    if evidence_flags['absent']:  
        limitations=["The literature retrieved is topically related, but does not address this question directly"]
        meta={
                "chunks_requested": topk_faiss + topk_bm25,
                "chunks_retrieved": len(retrieved_chunks),
                "retrieval_retry": retrieval_retry,
                "evidence_metrics": evidence_meta
            }
        confidence_profile = evaluate_confidence_profile(pipeline_status, evidence_score, evidence_flags, 
                                                         reason="Grounding score is absent because synthesis was not performed")
        debug={"expanded_query": expanded_query}
        return build_response(query, pipeline_status, limitations, meta=meta, confidence=confidence_profile, debug=debug)
        

    for chunk in strong_chunks:
        tmp_metadata = metadata.loc[chunk["paper_id"]]
        chunk['title'] = str(tmp_metadata['title'])
        chunk['authors'] = str(tmp_metadata['authors'])
        chunk['year'] = int(tmp_metadata['year'])
        chunk['journal'] = str(tmp_metadata['journal'])


    if evidence_flags['isolated']:        
        limitations=["The retrieved evidence is too narrow and context-specific to support synthesis across studies"]
        meta={
                "chunks_requested": request.topk_faiss + request.topk_bm25,
                "chunks_retrieved": len(retrieved_chunks),
                "retrieval_retry": retrieval_retry,
                "evidence_metrics": evidence_meta.pop("strong_hit_chunks")
            }
        confidence_profile = evaluate_confidence_profile(pipeline_status, evidence_score, evidence_flags, 
                                                         reason="Grounding score is absent because synthesis was not performed")
        
        debug={
            "relevant_passages": strong_chunks,
            "expanded_query": expanded_query
            }
        return build_response(query, pipeline_status, limitations, meta=meta, confidence=confidence_profile, debug=debug)          

    #
    # --- Synthesis ---
    #    

    top_N = 15
    relevant_chunks = reranked_chunks[:top_N]

    for c in relevant_chunks:
        c_metadata = metadata.loc[c["paper_id"]]
        c['title'] = str(c_metadata['title'])
        c['authors'] = str(c_metadata['authors'])
        c['year'] = int(c_metadata['year'])
        c['journal'] = str(c_metadata['journal'])
        c['first_tag'] = (None if pd.isna(c_metadata['first_tag']) else str(c_metadata['first_tag']))
        c['second_tag'] = (None if pd.isna(c_metadata['second_tag']) else str(c_metadata['second_tag']))

    source_lookup = {
        c["chunk_id"]: c
        for c in relevant_chunks
    }

    max_attempts = 2
    attempt = 0
    synthesis_output = None
    best_output, best_score = None, -1
    last_error = None
    retry_triggers = []

    prompt = TASK_HEADER + CORE_SYNTHESIS_INSTRUCTIONS

    t2 = time.perf_counter()

    while attempt < max_attempts:
        attempt += 1
        
        try:
            synthesis_output = synthesizer.synthesize(
                query,
                relevant_chunks,
                prompt
            )
        except ValueError as e:
            last_error = e
            logger.error("Synthesis failed after retries", exc_info=e)
            break  # hard failure → exit loop


        answer = synthesis_output["answer"]
        
        if not answer:
            limitations = synthesis_output.get("limitations") or ["No meaningful answer could be produced from the available literature."]
            meta={
                "chunks_requested": request.topk_faiss + request.topk_bm25,
                "chunks_retrieved": len(retrieved_chunks),
                "retrieval_retry": retrieval_retry,
                "relevance_metrics": evidence_meta
            }
            confidence_profile = evaluate_confidence_profile(pipeline_status, evidence_score, evidence_flags, 
                                                             reason="Grounding score is absent because the synthesizer abstained from synthesis")
            debug={
            "relevant_passages": strong_chunks,
            "expanded_query": expanded_query
            }
            return build_response(query, pipeline_status, limitations, meta=meta, confidence=confidence_profile, debug=debug)   

        ## Evidence metrics
        sentence_papers = extract_sentence_paper_ids(synthesis_output["answer"], source_lookup)

        used_chunks_ids = {
            cid
            for sentence in synthesis_output["answer"]
            for cid in sentence.get("citations", [])
        }

        aggregation = aggregate_evidence(relevant_chunks, used_chunks_ids)
        metrics = compute_evidence_metrics(aggregation, sentence_papers)

        grounding_score, grounding_flags = evaluate_grounding_quality(metrics)
        confidence_profile = evaluate_confidence_profile(pipeline_status, evidence_score, evidence_flags, grounding_score, grounding_flags)
        
        score = confidence_profile["grounding"]["score"]
        if score > best_score:
            best_output = synthesis_output
            best_sentence_papers = sentence_papers
            best_aggregation = aggregation
            best_metrics = metrics
            best_confidence = confidence_profile
            best_score = score

        # --- Retry decision ---
        if grounding_score < 0.5:
            retry_reason = determine_retry_reason(metrics, evidence_meta["distinct_strong_sources"])
            retry_triggers.append(retry_reason) 
        else:
            retry_reason = None             

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
        pipeline_status = "generation_error"
        limitations=["The system was unable to generate a reliable answer this time. Please try again."]
        meta={
            "generation_error": True,
            "last_error": str(last_error) if last_error else None
        }
        confidence_profile = evaluate_confidence_profile(pipeline_status, evidence_score, evidence_flags, 
                                                         reason="Grounding score is absent because the synthesis generation failed")
        return build_response(query, pipeline_status, limitations, meta=meta, confidence=confidence_profile)  

    synthesis_time = time.perf_counter() - t2
    total_time = time.perf_counter() - start_time

    ## Build list of sources
    citation_index = build_citation_index(best_sentence_papers)
    sources = build_sources(citation_index, source_lookup)
    debug = get_debug_info(best_aggregation)

    #
    # --- Output preparation ---
    #
    
    style = CitationStyle.NUMERIC
    resolved_answer = resolve_answer_citations(best_output["answer"], source_lookup, citation_index, FORMATTERS[style])

    resolved_answer = remove_citations_inside_text(resolved_answer) ## remove references from synthesis text

    return QueryResponse(
        question=query,
        pipeline_status = pipeline_status,
        answer=[Sentence(**s) for s in resolved_answer],
        limitations=best_output["limitations"],
        sources=sources,
        meta={
            "chunks_requested": request.topk_faiss + request.topk_bm25,
            "chunks_retrieved": len(retrieved_chunks),
            "retrieval_retry": retrieval_retry,
            "relevance_metrics": evidence_meta,
            "retrieval_time_sec": round(retrieval_time, 3),
            "profiling_time_sec": round(profiling_time, 3),
            "synthesis_time_sec": round(synthesis_time, 3),
            "total_time_sec": round(total_time, 3),
            "synthesis_retry": {
                "attempted": attempt>1,
                "total_attempts": attempt,
                "retry_trigger": retry_triggers
            }
        },
        evidence_metrics=best_metrics,
        confidence = best_confidence,
        debug=debug
    )