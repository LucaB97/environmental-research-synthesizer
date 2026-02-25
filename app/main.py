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
from app.utils.evidence_analysis import aggregate_evidence, extract_sentence_paper_ids, compute_grounding_metrics
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
    
    meta = {
        "scope": {
            "decision": "in_scope"
        },
        "retrieval": {
            "retriever_info": {
                "strategy": "hybrid",
                "semantic": {
                    "embedding_backend": type(retriever.semantic_retriever.embedding_fn).__name__,
                    "embedding_model": getattr(retriever.semantic_retriever.embedding_fn, "model_name", None),
                    "faiss_index_type": type(retriever.semantic_retriever.index).__name__,
                    "chunks_requested": None,
                },
                "bm25": {
                    "enabled": hasattr(retriever, "bm25_retriever"),
                    "chunks_requested": None,
                }
            },
            "query_expansion": {
                "state": None,
                "num_queries": None
            },
            "candidate_pool_size": None,
            "retrieval_time_sec": None,
        },
        "profiling": {
            "model": relevance_profiler.model_name,
            "profiling_time_sec": None,
        },
        "errors": {
            "generation_error": None,
            "total_attempts": None,
            "last_error": None
        },
        "synthesis": {
            "synthesizer_info": {
                "llm_type": type(synthesizer.llm).__name__,
                "model": synthesizer.llm.model_name,
                "max_tokens": synthesizer.llm.max_tokens,
                "temperature": synthesizer.llm.temperature,
            },
            "synthesis_retry": {
                "attempted": None,
                "total_attempts": None,
                "retry_triggers": None
            },
            "chunks_selected": None,
            "synthesis_time_sec": None,
            "total_time_sec": None
        }
    }
    
    pipeline_status = "success"

    #
    # --- Zero-shot classification of scope---
    #
    query = request.question

    if not scope_classifier.is_in_scope(query, SCOPE_CLASSIFIER_PROMPT):
        pipeline_status = "out_of_scope"
        limitations = ["The question cannot be answered from the available sources"]
        meta["scope"] = "out_of_scope"
        confidence_profile = evaluate_confidence_profile(pipeline_status, reason="Early termination because the question is out of scope")
        return build_response(query, pipeline_status, limitations, meta=meta, confidence=confidence_profile)

    #
    # --- Retrieval ---
    #
    topk_faiss, topk_bm25 = request.topk_faiss, request.topk_bm25
    query_expansion = False
    expanded_query = None

    t0 = time.perf_counter()
    retrieved_chunks = retriever.search(query, topk_faiss=topk_faiss, topk_bm25=topk_bm25)
    retrieval_time = time.perf_counter() - t0
    
    meta["retrieval"]["retriever_info"]["semantic"]["chunks_requested"] = topk_faiss
    meta["retrieval"]["retriever_info"]["bm25"]["chunks_requested"] = topk_bm25
    meta["retrieval"]["candidate_pool_size"] = len(retrieved_chunks)
    meta["retrieval"]["query_expansion"]["state"] = query_expansion
    meta["retrieval"]["query_expansion"]["num_queries"] = 1
    meta["retrieval"]["retrieval_time_sec"] = round(retrieval_time, 3)
    
    if not retrieved_chunks:
        pipeline_status = "retrieval_failed"
        limitations = ["No documents could be retrieved for this question"]
        confidence_profile = evaluate_confidence_profile(pipeline_status, reason="Early termination because no evidence was retrieved")
        return build_response(query, pipeline_status, limitations, meta=meta, confidence=confidence_profile)

    #
    # --- Reranking & Profiling ---
    #
    t1 = time.perf_counter()
    reranked_chunks = relevance_profiler.rerank(query, retrieved_chunks)
    profiling_time = time.perf_counter() - t1

    meta["profiling"]["profiling_time_sec"] = round(profiling_time, 3)

    evidence_score, evidence_flags, evidence_meta, strong_chunks = evaluate_evidence_structure(reranked_chunks)

    #
    # --- Retrieval retry ---
    #
    if evidence_flags['absent'] or evidence_flags['isolated'] or evidence_flags['low_density']:
        
        expanded_query = query_expander.produce_expansion(query, QUERY_EXPANDER_PROMPT)
        queries = [query, expanded_query]
        query_expansion = True

        retrieved_chunks = []

        t0 = time.perf_counter()
        for q in queries:
            retrieved_chunks.extend(retriever.search(q, topk_faiss=topk_faiss, topk_bm25=topk_bm25))
        retrieval_time += time.perf_counter() - t0
        
        retrieved_chunks = deduplicate(retrieved_chunks)

        meta["retrieval"]["candidate_pool_size"] = len(retrieved_chunks)
        meta["retrieval"]["query_expansion"]["state"] = query_expansion
        meta["retrieval"]["query_expansion"]["num_queries"] = len(queries)
        meta["retrieval"]["retrieval_time_sec"] = round(retrieval_time, 3)

        if not retrieved_chunks:
            pipeline_status = "retrieval_failed"
            limitations = ["No documents could be retrieved for this question"]
            confidence_profile = evaluate_confidence_profile(pipeline_status, reason="Early termination because no evidence was retrieved")
            trace={"query_expansion": expanded_query}
            return build_response(query, pipeline_status, limitations, meta=meta, 
                                  confidence=confidence_profile, trace=trace)
        
        t1 = time.perf_counter()
        reranked_chunks = relevance_profiler.rerank(query, retrieved_chunks)
        profiling_time += time.perf_counter() - t1
        meta["profiling"]["profiling_time_sec"] = round(profiling_time, 3)

        evidence_score, evidence_flags, evidence_meta, strong_chunks = evaluate_evidence_structure(reranked_chunks)

    #
    # --- Early returns ---
    #

    if evidence_flags['absent']:  
        limitations=["The literature retrieved is topically related, but does not address this question directly"]
        confidence_profile = evaluate_confidence_profile(pipeline_status, evidence_score, evidence_flags, 
                                                         reason="Grounding score is absent because synthesis was not performed")
        trace={"query_expansion": expanded_query}
        return build_response(query, pipeline_status, limitations, meta=meta, 
                              evidence_structure=evidence_meta, confidence=confidence_profile, trace=trace)
        

    for chunk in strong_chunks:
        tmp_metadata = metadata.loc[chunk["paper_id"]]
        chunk['title'] = str(tmp_metadata['title'])
        chunk['authors'] = str(tmp_metadata['authors'])
        chunk['year'] = int(tmp_metadata['year'])
        chunk['journal'] = str(tmp_metadata['journal'])


    if evidence_flags['isolated']:        
        limitations=["The retrieved evidence is too narrow and context-specific to support synthesis across studies"]
        confidence_profile = evaluate_confidence_profile(pipeline_status, evidence_score, evidence_flags, 
                                                         reason="Grounding score is absent because synthesis was not performed")
        trace={
            "query_expansion": expanded_query,
            "strong_hit_chunks": strong_chunks            
            }
        return build_response(query, pipeline_status, limitations, meta=meta, 
                              evidence_structure=evidence_meta, confidence=confidence_profile, trace=trace)          

    #
    # --- Synthesis ---
    #    

    top_N = 15
    meta["synthesis"]["chunks_selected"] = top_N

    provided_chunks = reranked_chunks[:top_N]

    for c in provided_chunks:
        c_metadata = metadata.loc[c["paper_id"]]
        c['title'] = str(c_metadata['title'])
        c['authors'] = str(c_metadata['authors'])
        c['year'] = int(c_metadata['year'])
        c['journal'] = str(c_metadata['journal'])
        c['first_tag'] = (None if pd.isna(c_metadata['first_tag']) else str(c_metadata['first_tag']))
        c['second_tag'] = (None if pd.isna(c_metadata['second_tag']) else str(c_metadata['second_tag']))

    source_lookup = {
        c["chunk_id"]: c
        for c in provided_chunks
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
            synthesis_output = synthesizer.synthesize(query, provided_chunks, prompt)
        except ValueError as e:
            last_error = e
            logger.error("Synthesis failed after retries", exc_info=e)
            break  # hard failure → exit loop


        answer = synthesis_output["answer"]
        
        if not answer:
            synthesis_time = time.perf_counter() - t2
            total_time = time.perf_counter() - start_time

            limitations = synthesis_output.get("limitations") or ["No meaningful answer could be produced from the available literature"]

            meta["synthesis"]["synthesis_time_sec"] = round(synthesis_time, 3)
            meta["synthesis"]["total_time_sec"] = round(total_time, 3)

            # if attempt > 1:
            #     meta["synthesis"]["synthesis_retry"]["attempted"] = attempt>1
            #     meta["synthesis"]["synthesis_retry"]["total_attempts"] = attempt
            #     meta["synthesis"]["synthesis_retry"]["retry_triggers"] = retry_triggers

            confidence_profile = evaluate_confidence_profile(pipeline_status, evidence_score, evidence_flags, 
                                                             reason="Grounding score is absent because the synthesizer abstained from synthesis")
            trace={
            "query_expansion": expanded_query,
            "strong_hit_chunks": strong_chunks
            }
            return build_response(query, pipeline_status, limitations, meta=meta, 
                                  evidence_structure=evidence_meta, confidence=confidence_profile, trace=trace)   

        ## Evidence metrics
        sentence_papers = extract_sentence_paper_ids(synthesis_output["answer"], source_lookup)

        used_chunks_ids = {
            cid
            for sentence in synthesis_output["answer"]
            for cid in sentence.get("citations", [])
        }

        aggregation = aggregate_evidence(provided_chunks, used_chunks_ids)
        grounding_metrics = compute_grounding_metrics(aggregation, sentence_papers)

        grounding_score, grounding_flags = evaluate_grounding_quality(grounding_metrics)
        confidence_profile = evaluate_confidence_profile(pipeline_status, evidence_score, evidence_flags, grounding_score, grounding_flags)
        
        score = confidence_profile["grounding"]["score"]
        if score > best_score:
            best_output = synthesis_output
            best_sentence_papers = sentence_papers
            best_aggregation = aggregation
            best_grounding_metrics = grounding_metrics
            best_confidence = confidence_profile
            best_score = score

        # --- Retry decision ---
        if grounding_score < 0.5:
            retry_reason = determine_retry_reason(grounding_metrics, evidence_meta["distinct_strong_sources"])
            retry_triggers.append(retry_reason) 
        else:
            retry_reason = None             

        if grounding_metrics and retry_reason and attempt < max_attempts:
            logger.info(
                "Retrying synthesis due to weak grounding",
                extra={"grounding_metrics": grounding_metrics}
            )
            prompt = TASK_HEADER + RETRY_PROMPTS[retry_reason] + CORE_SYNTHESIS_INSTRUCTIONS
            continue
        else:
            break  # synthesis accepted

    synthesis_time = time.perf_counter() - t2
    total_time = time.perf_counter() - start_time
    meta["synthesis"]["synthesis_time_sec"] = round(synthesis_time, 3)
    meta["synthesis"]["total_time_sec"] = round(total_time, 3)
    
    # --- Failure fallback ---
    if last_error and not synthesis_output:
        pipeline_status = "generation_error"
        limitations=["The system was unable to generate a reliable answer this time. Please try again."]
        meta["errors"]["generation_error"] = True
        meta["errors"]["total_attempts"] = synthesizer.max_attempts
        meta["errors"]["last_error"] = last_error
        confidence_profile = evaluate_confidence_profile(pipeline_status, evidence_score, evidence_flags, 
                                                         reason="Grounding score is absent because the synthesis generation failed")
        return build_response(query, pipeline_status, limitations, meta=meta, 
                              evidence_structure=evidence_meta, confidence=confidence_profile)  

    #
    # --- Output preparation ---
    #
    meta["synthesis"]["synthesis_retry"]["attempted"] = attempt>1
    meta["synthesis"]["synthesis_retry"]["total_attempts"] = attempt
    meta["synthesis"]["synthesis_retry"]["retry_triggers"] = retry_triggers

    citation_index = build_citation_index(best_sentence_papers)
    sources = build_sources(citation_index, source_lookup)

    style = CitationStyle.NUMERIC
    resolved_answer = resolve_answer_citations(best_output["answer"], source_lookup, citation_index, FORMATTERS[style])
    resolved_answer = remove_citations_inside_text(resolved_answer) ## remove references from synthesis text

    trace={
        "query_expansion": expanded_query,
        "strong_hit_chunks": strong_chunks,
        "chunks_provided_to_synthesizer": best_aggregation["chunks"],
        "paper_stats": [
            {"paper_id": pid, **stats}
            for pid, stats in best_aggregation["paper_stats"].items()
        ]
    }

    return build_response(query, pipeline_status, limitations=best_output["limitations"], answer=[Sentence(**s) for s in resolved_answer], 
                          sources=sources, meta=meta, evidence_structure=evidence_meta, grounding_metrics=best_grounding_metrics,
                          confidence=best_confidence, trace=trace)