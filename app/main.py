import pandas as pd
import time
import logging
from fastapi import FastAPI, Request

from app.dependencies import load_system
from app.schemas import QueryRequest, QueryResponse, Sentence, Confidence
from app.response import build_response
from app.utils.synthesis_prompt import SCOPE_CLASSIFIER_PROMPT, TASK_HEADER, CORE_SYNTHESIS_INSTRUCTIONS, RETRY_PROMPTS
from app.utils.citations import build_citation_index, build_sources, CitationStyle, FORMATTERS, resolve_answer_citations, remove_citations_inside_text 
from app.utils.heuristics import determine_retry_reason
from app.utils.evidence_analysis import aggregate_evidence, extract_sentence_paper_ids, compute_evidence_metrics, get_debug_info
from app.utils.confidence import evaluate_evidence_structure, determine_grounding, compute_confidence


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
        # "index_loaded": (
        #     retriever is not None
        #     and hasattr(retriever, "index")
        #     and retriever.index is not None
        # ),
        # "index_size": retriever.index.ntotal if retriever else None,
        "scope_classifier_loaded": hasattr(req.app.state, "scope_classifier"),
        "retriever_loaded": hasattr(req.app.state, "retriever"),
        "relevance_profiler_loaded": hasattr(req.app.state, "relevance_profiler"),
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
    synthesizer = req.app.state.synthesizer
    
    pipeline_status = "success"

    #
    # --- Zero-shot classification of scope---
    #
    if not scope_classifier.is_in_scope(request.question, SCOPE_CLASSIFIER_PROMPT):
        pipeline_status = "out_of_scope"
        limitations = ["The question cannot be answered from the available sources."]
        meta = {"scope_decision": "out_of_scope"}
        return build_response(request.question, pipeline_status, limitations, meta=meta)

    #
    # --- Retrieval ---
    #
    t0 = time.perf_counter()
    
    retrieved_chunks = retriever.search(request.question, top_k_faiss=request.top_k_faiss, top_k_bm25=request.top_k_bm25)
    
    retrieval_time = time.perf_counter() - t0
    
    if not retrieved_chunks:
        pipeline_status = "retrieval_failed"
        limitations = ["No documents could be retrieved for this question."]
        meta={"top_k": request.top_k_faiss + request.top_k_bm25, "chunks_retrieved": 0}
        return build_response(request.question, pipeline_status, limitations, meta=meta)

    #
    # --- Reranking & Profiling ---
    #
    t1 = time.perf_counter()
    reranked_chunks = relevance_profiler.rerank(request.question, retrieved_chunks)
    profiling_time = time.perf_counter() - t1

    evidence_score, evidence_flags, evidence_meta = evaluate_evidence_structure(reranked_chunks)

    if evidence_flags['absent']:  
        limitations=["The literature retrieved is topically related, but does not address this question directly."]
        meta={
                "chunks_requested": request.top_k_faiss + request.top_k_bm25,
                "chunks_retrieved": len(retrieved_chunks),
                "evidence_metrics": evidence_meta
            }
        confidence = Confidence(structure_score=evidence_score, grounding_score=None, region_label="Not_applicable", 
                                explanation="Grounding score is absent because synthesis was not performed")
        return build_response(request.question, pipeline_status, limitations, meta=meta, confidence=confidence)
        
    elif evidence_flags['isolated']:
        strong_hits = evidence_meta["strong_hit_chunks"]
        for chunk in strong_hits:
            tmp_metadata = metadata.loc[chunk["paper_id"]]
            chunk['title'] = str(tmp_metadata['title'])
            chunk['authors'] = str(tmp_metadata['authors'])
            chunk['year'] = int(tmp_metadata['year'])
            chunk['journal'] = str(tmp_metadata['journal'])
        
        limitations=["The retrieved evidence is too narrow and context-specific to support synthesis across studies."]
        meta={
                "chunks_requested": request.top_k_faiss + request.top_k_bm25,
                "chunks_retrieved": len(retrieved_chunks),
                "evidence_metrics": evidence_meta.pop("strong_hit_chunks")
            }
        debug={"chunks": strong_hits}
        confidence = Confidence(structure_score=evidence_score, grounding_score=None, region_label="Not_applicable", 
                                explanation="Grounding score is absent because synthesis was not performed")
        return build_response(request.question, pipeline_status, limitations, meta=meta, confidence=confidence, debug=debug)          

    ### RETRY FOR MONO_SOURCE_STRONG & LOW DENSITY here 

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

    max_attempts = 3
    attempt = 0
    synthesis_output = None
    best_output, best_score = None, -1
    last_error = None
    retry_triggers = set()

    prompt = TASK_HEADER + CORE_SYNTHESIS_INSTRUCTIONS

    t2 = time.perf_counter()

    while attempt < max_attempts:
        attempt += 1
        
        try:
            synthesis_output = synthesizer.synthesize(
                request.question,
                relevant_chunks,
                prompt
            )
        except ValueError as e:
            last_error = e
            logger.error("Synthesis failed after retries", exc_info=e)
            break  # hard failure → exit loop


        answer = synthesis_output["answer"]
        
        if not answer:
            grounding_quality = "not_answered"
            limitations = synthesis_output.get("limitations") or ["No meaningful answer could be produced from the available literature."]
            meta={
                "chunks_requested": request.top_k_faiss + request.top_k_bm25,
                "chunks_retrieved": len(retrieved_chunks),
                "relevance_metrics": evidence_meta
            }
            confidence = Confidence(structure_score=evidence_score, grounding_score=None, region_label="Not_applicable", 
                                explanation="Grounding score is absent because the synthesizer abstained from synthesis.")
            return build_response(request.question, pipeline_status, limitations, meta=meta, confidence=confidence)   

        ## Evidence metrics
        sentence_papers = extract_sentence_paper_ids(synthesis_output["answer"], source_lookup)

        used_chunks_ids = {
            cid
            for sentence in synthesis_output["answer"]
            for cid in sentence.get("citations", [])
        }

        aggregation = aggregate_evidence(relevant_chunks, used_chunks_ids)
        metrics = compute_evidence_metrics(aggregation, sentence_papers)

        grounding_score, grounding_quality = determine_grounding(metrics)
        score, label, explanation = compute_confidence(pipeline_status, evidence_structure, grounding_quality, grounding_score)
        
        if score > best_score:
            best_output = synthesis_output
            best_sentence_papers = sentence_papers
            best_aggregation = aggregation
            best_metrics = metrics
            best_grounding_quality = grounding_quality
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
        pipeline_status = "generation_error"
        limitations=["The system was unable to generate a reliable answer this time. Please try again."]
        meta={
            "generation_error": True,
            "last_error": str(last_error) if last_error else None
        }
        confidence = Confidence(structure_score=evidence_score, grounding_score=None, region_label="Not_applicable", 
                                explanation="Grounding score is absent because the synthesis generation failed.")
        return build_response(request.question, pipeline_status, limitations, meta=meta, confidence=confidence)  

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
        question=request.question,
        pipeline_status = pipeline_status,
        evidence_structure = evidence_structure,
        grounding_quality = best_grounding_quality,
        answer=[Sentence(**s) for s in resolved_answer],
        limitations=best_output["limitations"],
        sources=sources,
        meta={
            "chunks_requested": request.top_k_faiss + request.top_k_bm25,
            "chunks_retrieved": len(retrieved_chunks),
            "relevance_metrics": profile["metrics"],
            "retrieval_time_sec": round(retrieval_time, 3),
            "profiling_time_sec": round(profiling_time, 3),
            "synthesis_time_sec": round(synthesis_time, 3),
            "total_time_sec": round(total_time, 3),
            "retry": {
                "attempted": attempt>1,
                "num_attempts": attempt,
                "trigger": list(retry_triggers)
            }
        },
        evidence_metrics=best_metrics,
        confidence = Confidence(
            score=best_score,
            label=best_label,
            explanation=best_explanation
        ),
        debug=debug
    )