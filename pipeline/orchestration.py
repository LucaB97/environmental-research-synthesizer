import pandas as pd
import time
import logging

from utils.prompt import SCOPE_CLASSIFIER_PROMPT, QUERY_EXPANDER_PROMPT, TASK_HEADER, CORE_SYNTHESIS_INSTRUCTIONS, RETRY_PROMPTS
from utils.citations import build_citation_index, build_sources, CitationStyle, FORMATTERS, resolve_answer_citations, remove_citations_inside_text 
from utils.chunking import deduplicate

from .evaluation.evidence_analysis import aggregate_evidence, extract_sentence_paper_ids, compute_grounding_metrics
from .evaluation.confidence import evaluate_semantic_alignment, evaluate_evidence_structure, evaluate_grounding_quality, evaluate_confidence_profile
from .evaluation.retry_policy import need_retry_semantic, reason_retry_grounding
from .postprocessing.limitations import assign_limitations
from .postprocessing.response_builder import build_query_response

from schemas.request import QueryRequest
from schemas.response import Sentence

class RAGPipeline:

    def __init__(
        self,
        metadata,
        scope_classifier,
        retriever,
        relevance_profiler,
        tuned_parameters,
        query_expander,
        synthesizer
    ):
        self.metadata = metadata
        self.scope_classifier = scope_classifier
        self.retriever = retriever
        self.relevance_profiler = relevance_profiler
        self.params = tuned_parameters
        self.query_expander = query_expander
        self.synthesizer = synthesizer


    def initialize_output_meta(self):

        return {
            "scope": {
                "decision": "in_scope"
            },
            "retrieval": {
                "retriever_info": {
                    "strategy": "hybrid",
                    "semantic": {
                        "embedding_backend": type(self.retriever.semantic_retriever.embedding_fn).__name__,
                        "embedding_model": getattr(self.retriever.semantic_retriever.embedding_fn, "model_name", None),
                        "faiss_index_type": type(self.retriever.semantic_retriever.index).__name__,
                        "chunks_requested": None,
                    },
                    "bm25": {
                        "enabled": hasattr(self.retriever, "bm25_retriever"),
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
                "model": self.relevance_profiler.model_name,
                "profiling_time_sec": None,
            },
            "errors": {
                "generation_error": None,
                "total_attempts": None,
                "last_error": None
            },
            "synthesis": {
                "synthesizer_info": {
                    "llm_type": type(self.synthesizer.llm).__name__,
                    "model": self.synthesizer.llm.model_name,
                    "max_tokens": self.synthesizer.llm.max_tokens,
                    "temperature": self.synthesizer.llm.temperature,
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
    

    def run(self, request: QueryRequest):
        logger = logging.getLogger(__name__)
        start_time = time.perf_counter()
        
        meta = self.initialize_output_meta()
        
        pipeline_status = "success"
        top_N = 15
        #
        # --- Zero-shot classification of scope---
        #
        query = request.question

        if not self.scope_classifier.is_in_scope(query, SCOPE_CLASSIFIER_PROMPT):
            pipeline_status = "out_of_scope"
            limitations = ["The question cannot be answered from the available sources"]
            meta["scope"] = "out_of_scope"
            confidence_profile = evaluate_confidence_profile(pipeline_status, reason="Early termination because the question is out of scope")
            return build_query_response(query, pipeline_status, limitations, meta=meta, confidence=confidence_profile)

        #
        # --- Retrieval ---
        #
        topk_faiss, topk_bm25 = request.topk_faiss, request.topk_bm25
        query_expansion = False
        queries = None

        t0 = time.perf_counter()
        retrieved_chunks = self.retriever.search(query, topk_faiss=topk_faiss, topk_bm25=topk_bm25)
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
            return build_query_response(query, pipeline_status, limitations, meta=meta, confidence=confidence_profile)

        #
        # --- Reranking & Profiling ---
        #
        t1 = time.perf_counter()
        reranked_chunks = self.relevance_profiler.rerank(query, retrieved_chunks)
        profiling_time = time.perf_counter() - t1

        meta["profiling"]["profiling_time_sec"] = round(profiling_time, 3)

        semantic_alignment = evaluate_semantic_alignment(reranked_chunks, self.params, top_N)
        
        evidence_structure, evidence_flags, evidence_meta = evaluate_evidence_structure(reranked_chunks[:top_N], self.params)

        #
        # --- Retrieval retry ---
        #

        if need_retry_semantic(semantic_alignment, evidence_flags):
            
            if not query_expansion:

                query_expansion = True
                expanded_query = self.query_expander.produce_expansion(query, QUERY_EXPANDER_PROMPT)
                queries = [query, expanded_query]
                retrieved_chunks = []

                t0 = time.perf_counter()
                for q in queries:
                    retrieved_chunks.extend(self.retriever.search(q, topk_faiss=topk_faiss, topk_bm25=topk_bm25))
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
                    trace={"query_expansion": queries}
                    return build_query_response(query, pipeline_status, limitations, meta=meta, confidence=confidence_profile, trace=trace)
                
                t1 = time.perf_counter()
                reranked_chunks = self.relevance_profiler.rerank(query, retrieved_chunks)
                profiling_time += time.perf_counter() - t1
                meta["profiling"]["profiling_time_sec"] = round(profiling_time, 3)

                semantic_alignment = evaluate_semantic_alignment(reranked_chunks, self.params, top_N)
        
                evidence_structure, evidence_flags, evidence_meta = evaluate_evidence_structure(reranked_chunks[:top_N], self.params)

        #
        # --- Early returns ---
        #

        if semantic_alignment < 0.25 or evidence_flags["absent"]:
            limitations=assign_limitations(semantic_alignment, absent=evidence_flags["absent"])
            confidence_profile = evaluate_confidence_profile(pipeline_status, semantic_alignment, evidence_structure, evidence_meta, evidence_flags, 
                                                             reason="Grounding score is absent because synthesis was not performed")
            trace={
                "query_expansion": queries,
                }
            return build_query_response(query, pipeline_status, limitations, meta=meta, evidence_structure=evidence_meta, confidence=confidence_profile, trace=trace)   

        #
        # --- Synthesis ---
        #    
        meta["synthesis"]["chunks_selected"] = top_N

        provided_chunks = reranked_chunks[:top_N]

        for c in provided_chunks:
            c_metadata = self.metadata.loc[c["paper_id"]]
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

        max_attempts = 3
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
                synthesis_output = self.synthesizer.synthesize(query, provided_chunks, prompt)
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

                confidence_profile = evaluate_confidence_profile(pipeline_status, semantic_alignment, evidence_structure, evidence_meta, evidence_flags, 
                                                                reason="Grounding score is absent because the synthesizer abstained from synthesis")
                trace={
                "query_expansion": queries,
                }
                return build_query_response(query, pipeline_status, limitations, meta=meta, 
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
            confidence_profile = evaluate_confidence_profile(pipeline_status, semantic_alignment, evidence_structure, evidence_meta, evidence_flags, 
                                                             grounding_score, grounding_metrics, grounding_flags)
            
            score = confidence_profile["grounding"]["score"]
            if score > best_score:
                best_output = synthesis_output
                best_sentence_papers = sentence_papers
                best_aggregation = aggregation
                best_grounding_metrics = grounding_metrics
                best_confidence = confidence_profile
                best_score = score

            # --- Retry decision ---
            if grounding_score < 0.5 and attempt < max_attempts:
                retry_reason = reason_retry_grounding(grounding_metrics)
                retry_triggers.append(retry_reason) 
                print(f"GROUNDING SCORE: {grounding_score}\nGROUNDING FLAGS: {grounding_flags}\nGROUNDING METRICS: {grounding_metrics}")
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
            meta["errors"]["total_attempts"] = self.synthesizer.max_attempts
            meta["errors"]["last_error"] = str(last_error)
            confidence_profile = evaluate_confidence_profile(pipeline_status, semantic_alignment, evidence_structure, evidence_meta, evidence_flags, 
                                                            reason="Grounding score is absent because the synthesis generation failed")
            return build_query_response(query, pipeline_status, limitations, meta=meta, 
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
            "query_expansion": queries,
            "chunks_provided_to_synthesizer": best_aggregation["chunks"],
            "paper_stats": [
                {"paper_id": pid, **stats}
                for pid, stats in best_aggregation["paper_stats"].items()
            ]
        }

        return build_query_response(query, pipeline_status, limitations=best_output["limitations"], answer=[Sentence(**s) for s in resolved_answer], 
                            sources=sources, meta=meta, evidence_structure=evidence_meta, grounding_metrics=best_grounding_metrics,
                            confidence=best_confidence, trace=trace)