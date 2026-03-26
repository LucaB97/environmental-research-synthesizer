import os
import pandas as pd
import json
from dotenv import load_dotenv

from initialization.config import DEFAULT_CONFIG
from initialization.pipeline import initialize_system

from services.embeddings import OpenAIEmbedding, HFEmbedding
from services.indexing import load_faiss
from services.llm_clients import OpenAIClient, HFClient

from pipeline.retrieval.retriever import SemanticRetriever, BM25Retriever, HybridRetriever
from pipeline.retrieval.reranker import RelevanceProfiler
from pipeline.llm.scope_classification import QueryScopeClassifier
from pipeline.llm.query_expansion import QueryExpander
from pipeline.llm.generation import ResearchSynthesisEngine
from pipeline.orchestration import RAGPipeline


def load_system(app):
    
    load_dotenv()
    
    profile = os.getenv("SYNTH_PROFILE", "public")

    if profile not in ["public", "gpu"]:
        raise ValueError(f"Invalid profile: {profile}")

    config = DEFAULT_CONFIG

    artifacts = initialize_system(config)

    metadata_path = artifacts["metadata_path"]
    chunks_path = artifacts["chunks_path"]
    index_path = artifacts["index_path"]
    params_path = artifacts["params_path"]

    with open(chunks_path, encoding="utf-8") as f:
        chunks = json.load(f)

    with open(params_path, encoding="utf-8") as f:
        params = json.load(f)

    index = load_faiss(index_path)

    if config.embedding == "hf":
        embedding_fn = HFEmbedding()
    else:
        embedding_fn = OpenAIEmbedding()

    semantic_retriever = SemanticRetriever(index, chunks, embedding_fn)
    bm25_retriever = BM25Retriever(chunks)

    # ---- LLM selection ----
    if profile == "public":
        llm = OpenAIClient()
    elif profile == "gpu":
        llm = HFClient(
            "mistralai/Mistral-7B-Instruct-v0.2",
            load_in_4bit=True
        )

    # ---- Core components ----
    scope_classifier = QueryScopeClassifier(llm)
    retriever = HybridRetriever(semantic_retriever, bm25_retriever)
    relevance_profiler = RelevanceProfiler()
    query_expander = QueryExpander(llm)
    synthesizer = ResearchSynthesisEngine(llm, max_attempts=3)

    # ---- Attach to app ----
    app.state.pipeline = RAGPipeline(
        metadata=pd.read_csv(metadata_path).set_index("paper_id"),
        scope_classifier=scope_classifier,
        retriever=retriever,
        relevance_profiler=relevance_profiler,
        tuned_parameters=params,
        query_expander=query_expander,
        synthesizer=synthesizer
    )