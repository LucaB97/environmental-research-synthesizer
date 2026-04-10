import json
import numpy as np

from services.embeddings import HFEmbedding, OpenAIEmbedding
from services.indexing import load_faiss
from pipeline.retrieval.retriever import SemanticRetriever, BM25Retriever, HybridRetriever
from pipeline.retrieval.reranker import RelevanceProfiler
from pipeline.evaluation.confidence import semantic_norm, compute_contribution


QUERIES = [
    # --- General / high-recall (anchors) ---
    "Renewable energy adoption barriers",
    "Community response to wind energy",
    "Solar panel adoption factors",
    "Energy poverty and policy",
    "Public perception of renewable energy",
    "Community engagement in renewable energy projects",

    # --- Realistic user queries (core calibration set) ---
    "Social acceptance of community wind projects",
    "Impact of policy changes on energy literacy programs",
    "Participation of local groups in renewable energy initiatives",
    "Influence of social networks on adoption of solar panels",
    "Energy poverty considerations in renewable energy transitions",
    "Challenges in promoting energy literacy",
    "Impact of local regulations on renewable energy projects",
    "Volunteer involvement in energy transition awareness campaigns",

    "Household affordability and adoption of solar energy",
    "Community engagement strategies in renewable energy projects",
    "Public attitudes toward wind farms",
    "Energy literacy programs for energy efficiency",
    "Distributional effects of subsidies for residential solar panels",
    "Trust in local authorities and renewable energy adoption",
    "Equity in decision-making for renewable energy projects",
    "Economic effects of community-owned renewable energy initiatives",
    "Community engagement in microgrid projects",
    "Gender dynamics in renewable energy participation",
    "Impact of communication strategies on renewable energy awareness",

    # --- Moderately complex (keep some structure) ---
    "Participatory decision-making and social acceptance of wind projects",
    "Impact of energy literacy programs on solar adoption",
    "Energy poverty alleviation and renewable energy in rural communities",
    "Trust in institutions and energy transition outcomes",
    "Social norms and investment in renewable energy cooperatives",
    "Inclusion in renewable energy governance",
    "Participatory governance and equitable energy outcomes",

    # --- Slightly harder / edge (limited but useful) ---
    "Social norms and trust in community solar adoption",
    "Energy poverty and inclusive policy design in renewable energy programs"
]


def normalization_params(queries, parameters, retriever, relevance_profiler, topN):

    all_scores = []

    for i, q in enumerate(queries):
        retrieved_chunks = retriever.search(q, topk_faiss=30, topk_bm25=30)
        reranked_chunks = relevance_profiler.rerank(q, retrieved_chunks)
        scores = [chunk["final_score"] for chunk in reranked_chunks[:topN]]
        all_scores.extend(scores)

    all_scores = np.array(all_scores)
    b = np.median(all_scores)
    x90 = np.percentile(all_scores, 90)
    y_target = 0.9
    a = -np.log(1 / y_target - 1) / (x90 - b)

    parameters["normalization_params"] = {
        "a": a,
        "b": b,
        "std_global": all_scores.std()
    }

    return parameters


def contribution_quantiles(queries, parameters, retriever, relevance_profiler, topN, alpha):

    a,b, std_global = (parameters["normalization_params"]["a"], 
                       parameters["normalization_params"]["b"], 
                       parameters["normalization_params"]["std_global"])
    
    sum_contributions_list = []
    all_contributions = []

    for i, q in enumerate(queries):        
        retrieved_chunks = retriever.search(q, topk_faiss=30, topk_bm25=30)
        reranked_chunks = relevance_profiler.rerank(q, retrieved_chunks)
        
        relevant_chunks = reranked_chunks[:topN]
        scores = np.array([chunk["final_score"] for chunk in relevant_chunks])    
        mean_score, std_score = scores.mean(), scores.std()

        if std_score < 1e-6:
            z = np.zeros_like(scores)
        else:
            std = max(std_score, std_global*alpha) 
            z = (scores - mean_score) / std
        
        abs_relevance = np.array([semantic_norm(s, a, b) for s in scores])
        contributions = compute_contribution(abs_relevance, z)
        
        all_contributions.extend(contributions)
        sum_contributions_list.append(contributions.sum())

        parameters["contributions_per_query"] = {
            "q10": np.percentile(sum_contributions_list, 10),
            "q25": np.percentile(sum_contributions_list, 25),
            "q50": np.percentile(sum_contributions_list, 50),
            "q75": np.percentile(sum_contributions_list, 75),
            "q90": np.percentile(sum_contributions_list, 90),
            "observed_values": sum_contributions_list,
        }
        
        parameters["chunk_contributions"] = {
            "q10": np.percentile(all_contributions, 10),
            "q25": np.percentile(all_contributions, 25),
            "q50": np.percentile(all_contributions, 50),
        }

    return parameters


def effective_sources_quantiles(queries, parameters, retriever, relevance_profiler, topN, alpha):

    a,b, std_global, min_contribution = (parameters["normalization_params"]["a"], 
                                         parameters["normalization_params"]["b"], 
                                         parameters["normalization_params"]["std_global"],
                                         parameters["chunk_contributions"]["q25"])
    
    effective_sources_list = []

    for i, q in enumerate(queries):       
        retrieved_chunks = retriever.search(q, topk_faiss=30, topk_bm25=30)
        reranked_chunks = relevance_profiler.rerank(q, retrieved_chunks)
        
        relevant_chunks = reranked_chunks[:topN]
        paper_ids = [c["paper_id"] for c in relevant_chunks]
        scores = np.array([chunk["final_score"] for chunk in relevant_chunks])    
        mean_score, std_score, max_score = scores.mean(), scores.std(), scores.max()

        if std_score < 1e-6:
            z = np.zeros_like(scores)
        else:
            std = max(std_score, std_global*alpha) 
            z = (scores - mean_score) / std
        
        abs_relevance = np.array([semantic_norm(s, a, b) for s in scores])
        contributions = compute_contribution(abs_relevance, z)

        source_weights = {}
        for i, contrib in enumerate(contributions):
            if contrib > min_contribution:
                pid = paper_ids[i]
                source_weights[pid] = source_weights.get(pid, 0) + contrib

        effective_sources_list.append(len(source_weights))

    parameters["effective_sources_per_query"] = {
        "q10": np.percentile(effective_sources_list, 10),
        "q25": np.percentile(effective_sources_list, 25),
        "q50": np.percentile(effective_sources_list, 50),
        "q75": np.percentile(effective_sources_list, 75),
        "q90": np.percentile(effective_sources_list, 90),
        "observed_values": effective_sources_list,
    }

    return parameters


def run_tuning(config, chunks_path, index_path, params_path):
    
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)   

    index = load_faiss(index_path)

    if config.embedding == "hf":
        embedding_fn = HFEmbedding()
    else:
        embedding_fn = OpenAIEmbedding()

    semantic_retriever = SemanticRetriever(index, chunks, embedding_fn)
    bm25_retriever = BM25Retriever(chunks)
    retriever = HybridRetriever(semantic_retriever, bm25_retriever)
    relevance_profiler = RelevanceProfiler()
    
    parameters = {}
    parameters = normalization_params(QUERIES, parameters, retriever, relevance_profiler, config.topN)
    parameters = contribution_quantiles(QUERIES, parameters, retriever, relevance_profiler, config.topN, alpha=0.5)
    parameters = effective_sources_quantiles(QUERIES, parameters, retriever, relevance_profiler, config.topN, alpha=0.5)

    with open(params_path, "w", encoding="utf-8") as f:
        json.dump(parameters, f, indent=2)

    return parameters