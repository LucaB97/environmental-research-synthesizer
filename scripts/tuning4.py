from pathlib import Path
import json
import numpy as np
from utils.embeddings import OpenAIEmbedding
from utils.indexing import load_faiss
from utils.retriever import SemanticRetriever, BM25Retriever, HybridRetriever
from utils.cross_encoder import RelevanceProfiler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHUNKS_PATH = PROJECT_ROOT / "data" / "chunks_500t_100o.json"
FAISS_PATH = PROJECT_ROOT / "data" / "faiss_openai_500t_100o.index"
SEMANTIC_ALIGNMENT_PARAMS_PATH = PROJECT_ROOT / "data" / "semantic_alignment_params.json"
HITS_DISTRIBUTION_PARAMS_PATH = PROJECT_ROOT / "data" / "contributions_params.json"

def semantic_norm(score, a, b):
    import math
    return 1 / (1 + math.exp(-a * (score - b)))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def compute_contribution(abs_relevance, z, z_thr=0, kz=1.5):
    rel_relevance = sigmoid(kz * (z - z_thr))
    return abs_relevance * rel_relevance



with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
    chunks = json.load(f)

with open(SEMANTIC_ALIGNMENT_PARAMS_PATH, encoding="utf-8") as f:
    semantic_alignment_params = json.load(f)

with open(HITS_DISTRIBUTION_PARAMS_PATH, encoding="utf-8") as f:
    contributions_params = json.load(f)

# Parameters configuration
a,b = semantic_alignment_params["a"], semantic_alignment_params["b"]
std_global = semantic_alignment_params["std_global"]
alpha = 0.5
min_contribution_threshold = contributions_params["chunk_contributions"]["q25"]


index = load_faiss(FAISS_PATH)
embedding_fn = OpenAIEmbedding()
semantic_retriever = SemanticRetriever(index, chunks, embedding_fn)
bm25_retriever = BM25Retriever(chunks)
retriever = HybridRetriever(semantic_retriever, bm25_retriever)
relevance_profiler = RelevanceProfiler()

queries = [
    "Social acceptance of community wind projects in areas with low electricity demand",
    "Effects of minor policy changes on household energy literacy programs",
    "Participation of informal local groups in renewable energy initiatives",
    "Influence of neighborhood social networks on adoption of solar panels in small towns",
    "Energy poverty considerations in communities with high off-grid adoption",
    "Challenges of promoting energy literacy in transient communities",
    "Influence of minor local regulations on small-scale renewable energy projects",
    "Participation of volunteers in energy transition awareness campaigns in sparsely populated regions",

    "Household affordability and adoption of solar energy in low-income communities",
    "Community engagement strategies in renewable energy projects",
    "Public attitudes toward wind farms in semi-urban areas",
    "Energy literacy programs for promoting energy efficiency",
    "Distributional effects of subsidies for residential solar panels",
    "Trust in local authorities influencing renewable energy adoption",
    "Equity in decision-making for local renewable energy projects",
    "Economic well-being effects of community-owned renewable energy initiatives",
    "Public perception of renewable energy policies in minority communities",
    "Participation of local stakeholders in planning renewable energy infrastructure",
    "Community engagement in microgrid implementation projects",
    "Impact of gender dynamics on participation in renewable energy cooperatives",
    "Effect of municipal communication strategies on local renewable energy awareness",
    
    "Effectiveness of participatory decision-making on social acceptance of onshore wind projects",
    "Impact of targeted energy literacy programs on adoption of household solar PV in disadvantaged neighborhoods",
    "Relationship between energy poverty alleviation and distributed renewable energy schemes in rural communities",
    "Role of trust in institutions on equitable energy transition outcomes",
    "Influence of social norms on community investment in renewable energy cooperatives",
    "Governance frameworks supporting minority inclusion in renewable energy projects",
    "Effectiveness of participatory governance in achieving equitable renewable energy outcomes",
    "Role of social norms and trust in accelerating community-owned solar adoption",
    "Relationship between energy poverty alleviation and inclusive policy design in rural renewable energy programs"
    ]

max_score_list = []
mean_score_list = []
distinct_effective_sources_list = []

for i, q in enumerate(queries):
    print(f"{i+1}/{len(queries)}")
    
    retrieved_chunks = retriever.search(q, topk_faiss=30, topk_bm25=30)
    reranked_chunks = relevance_profiler.rerank(q, retrieved_chunks)
    
    relevant_chunks = reranked_chunks[:15]
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
        if contrib > min_contribution_threshold:
            pid = paper_ids[i]
            source_weights[pid] = source_weights.get(pid, 0) + contrib

    distinct_effective_sources_list.append(len(source_weights))


contributions_params["distinct_effective_sources_per_query"] = {
    "q10": np.percentile(distinct_effective_sources_list, 10),
    "q25": np.percentile(distinct_effective_sources_list, 25),
    "q50": np.percentile(distinct_effective_sources_list, 50),
    "q75": np.percentile(distinct_effective_sources_list, 75),
    "q90": np.percentile(distinct_effective_sources_list, 90),
    "observed_values": distinct_effective_sources_list,
}

# Choose a location in your project
params_path = PROJECT_ROOT / "data" / "contributions_params.json"
params_path.parent.mkdir(exist_ok=True)

with open(params_path, "w", encoding="utf-8") as f:
    json.dump(contributions_params, f, indent=2)

print(f"Saved parameters to {params_path}")