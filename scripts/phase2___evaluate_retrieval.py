import json
import numpy as np
from utils.embeddings import hf_embedding
from utils.indexing import load_faiss
from utils.retriever import SemanticRetriever
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHUNKS_PATH = PROJECT_ROOT / "data" / "chunks.json"
FAISS_PATH = PROJECT_ROOT / "data" / "faiss.index"

with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
    chunks = json.load(f)

index = load_faiss(FAISS_PATH)

retriever = SemanticRetriever(index, chunks, hf_embedding)

queries = [
    "How are local communities affected by large renewable energy projects?",
    "What are the effects of renewable energy adoption on local employment?",
    "How do renewable energy policies impact marginalized communities?",
]

for q in queries:
    print("\n" + "="*80)
    print("QUERY:", q)
    results = retriever.search(q, top_k=5)
    retriever.display(results)