import json
import numpy as np
from utils.embeddings import hf_embedding
from utils.indexing import load_faiss
from utils.retriever import SemanticRetriever
# from utils.synthesizer import OpenAIClient, Synthesizer
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHUNKS_PATH = PROJECT_ROOT / "data" / "chunks.json"
FAISS_PATH = PROJECT_ROOT / "data" / "faiss.index"

with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
    chunks = json.load(f)

index = load_faiss(FAISS_PATH)

retriever = SemanticRetriever(index, chunks, hf_embedding)
# llm = OpenAIClient()
# synthesizer = Synthesizer(llm)

query = "How are local communities affected by large renewable energy projects?"
# query = "How difficult is it to learn how to play chess?"

print("\n" + "="*80)
print("QUERY:", query)
print("Searching...")
results = retriever.search(query, top_k=5)
print("Displaying results...")
retriever.display(results)
# print("Synthesizing...")
# answer = synthesizer.synthesize(query, results)
# print("\n" + answer)

paper_lookup = {
    c["paper_id"]: c
    for c in results
}

print(paper_lookup)


